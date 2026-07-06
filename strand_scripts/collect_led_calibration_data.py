#!/usr/bin/env python3
"""Collect an LED calibration dataset: braid 3D positions + aux-camera 2D pixels.

Launches a local strand-cam on the auxiliary camera (2D LED point detection,
flytrax CSV output) while streaming 3D tracking from a braid server on another
machine. Move an LED around the volume seen by both; on stop, the two streams
are merged into per-frame 3D<->2D correspondences for camera calibration.

Standalone (no ROS). Usage:

    python3 collect_led_calibration_data.py                       # default toml
    python3 collect_led_calibration_data.py --config my.toml --duration 120
    python3 collect_led_calibration_data.py --merge-only DATASET_DIR

Flow: strand-cam launches and the browser UI URL is printed -- watch the live
image there and confirm the LED is detected (tweak the toml's
point_detection_config or the UI sliders). Press Enter to start recording,
Ctrl-C to stop. Each session creates its own dataset directory.

Clock domains: braid rows carry triggerbox-clock timestamps from the braid PC;
flytrax rows carry this PC's clock. The merge step (led_calibration_merge.py)
estimates the offset from the data itself, so unsynchronized PC clocks are OK.
"""

import os
import sys
import glob
import json
import time
import shutil
import signal
import atexit
import argparse
import tomllib
import threading
import subprocess

import requests
import yaml

import led_calibration_merge

BRAID_DATA_PREFIX = 'data: '

BRAID_CSV_COLUMNS = ['obj_id', 'frame', 'trigger_timestamp', 'receive_time_unix',
                     'x', 'y', 'z', 'xvel', 'yvel', 'zvel',
                     'P00', 'P11', 'P22', 'P33', 'P44', 'P55']

CLOCK_NOTE = ('three clock domains: trigger_timestamp = braid triggerbox clock '
              '(braid PC); receive_time_unix = this PC at row arrival; flytrax '
              'created_at + time_microseconds = this PC (camera driver). The '
              'alignment block reports the estimated offset between the braid '
              'and local clocks; frame-locked matching is exact regardless.')


def load_config(path):
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    cameras = config.get('cameras', [])
    if len(cameras) != 1:
        raise ValueError('config must contain exactly one [[cameras]] entry, '
                         'found %d' % len(cameras))
    return config, open(path).read()


class StrandCamManager:
    '''Launch and control a strand-cam instance doing 2D point detection.'''

    def __init__(self, config, dataset_dir, attach=False):
        cam = config['cameras'][0]
        cfg = config['calibration_data_collection']
        self.camera_name = cam['name']
        self.camera_settings_filename = cam.get('camera_settings_filename', '')
        self.detection_config = cam.get('point_detection_config', {})
        self.executable = cfg.get('strand_cam_executable', 'strand-cam')
        self.backend = cfg.get('camera_backend', 'pylon')
        self.addr = cfg.get('http_server_addr', '127.0.0.1:3441')
        self.trusted_network = cfg.get('trusted_network', '127.0.0.1/32')
        self.dataset_dir = dataset_dir
        self.staging_dir = os.path.join(dataset_dir, '.flytrax-staging')
        self.attach = attach

        self.proc = None
        self.version = ''
        self.latest_state = {}
        self.lock_state = threading.Lock()
        self.ready = threading.Event()
        self._shutting_down = False
        self.shutdown_done = False

    # -- lifecycle ---------------------------------------------------------

    def launch(self):
        os.makedirs(self.staging_dir, exist_ok=True)
        if self.attach:
            print('attaching to already-running strand-cam at %s' % self.addr)
            return
        try:
            r = subprocess.run([self.executable, '--version'],
                               capture_output=True, text=True, timeout=10)
            self.version = r.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        argv = [self.executable, '--no-browser',
                '--camera-name', self.camera_name,
                '--camera-backend', self.backend,
                '--http-server-addr', self.addr,
                '--trusted-network', self.trusted_network,
                '--csv-save-dir', self.staging_dir,
                '--data-dir', self.staging_dir,
                ]
        if self.camera_settings_filename != '':
            argv += ['--camera-settings-filename',
                     os.path.expanduser(self.camera_settings_filename)]
        print('launching: %s' % ' '.join(argv))
        self._log = open(os.path.join(self.dataset_dir, 'strand-cam.log'), 'a')
        self.proc = subprocess.Popen(argv, stdout=self._log,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     cwd=self.staging_dir)

    def wait_for_http(self, timeout=30.0):
        url = 'http://%s/' % self.addr
        t_start = time.time()
        while True:
            try:
                requests.get(url, timeout=2).raise_for_status()
                print('connected to strand-cam at %s' % url)
                return
            except requests.RequestException as err:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError('strand-cam exited during startup (code %s), '
                                       'see strand-cam.log' % self.proc.returncode)
                if time.time() - t_start > timeout:
                    raise RuntimeError('could not reach strand-cam at %s: %s'
                                       % (url, err))
                print('waiting for strand-cam at %s ...' % url)
                time.sleep(1.0)

    def start_sse_thread(self):
        threading.Thread(target=self._sse_thread, daemon=True).start()

    def _sse_thread(self):
        url = 'http://%s/strand-cam-events' % self.addr
        while not self._shutting_down:
            try:
                with requests.get(url, stream=True, timeout=(5, 60),
                                  headers={'Accept': 'text/event-stream'}) as r:
                    r.raise_for_status()
                    event_name = None
                    for line in r.iter_lines(decode_unicode=True):
                        if self._shutting_down:
                            return
                        if not line:
                            event_name = None
                            continue
                        if line.startswith('event: '):
                            event_name = line[len('event: '):].strip()
                        elif line.startswith('data: ') and event_name == 'strand-cam':
                            state = json.loads(line[len('data: '):])
                            with self.lock_state:
                                self.latest_state = state
                            self.ready.set()
            except (requests.RequestException, json.JSONDecodeError) as err:
                if self._shutting_down:
                    return
                print('WARNING: strand-cam event stream dropped (%s), reconnecting'
                      % err)
                time.sleep(1.0)

    def get_state(self, key, default=None):
        with self.lock_state:
            return self.latest_state.get(key, default)

    def alive(self):
        return self.attach or (self.proc is not None and self.proc.poll() is None)

    def shutdown(self):
        if self.shutdown_done:
            return
        self.shutdown_done = True
        self._shutting_down = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # -- control -----------------------------------------------------------

    def post_callback(self, cam_arg):
        # serde externally-tagged CamArg wrapped in CallbackType::ToCamera
        self.post_raw({'ToCamera': cam_arg})

    def post_raw(self, payload):
        r = requests.post('http://%s/callback' % self.addr, json=payload, timeout=5)
        if not r.ok:
            # the response body carries the actual serde/validation message
            raise requests.HTTPError('%d %s from strand-cam /callback: %s'
                                     % (r.status_code, r.reason, r.text.strip()),
                                     response=r)

    def push_detection_config(self):
        '''Merge the toml detection table over the live config and send it.'''
        if not self.detection_config:
            print('no [cameras.point_detection_config] in toml; using '
                  'strand-cam current settings')
            return
        live = self.get_state('im_pt_detect_cfg') or {}
        merged = dict(live)
        merged.update(self.detection_config)
        self.post_callback({'SetObjDetectionConfig': yaml.safe_dump(merged)})
        print('detection config sent (polarity=%s, diff_threshold=%s, '
              'max_num_points=%s)' % (merged.get('polarity'),
                                      merged.get('diff_threshold'),
                                      merged.get('max_num_points')))

    def take_background(self):
        # CallbackType unit variant: bare string, NOT ToCamera-wrapped
        self.post_raw('TakeCurrentImageAsBackground')

    def enable_detection(self):
        self.post_callback({'SetIsDoingObjDetection': True})

    def start_csv(self):
        self.post_callback({'SetIsSavingObjDetectionCsv': {'Saving': None}})

    def stop_csv(self):
        self.post_callback({'SetIsSavingObjDetectionCsv': 'NotSaving'})

    def save_reference_image(self, path):
        '''Record a very short MP4 and extract one frame as a reference image.

        (strand-cam has no plain HTTP snapshot route; the live view goes over
        an authenticated stream, so a mini-recording is the simplest way.)
        '''
        if shutil.which('ffmpeg') is None:
            print('WARNING: ffmpeg not found, skipping reference image')
            return
        try:
            self.post_callback({'SetIsRecordingMp4': True})
            time.sleep(0.6)
            self.post_callback({'SetIsRecordingMp4': False})
        except requests.RequestException as err:
            print('WARNING: reference-image recording failed: %s' % err)
            return
        mp4 = None
        t_start = time.time()
        last_size, stable = -1, 0
        while time.time() - t_start < 10:
            candidates = glob.glob(os.path.join(self.staging_dir, '*.mp4'))
            if candidates:
                mp4 = max(candidates, key=os.path.getmtime)
                size = os.path.getsize(mp4)
                stable = stable + 1 if size == last_size else 0
                last_size = size
                if stable >= 2:
                    break
            time.sleep(0.5)
        if mp4 is None:
            print('WARNING: no reference mp4 appeared, skipping reference image')
            return
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4,
                            '-frames:v', '1', path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print('WARNING: ffmpeg frame extraction failed: %s' % r.stderr.strip()[:200])
        for f in glob.glob(os.path.join(self.staging_dir, '*.mp4')):
            os.remove(f)

    def find_flytrax_csv(self, timeout=15.0):
        '''Newest CSV in the staging dir once its size is stable.'''
        t_start = time.time()
        last_size, stable, path = -1, 0, None
        while time.time() - t_start < timeout:
            candidates = glob.glob(os.path.join(self.staging_dir, '*.csv'))
            if not candidates:
                time.sleep(0.5)
                continue
            path = max(candidates, key=os.path.getmtime)
            size = os.path.getsize(path)
            stable = stable + 1 if size == last_size else 0
            last_size = size
            if stable >= 2:
                return path
            time.sleep(0.5)
        return path


class BraidStream(threading.Thread):
    '''Stream braid Update rows over SSE; append to CSV while recording.'''

    def __init__(self, braid_url):
        super().__init__(daemon=True)
        self.braid_url = braid_url
        self.recording = threading.Event()
        self.stop_event = threading.Event()
        self.n_rows_seen = 0        # since connect, regardless of recording
        self.n_rows_recorded = 0
        self.n_reconnects_while_recording = 0
        self.last_update_walltime = None
        self._csv_file = None
        self._lock = threading.Lock()

    def start_recording(self, csv_path):
        with self._lock:
            self._csv_file = open(csv_path, 'w', buffering=1)
            self._csv_file.write(','.join(BRAID_CSV_COLUMNS) + '\n')
        self.recording.set()

    def stop_recording(self):
        self.recording.clear()
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None

    def run(self):
        backoff = 1.0
        events_url = self.braid_url.rstrip('/') + '/events'
        first = True
        while not self.stop_event.is_set():
            try:
                with requests.get(events_url, stream=True, timeout=(5, 60),
                                  headers={'Accept': 'text/event-stream'}) as r:
                    r.raise_for_status()
                    if not first and self.recording.is_set():
                        self.n_reconnects_while_recording += 1
                    first = False
                    backoff = 1.0
                    for line in r.iter_lines(decode_unicode=True):
                        if self.stop_event.is_set():
                            return
                        if not line or not line.startswith(BRAID_DATA_PREFIX):
                            continue
                        self._handle(line[len(BRAID_DATA_PREFIX):])
            except (requests.RequestException, json.JSONDecodeError) as err:
                if self.stop_event.is_set():
                    return
                print('WARNING: braid stream dropped (%s), reconnecting in %.0f s'
                      % (err, backoff))
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 10.0)

    def _handle(self, buf):
        data = json.loads(buf)
        try:
            u = data['msg']['Update']
        except (KeyError, TypeError):
            return  # Birth/Death/other
        timestamp = data.get('trigger_timestamp')
        if timestamp is None:
            return
        self.n_rows_seen += 1
        self.last_update_walltime = time.time()
        if not self.recording.is_set():
            return
        row = [u['obj_id'], u['frame'], timestamp, time.time(),
               u['x'], u['y'], u['z'], u['xvel'], u['yvel'], u['zvel'],
               u['P00'], u['P11'], u['P22'], u['P33'], u['P44'], u['P55']]
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.write(','.join(repr(v) for v in row) + '\n')
                self.n_rows_recorded += 1


def run_session(config, config_text, args):
    cfg = config['calibration_data_collection']
    output_base = os.path.expanduser(args.output_dir or
                                     cfg.get('output_base_dir',
                                             '~/Desktop/led_calibration_datasets'))
    dataset_dir = os.path.join(output_base,
                               time.strftime('led_calib_%Y%m%d_%H%M%S'))
    os.makedirs(dataset_dir, exist_ok=True)
    print('dataset directory: %s' % dataset_dir)

    strand = StrandCamManager(config, dataset_dir, attach=args.attach)
    braid = BraidStream(cfg.get('braid_url', 'http://134.197.37.229:8397/'))
    started_at = None
    stopped_at = None
    recorded_anything = False

    def shutdown_all():
        braid.stop_event.set()
        braid.stop_recording()
        strand.shutdown()

    atexit.register(shutdown_all)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))

    try:
        # 1. camera up + detection configured
        strand.launch()
        strand.wait_for_http()
        strand.start_sse_thread()
        if not strand.ready.wait(timeout=15):
            raise RuntimeError('no state received from strand-cam event stream')
        strand.push_detection_config()
        time.sleep(0.5)
        strand.take_background()
        strand.enable_detection()
        print('\nstrand-cam live view: http://%s/  <- verify the LED is '
              'detected (green marker)\n' % strand.addr)

        # 2. braid stream up
        braid.start()
        t_wait = time.time()
        while braid.n_rows_seen == 0:
            if time.time() - t_wait > 5:
                print('no braid data yet -- is the LED (or anything) being '
                      'tracked in the braid volume?')
                t_wait = time.time()
            if not strand.alive():
                raise RuntimeError('strand-cam exited, see strand-cam.log')
            time.sleep(0.5)
        print('braid data flowing (%d updates seen)' % braid.n_rows_seen)

        # 3. start recording
        if not args.start_immediately and args.duration is None:
            print('press Enter to START recording ...')
            try:
                input()
            except EOFError:
                print('(no interactive stdin, starting immediately)')
        braid.start_recording(os.path.join(dataset_dir, 'braid_3d.csv'))
        strand.start_csv()
        strand.save_reference_image(os.path.join(dataset_dir, 'reference.png'))
        started_at = time.time()
        recorded_anything = True
        print('RECORDING -- move the LED around the shared volume, varying '
              'speed and covering the field of view. Ctrl-C to stop.')

        # 4. record until Ctrl-C / duration
        def stop_check():
            if not strand.alive():
                raise RuntimeError('strand-cam exited mid-recording')
        try:
            t_status = 0.0
            t_end = time.time() + args.duration if args.duration else None
            while t_end is None or time.time() < t_end:
                stop_check()
                if time.time() - t_status > 2.0:
                    t_status = time.time()
                    print('\r  braid rows: %-8d camera fps: %-8.1f elapsed: %5.0f s '
                          % (braid.n_rows_recorded,
                             strand.get_state('measured_fps', float('nan')),
                             time.time() - started_at), end='', flush=True)
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        print()

    except KeyboardInterrupt:
        print('\ninterrupted before recording started')
    finally:
        stopped_at = time.time()
        try:
            if recorded_anything:
                strand.stop_csv()
                time.sleep(0.5)
        except requests.RequestException:
            pass
        braid.stop_event.set()
        braid.stop_recording()

        if not recorded_anything:
            # no `return` here: an exception from the try block (if any) must
            # keep propagating after cleanup so the user sees the real error
            strand.shutdown()
            if not os.listdir(dataset_dir):
                os.rmdir(dataset_dir)
            elif not glob.glob(os.path.join(dataset_dir, '*.csv')):
                print('nothing recorded; leaving %s (contains logs only)'
                      % dataset_dir)

    if not recorded_anything:
        return

    # 5. file the flytrax csv + merge + metadata
    flytrax_path = strand.find_flytrax_csv()
    strand.shutdown()
    if flytrax_path is None:
        print('ERROR: no flytrax CSV found in %s' % strand.staging_dir)
        flytrax_name = ''
    else:
        flytrax_path = shutil.move(flytrax_path, dataset_dir)
        flytrax_name = os.path.basename(flytrax_path)

    measured_fps = strand.get_state('measured_fps')
    metadata = {
        'camera_name': strand.camera_name,
        'strand_cam_version': strand.version,
        'braid_url': braid.braid_url,
        'image_width': strand.get_state('image_width'),
        'image_height': strand.get_state('image_height'),
        'fps_configured': cfg.get('fps', 100.0),
        'measured_fps': measured_fps,
        'started_at_unix': started_at,
        'stopped_at_unix': stopped_at,
        'started_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S%z',
                                        time.localtime(started_at)),
        'duration_s': round(stopped_at - started_at, 1),
        'flytrax_csv': flytrax_name,
        'n_braid_rows': braid.n_rows_recorded,
        'braid_reconnects_while_recording': braid.n_reconnects_while_recording,
        'detection_config_sent': strand.detection_config,
        'clock_note': CLOCK_NOTE,
        'config_toml': config_text,
        'alignment': {'merge_error': ''},
    }

    try:
        report = led_calibration_merge.run_merge_on_dataset(
            dataset_dir, fps_hint=cfg.get('fps', 100.0),
            measured_fps=measured_fps, params=cfg)
        report['merge_error'] = ''
        metadata['alignment'] = _yaml_safe(report)
        print('merged: %d correspondences (method: %s, clock offset: %.4f s, '
              'frame offset: %s, residual rms: %.3f ms)'
              % (report['n_matched'], report['match_method'],
                 report['offset_seconds'], report['frame_offset'],
                 report['residual_rms_ms']))
    except Exception as err:
        print('WARNING: merge failed (%s). Raw data is intact; re-run with:\n'
              '  python3 %s --merge-only %s'
              % (err, os.path.abspath(__file__), dataset_dir))
        metadata['alignment'] = {'merge_error': str(err)}

    with open(os.path.join(dataset_dir, 'metadata.yaml'), 'w') as f:
        yaml.safe_dump(metadata, f, sort_keys=False, default_flow_style=False)
    print('dataset complete: %s' % dataset_dir)


def _yaml_safe(d):
    '''Make a report dict yaml.safe_dump-able (numpy scalars -> python).'''
    out = {}
    for k, v in d.items():
        if hasattr(v, 'item'):
            v = v.item()
        elif isinstance(v, dict):
            v = _yaml_safe(v)
        out[k] = v
    return out


def merge_only(dataset_dir, config):
    cfg = config['calibration_data_collection']
    report = led_calibration_merge.run_merge_on_dataset(
        dataset_dir, fps_hint=cfg.get('fps', 100.0), params=cfg)
    meta_path = os.path.join(dataset_dir, 'metadata.yaml')
    metadata = {}
    if os.path.exists(meta_path):
        metadata = yaml.safe_load(open(meta_path)) or {}
    report['merge_error'] = ''
    metadata['alignment'] = _yaml_safe(report)
    with open(meta_path, 'w') as f:
        yaml.safe_dump(metadata, f, sort_keys=False, default_flow_style=False)
    for k in sorted(report):
        print('%s: %s' % (k, report[k]))


def main():
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'led_calibration.toml')
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=default_config,
                        help='braid-style toml (see led_calibration.toml)')
    parser.add_argument('--output-dir', default='',
                        help='override output_base_dir from the config')
    parser.add_argument('--duration', type=float, default=None,
                        help='record for N seconds instead of waiting for Ctrl-C')
    parser.add_argument('--start-immediately', action='store_true',
                        help='skip the press-Enter prompt before recording')
    parser.add_argument('--attach', action='store_true',
                        help='use an already-running strand-cam at the configured '
                             'http_server_addr instead of launching one')
    parser.add_argument('--merge-only', metavar='DATASET_DIR', default='',
                        help='re-run the merge on an existing dataset and exit')
    args = parser.parse_args()

    config, config_text = load_config(args.config)
    if args.merge_only:
        merge_only(os.path.expanduser(args.merge_only), config)
        return
    run_session(config, config_text, args)


if __name__ == '__main__':
    main()
