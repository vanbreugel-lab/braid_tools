#!/usr/bin/env python3
"""Save short strand-cam video clips + braid tracking metadata on a ROS trigger.

Launches strand-cam as a managed subprocess, arms its post-trigger circular
frame buffer, and listens for braid_tools/BraidTrigger messages. On a trigger,
strand-cam flushes the pre-trigger buffer into a new MP4 and keeps recording
for post_trigger_seconds; the node then files the MP4 into a per-event
directory together with a metadata.hdf5 (trigger info + the triggered
obj_id's tracking rows from /flydra_mainbrain/super_packets) and, optionally,
a per-frame timestamp CSV extracted with show-timestamps.

Clock domains: the tracking window is evaluated on acquire_stamp (the braid
triggerbox clock, unix epoch based -- the same clock as the MISP timestamps
strand-cam embeds in the MP4 when run with --braid-url). The trigger stamp is
assumed to be in the same unix domain; both the trigger stamp and the node's
receive time are stored raw in metadata.hdf5 for post-hoc correction.

Configuration comes from a braid-style .toml (see video_config/). The runtime
knobs output_base_dir / pre_trigger_seconds / post_trigger_seconds can be
overridden with ROS parameters, e.g.:
    --ros-args -p post_trigger_seconds:=10.0
"""

import sys
import os
import math
import json
import glob
import time
import shutil
import signal
import atexit
import argparse
import tomllib
import threading
import subprocess
from collections import deque

import numpy as np
import h5py
import requests

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.utilities import remove_ros_args
from ament_index_python.packages import get_package_share_directory

from braid_tools.msg import FlydraMainbrainSuperPacket, BraidTrigger

CLOCK_NOTE = ('tracking window evaluated on acquire_stamp (braid triggerbox '
              'clock, unix epoch); trigger stamp assumed same domain; MP4 MISP '
              'timestamps share this clock when strand-cam runs with --braid-url')

# same fields/dtypes as braid_save_data_to_hdf5.py so downstream loaders are shared
TRACKING_DTYPE = [('frame_number', int),
                  ('reconstruction_stamp_secs', int),
                  ('reconstruction_stamp_nsecs', int),
                  ('acquire_stamp_secs', int),
                  ('acquire_stamp_nsecs', int),
                  ('obj_id', int),
                  ('position_x', float),
                  ('position_y', float),
                  ('position_z', float),
                  ('velocity_x', float),
                  ('velocity_y', float),
                  ('velocity_z', float),
                  ('posvel_covariance_d1', float),
                  ('posvel_covariance_d2', float),
                  ('posvel_covariance_d3', float),
                  ('posvel_covariance_d4', float),
                  ('posvel_covariance_d5', float),
                  ('posvel_covariance_d6', float),
                  ]


def time_msg_to_float(t):
    '''Convert a builtin_interfaces/Time message to a float unix timestamp.'''
    return t.sec + t.nanosec * 1e-9


class ActiveEvent:
    def __init__(self, obj_id, trigger_time, receive_time, metadata, event_dir,
                 fps_measured):
        self.obj_id = obj_id
        self.trigger_time = trigger_time
        self.receive_time = receive_time
        self.metadata = metadata
        self.event_dir = event_dir
        self.fps_measured = fps_measured


class TriggeredVideoSaver(Node):
    def __init__(self, config_path, output_dir_override='', attach=False):
        super().__init__('braid_triggered_video_saver')

        with open(config_path, 'rb') as f:
            self.config_toml_text = open(config_path, 'r').read()
            config = tomllib.load(f)

        cameras = config.get('cameras', [])
        if len(cameras) != 1:
            raise ValueError('config must contain exactly one [[cameras]] entry, '
                             'found %d' % len(cameras))
        self.camera_name = cameras[0]['name']
        self.camera_settings_filename = cameras[0].get('camera_settings_filename', '')

        cfg = config['triggered_video_saver']
        self.strand_cam_executable = cfg.get('strand_cam_executable', 'strand-cam')
        self.camera_backend = cfg.get('camera_backend', 'pylon')
        self.http_server_addr = cfg.get('http_server_addr', '127.0.0.1:3440')
        self.trusted_network = cfg.get('trusted_network', '127.0.0.1/32')
        self.braid_url = cfg.get('braid_url', '')
        self.fps = float(cfg.get('fps', 100.0))
        self.trigger_topic = cfg.get('trigger_topic', 'braid_trigger')
        self.run_show_timestamps = cfg.get('run_show_timestamps', True)
        self.mp4_bitrate = cfg.get('mp4_bitrate', '')        # e.g. "Bitrate5000"
        self.mp4_codec = cfg.get('mp4_codec', '')            # e.g. "H264Nvenc"
        self.recording_fps = cfg.get('recording_fps', '')    # e.g. "Fps100"
        self.mp4_max_framerate = cfg.get('mp4_max_framerate', '')  # e.g. "Fps30"
        self.pixel_format = cfg.get('pixel_format', '')      # e.g. "RGB8"; strand-cam
        # rejects --pixel-format together with --braid-url -- when braid-synced,
        # set PixelFormat in the camera settings .pfs file instead

        # toml values are the parameter defaults; --ros-args -p ... overrides
        self.declare_parameter('output_base_dir',
                               cfg.get('output_base_dir', '~/Desktop/triggered_videos'))
        self.declare_parameter('pre_trigger_seconds',
                               float(cfg.get('pre_trigger_seconds', 1.0)))
        self.declare_parameter('post_trigger_seconds',
                               float(cfg.get('post_trigger_seconds', 5.0)))
        if output_dir_override != '':
            output_base_dir = output_dir_override
        else:
            output_base_dir = self.get_parameter('output_base_dir').value
        self.output_base_dir = os.path.expanduser(output_base_dir)
        self.pre_trigger_seconds = self.get_parameter('pre_trigger_seconds').value
        self.post_trigger_seconds = self.get_parameter('post_trigger_seconds').value
        self.post_trigger_buffer_frames = math.ceil(self.pre_trigger_seconds * self.fps)

        self.events_dir = os.path.join(self.output_base_dir, 'events')
        self.staging_dir = os.path.join(self.output_base_dir, '.staging')
        os.makedirs(self.events_dir, exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        self._file_orphaned_staging_mp4s()

        self.tracking_buffer = deque()  # (acquire_time_float, obj_id, row_tuple)
        self.lock_buffer = threading.Lock()
        self.lock_event = threading.Lock()
        self.lock_state = threading.Lock()
        self.active_event = None
        self.latest_state = {}
        self.ready = threading.Event()
        self.shutdown_done = False
        self._shutting_down = False
        self._stop_timer = None
        self._finalize_thread = None
        self._fps_mismatch_warned = False

        self.proc = None
        self.strand_cam_version = ''
        if not attach:
            self._launch_strand_cam()
        self._wait_for_http()

        self._sse_thread_handle = threading.Thread(target=self._sse_thread, daemon=True)
        self._sse_thread_handle.start()

        self._post_callback({'SetPostTriggerBufferSize': self.post_trigger_buffer_frames})
        self.get_logger().info('post-trigger buffer armed: %d frames (%.2f s at %.1f fps)'
                               % (self.post_trigger_buffer_frames,
                                  self.pre_trigger_seconds, self.fps))
        if self.mp4_codec != '':
            self._post_callback({'SetMp4Codec': self.mp4_codec})
        if self.mp4_bitrate != '':
            self._post_callback({'SetMp4Bitrate': self.mp4_bitrate})
        if self.recording_fps != '':
            self._post_callback({'SetRecordingFps': self.recording_fps})
        if self.mp4_max_framerate != '':
            self._post_callback({'SetMp4MaxFramerate': self.mp4_max_framerate})

        self.subTrackedObjects = self.create_subscription(FlydraMainbrainSuperPacket,
            '/flydra_mainbrain/super_packets', self.tracking_callback, 500)
        self.subTrigger = self.create_subscription(BraidTrigger,
            self.trigger_topic, self.trigger_callback, 10)

        self.housekeeping_timer = self.create_timer(0.5, self._housekeeping)
        self.get_logger().info('saving triggered videos to %s (trigger topic: %s)'
                               % (self.output_base_dir, self.trigger_topic))

    # ------------------------------------------------------------ strand-cam

    def _file_orphaned_staging_mp4s(self):
        leftovers = glob.glob(os.path.join(self.staging_dir, '*.mp4'))
        if leftovers:
            orphan_dir = os.path.join(self.output_base_dir, 'orphaned')
            os.makedirs(orphan_dir, exist_ok=True)
            for f in leftovers:
                shutil.move(f, orphan_dir)
            self.get_logger().warn('moved %d leftover mp4(s) from a previous run to %s'
                                   % (len(leftovers), orphan_dir))

    def _launch_strand_cam(self):
        try:
            r = subprocess.run([self.strand_cam_executable, '--version'],
                               capture_output=True, text=True, timeout=10)
            self.strand_cam_version = r.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass

        argv = [self.strand_cam_executable,
                '--no-browser',
                '--camera-name', self.camera_name,
                '--camera-backend', self.camera_backend,
                '--http-server-addr', self.http_server_addr,
                '--trusted-network', self.trusted_network,
                '--data-dir', self.staging_dir,
                ]
        if self.camera_settings_filename != '':
            argv += ['--camera-settings-filename',
                     os.path.expanduser(self.camera_settings_filename)]
        if self.braid_url != '':
            argv += ['--braid-url', self.braid_url]
        if self.pixel_format != '':
            argv += ['--pixel-format', self.pixel_format]

        self.get_logger().info('launching: %s' % ' '.join(argv))
        self._strand_cam_log = open(os.path.join(self.output_base_dir, 'strand-cam.log'), 'a')
        self.proc = subprocess.Popen(argv,
                                     stdout=self._strand_cam_log,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     cwd=self.staging_dir)

    def _wait_for_http(self, timeout=30.0):
        url = 'http://%s/' % self.http_server_addr
        t_start = time.time()
        while True:
            try:
                r = requests.get(url, timeout=2)
                r.raise_for_status()
                self.get_logger().info('connected to strand-cam at %s' % url)
                return
            except requests.RequestException as err:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError('strand-cam exited during startup (code %s), '
                                       'see strand-cam.log' % self.proc.returncode)
                if time.time() - t_start > timeout:
                    raise RuntimeError('could not reach strand-cam at %s: %s' % (url, err))
                self.get_logger().info('waiting for strand-cam at %s' % url)
                time.sleep(1.0)

    def _post_callback(self, cam_arg):
        # cam_arg is a serde externally-tagged CamArg: a dict like
        # {"SetPostTriggerBufferSize": 100} or the bare string "PostTrigger"
        url = 'http://%s/callback' % self.http_server_addr
        r = requests.post(url, json={'ToCamera': cam_arg}, timeout=5)
        r.raise_for_status()

    def _sse_thread(self):
        url = 'http://%s/strand-cam-events' % self.http_server_addr
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
                self.get_logger().warn('strand-cam event stream dropped (%s), reconnecting' % err)
                time.sleep(1.0)

    def _get_state(self, key, default=None):
        with self.lock_state:
            return self.latest_state.get(key, default)

    # ------------------------------------------------------------- callbacks

    def tracking_callback(self, super_packet):
        # generous retention: finalizing an event (waiting on the mp4) can take
        # tens of seconds, and the rows must survive until they are snapshotted
        window = self.pre_trigger_seconds + self.post_trigger_seconds + 30.0
        with self.lock_buffer:
            for packet in super_packet.packets:
                acquire_time = time_msg_to_float(packet.acquire_stamp)
                for obj in packet.objects:
                    row = (packet.framenumber,
                           packet.reconstruction_stamp.sec,
                           packet.reconstruction_stamp.nanosec,
                           packet.acquire_stamp.sec,
                           packet.acquire_stamp.nanosec,
                           obj.obj_id,
                           obj.position.x,
                           obj.position.y,
                           obj.position.z,
                           obj.velocity.x,
                           obj.velocity.y,
                           obj.velocity.z,
                           obj.posvel_covariance_diagonal[0],
                           obj.posvel_covariance_diagonal[1],
                           obj.posvel_covariance_diagonal[2],
                           obj.posvel_covariance_diagonal[3],
                           obj.posvel_covariance_diagonal[4],
                           obj.posvel_covariance_diagonal[5],
                           )
                    self.tracking_buffer.append((acquire_time, obj.obj_id, row))
                while (len(self.tracking_buffer) > 0 and
                       acquire_time - self.tracking_buffer[0][0] > window):
                    self.tracking_buffer.popleft()

    def trigger_callback(self, msg):
        if not msg.trigger:
            self.get_logger().debug('ignoring BraidTrigger with trigger=false')
            return
        if not self.ready.is_set():
            self.get_logger().warn('dropping trigger for obj %d: strand-cam not ready yet'
                                   % msg.obj_id)
            return
        with self.lock_event:
            if self.active_event is not None:
                self.get_logger().warn(
                    'dropping trigger for obj %d: event for obj %d still in progress '
                    '(strand-cam has a single post-trigger buffer)'
                    % (msg.obj_id, self.active_event.obj_id))
                return
            # trigger the camera first: the pre-trigger buffer is a moving window
            try:
                self._post_callback('PostTrigger')
            except requests.RequestException as err:
                self.get_logger().error('PostTrigger failed, dropping trigger: %s' % err)
                return
            receive_time = time.time()
            trigger_time = time_msg_to_float(msg.stamp)
            if trigger_time == 0.0:
                trigger_time = receive_time
            event_name = (time.strftime('%Y%m%d_%H%M%S', time.localtime(receive_time))
                          + '.%06d_obj%d' % ((receive_time % 1) * 1e6, msg.obj_id))
            event_dir = os.path.join(self.events_dir, event_name)
            os.makedirs(event_dir, exist_ok=True)
            self.active_event = ActiveEvent(msg.obj_id, trigger_time, receive_time,
                                            msg.metadata, event_dir,
                                            self._get_state('measured_fps', float('nan')))
        self.get_logger().info('trigger received for obj %d -> %s' % (msg.obj_id, event_dir))
        self._stop_timer = self.create_timer(self.post_trigger_seconds + 0.25,
                                             self._on_post_window_done)

    def _on_post_window_done(self):
        self.destroy_timer(self._stop_timer)
        self._stop_timer = None
        try:
            self._post_callback({'SetIsRecordingMp4': False})
        except requests.RequestException as err:
            self.get_logger().error('failed to stop mp4 recording: %s' % err)
        self._finalize_thread = threading.Thread(target=self._finalize_event, daemon=True)
        self._finalize_thread.start()

    # ------------------------------------------------------------ finalizing

    def _wait_recording_stopped(self, timeout=10.0):
        t_start = time.time()
        while time.time() - t_start < timeout:
            if self._get_state('is_recording_mp4') is None:
                return True
            if self._shutting_down:
                # SSE state is frozen during shutdown; the file-size stability
                # check in _find_staging_mp4 covers finalization instead
                return True
            time.sleep(0.25)
        return False

    def _find_staging_mp4(self, timeout=30.0):
        '''Return the finished MP4 in the staging dir once its size is stable.

        The SSE state's RecordingPath filename can differ from the on-disk name
        (observed ~1 s offset: the state reports the template resolved at
        trigger time, the file is named from the actual start time), so glob
        the private staging dir instead of trusting it. The encoder can keep
        draining its queue for several seconds after the stop command, so
        require the size to hold still for a few consecutive checks.
        '''
        t_start = time.time()
        path = None
        stable_count = 0
        last_size = -1
        while time.time() - t_start < timeout:
            candidates = glob.glob(os.path.join(self.staging_dir, '*.mp4'))
            if not candidates:
                time.sleep(0.5)
                continue
            if len(candidates) > 1:
                self.get_logger().warn('multiple mp4s in staging dir, using newest: %s'
                                       % candidates)
            path = max(candidates, key=os.path.getmtime)
            size = os.path.getsize(path)
            stable_count = stable_count + 1 if size == last_size else 0
            last_size = size
            if stable_count >= 2:  # unchanged across two consecutive 1 s intervals
                return path
            time.sleep(1.0)
        return None

    def _finalize_event(self):
        event = self.active_event
        error = ''
        mp4_filename = ''
        try:
            # snapshot the tracking window first: the post window has just
            # closed, and waiting on the mp4 below can outlive buffer retention
            t0 = event.trigger_time - self.pre_trigger_seconds
            t1 = event.trigger_time + self.post_trigger_seconds
            with self.lock_buffer:
                rows = [row for (t, obj_id, row) in self.tracking_buffer
                        if obj_id == event.obj_id and t0 <= t <= t1]

            if not self._wait_recording_stopped():
                self.get_logger().warn('strand-cam still reports recording after timeout')
            mp4_path = self._find_staging_mp4()
            if mp4_path is None:
                error = 'mp4 not found in staging dir'
                self.get_logger().error(error)
            else:
                mp4_filename = os.path.basename(mp4_path)
                mp4_path = shutil.move(mp4_path, event.event_dir)
                if self.run_show_timestamps and shutil.which('show-timestamps'):
                    self._extract_frame_timestamps(mp4_path, event.event_dir)

            if len(rows) == 0:
                self.get_logger().warn('no tracking rows for obj %d in the trigger window'
                                       % event.obj_id)
                if error == '':
                    error = 'no tracking rows for obj_id in window'

            self._write_metadata_hdf5(event, rows, mp4_filename, error)
            self.get_logger().info('event complete: %s (%d tracking rows)'
                                   % (event.event_dir, len(rows)))
        except Exception as err:
            self.get_logger().error('finalizing event failed: %s' % err)
        finally:
            with self.lock_event:
                self.active_event = None

    def _extract_frame_timestamps(self, mp4_path, event_dir):
        # retry: the mp4 may not be parseable until the writer has fully closed it
        for attempt in range(4):
            try:
                r = subprocess.run(['show-timestamps', '--output', 'csv', mp4_path],
                                   capture_output=True, text=True, timeout=60)
                if r.returncode == 0 and r.stdout != '':
                    with open(os.path.join(event_dir, 'frame_timestamps.csv'), 'w') as f:
                        f.write(r.stdout)
                    return
                self.get_logger().warn('show-timestamps failed (code %d, attempt %d): %s'
                                       % (r.returncode, attempt + 1,
                                          r.stderr.strip()[:500]))
            except (OSError, subprocess.TimeoutExpired) as err:
                self.get_logger().warn('show-timestamps failed (attempt %d): %s'
                                       % (attempt + 1, err))
            time.sleep(2.0)

    def _write_metadata_hdf5(self, event, rows, mp4_filename, error):
        filename = os.path.join(event.event_dir, 'metadata.hdf5')
        with h5py.File(filename, 'w') as f:
            f.attrs.create('camera_name', self.camera_name)
            f.attrs.create('mp4_filename', mp4_filename)
            f.attrs.create('obj_id', event.obj_id)
            f.attrs.create('trigger_time_unix', event.trigger_time)
            f.attrs.create('trigger_receive_time_unix', event.receive_time)
            f.attrs.create('trigger_metadata', event.metadata)
            f.attrs.create('pre_trigger_seconds', self.pre_trigger_seconds)
            f.attrs.create('post_trigger_seconds', self.post_trigger_seconds)
            f.attrs.create('post_trigger_buffer_frames', self.post_trigger_buffer_frames)
            f.attrs.create('fps_configured', self.fps)
            f.attrs.create('fps_measured', event.fps_measured)
            f.attrs.create('strand_cam_version', self.strand_cam_version)
            f.attrs.create('braid_url', self.braid_url)
            f.attrs.create('created_at', time.time())
            f.attrs.create('clock_note', CLOCK_NOTE)
            f.attrs.create('config_toml', self.config_toml_text)
            f.attrs.create('n_tracking_rows', len(rows))
            f.attrs.create('error', error)
            f.create_dataset('tracking_rows', data=np.array(rows, dtype=TRACKING_DTYPE))

    # ----------------------------------------------------------- supervision

    def _housekeeping(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.get_logger().fatal('strand-cam exited unexpectedly (code %s), shutting down'
                                    % self.proc.returncode)
            raise SystemExit
        measured = self._get_state('measured_fps')
        if (measured is not None and not self._fps_mismatch_warned and measured > 0
                and abs(measured - self.fps) / self.fps > 0.2):
            self._fps_mismatch_warned = True
            self.get_logger().warn(
                'camera measured fps (%.1f) differs from configured fps (%.1f); '
                'the pre-trigger buffer of %d frames spans %.2f s, not %.2f s'
                % (measured, self.fps, self.post_trigger_buffer_frames,
                   self.post_trigger_buffer_frames / measured, self.pre_trigger_seconds))

    def shutdown(self):
        if self.shutdown_done:
            return
        self.shutdown_done = True
        self._shutting_down = True

        with self.lock_event:
            event_active = self.active_event is not None
        if event_active:
            self.get_logger().info('shutting down with an active event, stopping recording')
            try:
                self._post_callback({'SetIsRecordingMp4': False})
            except requests.RequestException:
                pass
            if self._finalize_thread is None or not self._finalize_thread.is_alive():
                self._finalize_thread = threading.Thread(target=self._finalize_event,
                                                         daemon=True)
                self._finalize_thread.start()
        if self._finalize_thread is not None:
            self._finalize_thread.join(timeout=15.0)

        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        print('shut down nicely')


def main():
    default_config = os.path.join(get_package_share_directory('braid_tools'),
                                  'video_config', 'triggered_video_saver.toml')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config,
                        help="path to a triggered_video_saver .toml config")
    parser.add_argument("--output-dir", type=str, dest="output_dir", default='',
                        help="override the output_base_dir from the config")
    parser.add_argument("--attach", action="store_true",
                        help="do not launch strand-cam; attach to an already-running "
                             "instance at the configured http_server_addr")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = TriggeredVideoSaver(config_path=args.config,
                               output_dir_override=args.output_dir,
                               attach=args.attach)
    atexit.register(node.shutdown)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
