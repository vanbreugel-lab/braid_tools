#!/usr/bin/env python3
"""Print a running strand-cam's current point-detection settings as toml.

Tune the detection interactively in the strand-cam browser UI (live image with
detections overlaid), then run this to get a paste-ready
[cameras.point_detection_config] block for your .toml:

    python3 dump_detection_config.py                    # default 127.0.0.1:3441
    python3 dump_detection_config.py --addr 127.0.0.1:3440
"""

import json
import argparse

import requests


def fetch_state(addr, timeout=10.0):
    url = 'http://%s/strand-cam-events' % addr
    with requests.get(url, stream=True, timeout=(5, timeout),
                      headers={'Accept': 'text/event-stream'}) as r:
        r.raise_for_status()
        event_name = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                event_name = None
            elif line.startswith('event: '):
                event_name = line[len('event: '):].strip()
            elif line.startswith('data: ') and event_name == 'strand-cam':
                return json.loads(line[len('data: '):])
    raise RuntimeError('no strand-cam state received from %s' % addr)


def toml_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    return '"%s"' % v


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--addr', default='127.0.0.1:3441',
                        help='strand-cam http address (default %(default)s)')
    args = parser.parse_args()

    state = fetch_state(args.addr)
    cfg = state.get('im_pt_detect_cfg')
    if not cfg:
        raise SystemExit('strand-cam at %s reported no im_pt_detect_cfg' % args.addr)

    print('# current detection settings of %s (camera %s)'
          % (args.addr, state.get('camera_name', '?')))
    print('[cameras.point_detection_config]')
    for key, value in cfg.items():
        if isinstance(value, (dict, list)):
            # e.g. valid_region when set to a shape; keep as a comment
            print('# %s = %s   (complex value; set via the browser UI)'
                  % (key, json.dumps(value)))
        else:
            print('%s = %s' % (key, toml_value(value)))


if __name__ == '__main__':
    main()
