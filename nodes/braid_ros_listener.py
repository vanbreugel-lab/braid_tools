#!/usr/bin/env python3
"""# This script listens to the HTTP JSON Event Stream of Strand Braid.

Writes /flydra_mainbrain_super_packets.

You probably also want the ros_flydra flydra2ros node to make pose messages.
"""

from __future__ import print_function
import argparse
import requests
import json
import time
import socket

# from ros_flydra.msg import flydra_mainbrain_super_packet, flydra_mainbrain_packet, flydra_object
from braid_tools.msg import flydra_mainbrain_super_packet, flydra_mainbrain_packet, flydra_object

import rospy

DATA_PREFIX = 'data: '


class BraidProxy:
    def __init__(self, braid_model_server_url):
        self.braid_model_server_url = braid_model_server_url
        self.session = requests.session()
        count = 0
        while True:
            try:
                r = self.session.get(self.braid_model_server_url)
                rospy.loginfo('Connected to %s' % self.braid_model_server_url)
                break
            except requests.exceptions.ConnectionError as err:
                if count > 20:
                    raise err
                rospy.loginfo('Sleeping because we failed to connect to server at %s' % self.braid_model_server_url)
                time.sleep(1.0)
                count += 1
        assert (r.status_code == requests.codes.ok)

        self.pub = rospy.Publisher('flydra_mainbrain/super_packets',
                                   flydra_mainbrain_super_packet, queue_size=100)


    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        events_url = self.braid_model_server_url + 'events'
        r = self.session.get(events_url,
                             stream=True,
                             headers={'Accept': 'text/event-stream'},
                             )

        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            data = parse_chunk(chunk)
            version = data.get('v', 1)  # default because missing in first release
            assert version in (1, 2)  # check the data version

            try:
                msg_dict = data['msg']
                update_dict = msg_dict['Update']
            except KeyError:
                continue

            msg = flydra_mainbrain_super_packet()
            packet = flydra_mainbrain_packet()

            objects = []

            obj = flydra_object()
            obj.obj_id = update_dict['obj_id']
            obj.position.x = update_dict['x']
            obj.position.y = update_dict['y']
            obj.position.z = update_dict['z']
            obj.velocity.x = update_dict['xvel']
            obj.velocity.y = update_dict['yvel']
            obj.velocity.z = update_dict['zvel']
            obj.posvel_covariance_diagonal = [update_dict['P%d%d' % (i, i)] for i in range(6)]
            objects.append(obj)

            packet.framenumber = update_dict['frame']
            packet.reconstruction_stamp = rospy.get_rostime()
            timestamp = data['trigger_timestamp']
            if timestamp is None:
                rospy.loginfo('Skipping data transmission due to missing timestamp')
                continue
            packet.acquire_stamp = rospy.Time.from_sec(timestamp)
            packet.objects = objects

            msg.packets = [packet]

            self.pub.publish(msg)


def parse_chunk(chunk):
    lines = chunk.strip().split('\n')
    assert (len(lines) == 2)
    assert (lines[0] == 'event: braid')
    assert (lines[1].startswith(DATA_PREFIX))
    buf = lines[1][len(DATA_PREFIX):]
    data = json.loads(buf)
    return data


def main():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--braid-model-server-url", default='http://0.0.0.0:8397/',
    parser.add_argument("--braid-model-server-url", default='http://127.0.0.1:8397/',
                        help="URL of Braid model server")

    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    rospy.init_node('braid_ros_listener', disable_signals=True)
    BraidProxy(args.braid_model_server_url).run()


if __name__ == '__main__':
    main()
