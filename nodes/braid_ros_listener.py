#!/usr/bin/env python3
"""# This script listens to the HTTP JSON Event Stream of Strand Braid.

Writes /flydra_mainbrain/super_packets.
"""

import sys
import argparse
import requests
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from builtin_interfaces.msg import Time

from braid_tools.msg import FlydraMainbrainSuperPacket, FlydraMainbrainPacket, FlydraObject

DATA_PREFIX = 'data: '


def unix_time_to_msg(t):
    '''Convert a float unix timestamp to a builtin_interfaces/Time message.'''
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


class BraidProxy(Node):
    def __init__(self, braid_model_server_url):
        super().__init__('braid_ros_listener')
        self.braid_model_server_url = braid_model_server_url
        self.session = requests.session()
        count = 0
        while True:
            try:
                r = self.session.get(self.braid_model_server_url)
                self.get_logger().info('Connected to %s' % self.braid_model_server_url)
                break
            except requests.exceptions.ConnectionError as err:
                if count > 20:
                    raise err
                self.get_logger().info('Sleeping because we failed to connect to server at %s' % self.braid_model_server_url)
                time.sleep(1.0)
                count += 1
        assert (r.status_code == requests.codes.ok)

        self.pub = self.create_publisher(FlydraMainbrainSuperPacket,
                                         'flydra_mainbrain/super_packets', 100)

    def run(self):
        events_url = self.braid_model_server_url + 'events'
        r = self.session.get(events_url,
                             stream=True,
                             headers={'Accept': 'text/event-stream'},
                             )

        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            data = parse_chunk(chunk)
            version = data.get('v', 1)  # default because missing in first release
            #assert version in (1, 2)  # check the data version

            try:
                msg_dict = data['msg']
                update_dict = msg_dict['Update']
            except KeyError:
                continue

            msg = FlydraMainbrainSuperPacket()
            packet = FlydraMainbrainPacket()

            objects = []

            obj = FlydraObject()
            obj.obj_id = int(update_dict['obj_id'])
            obj.position.x = float(update_dict['x'])
            obj.position.y = float(update_dict['y'])
            obj.position.z = float(update_dict['z'])
            obj.velocity.x = float(update_dict['xvel'])
            obj.velocity.y = float(update_dict['yvel'])
            obj.velocity.z = float(update_dict['zvel'])
            obj.posvel_covariance_diagonal = [float(update_dict['P%d%d' % (i, i)]) for i in range(6)]
            objects.append(obj)

            packet.framenumber = int(update_dict['frame'])
            packet.reconstruction_stamp = self.get_clock().now().to_msg()
            timestamp = data['trigger_timestamp']
            if timestamp is None:
                self.get_logger().info('Skipping data transmission due to missing timestamp')
                continue
            packet.acquire_stamp = unix_time_to_msg(timestamp)
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
    parser.add_argument("--braid-model-server-url", default='http://134.197.37.229:8397/',
                        help="URL of Braid model server")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = BraidProxy(args.braid_model_server_url)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
