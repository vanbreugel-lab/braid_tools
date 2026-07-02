#!/usr/bin/env python3
"""ROS 2 client for the braid_tools ROS 1 -> ROS 2 topic relay.

Connects to the HTTP event stream served by topic_relay_server.py (which
runs on a ROS 1 machine, see the `main` branch of this repo) and republishes
every relayed message into the local ROS 2 graph, on the same topic names.

The client needs no configuration beyond the server URL: the server's
`hello` event announces which topics it relays and their ROS 2 message
types, and publishers are created from that.

Wire format (one SSE event per HTTP/1.1 chunk):

    event: ros1_relay_hello
    data: {"version": 1, "topics": {"/topic": "pkg/msg/Type", ...}}

    event: ros1_relay
    data: {"topic": "/topic", "type": "pkg/msg/Type", "t_wall": <float>, "msg": {...}}

    event: ros1_relay_heartbeat
    data: {"t_wall": <float>}

Time/Duration fields inside "msg" use ROS 2 spelling: {"sec": int, "nanosec": int}.

Usage:
    ros2 run braid_tools topic_relay_client.py --url http://ROS1.MACHINE.IP:8398/
"""

import sys
import argparse
import json
import time

import requests

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_message

HEARTBEAT_S = 2.0  # must match the server's --heartbeat


def parse_chunk(chunk):
    '''Parse one SSE event into (event_name, data_dict).'''
    lines = chunk.strip().split('\n')
    assert len(lines) == 2, 'malformed SSE event: %r' % chunk[:200]
    assert lines[0].startswith('event: ')
    assert lines[1].startswith('data: ')
    event_name = lines[0][len('event: '):]
    data = json.loads(lines[1][len('data: '):])
    return event_name, data


class TopicRelayClient(Node):
    def __init__(self, relay_server_url):
        super().__init__('topic_relay_client')
        self.relay_server_url = relay_server_url
        self.pubs = {}  # topic -> (publisher, msg_class)

    def get_or_create_pub(self, topic, type_str):
        if topic not in self.pubs:
            cls = get_message(type_str)
            self.pubs[topic] = (self.create_publisher(cls, topic, 100), cls)
            self.get_logger().info('Publishing %s [%s]' % (topic, type_str))
        return self.pubs[topic]

    def handle_event(self, event_name, data):
        if event_name == 'ros1_relay':
            pub, cls = self.get_or_create_pub(data['topic'], data['type'])
            msg = cls()
            set_message_fields(msg, data['msg'])
            pub.publish(msg)
        elif event_name == 'ros1_relay_hello':
            for topic, type_str in data['topics'].items():
                self.get_or_create_pub(topic, type_str)
        elif event_name == 'ros1_relay_heartbeat':
            pass
        else:
            self.get_logger().warn('Unknown relay event: %s' % event_name)

    def run_once(self):
        '''Connect and consume the stream until it ends or errors.'''
        session = requests.session()
        events_url = self.relay_server_url + 'events'
        r = session.get(events_url,
                        stream=True,
                        headers={'Accept': 'text/event-stream'},
                        timeout=(5, 3 * HEARTBEAT_S))
        r.raise_for_status()
        self._connected = True
        self.get_logger().info('Connected to %s' % events_url)

        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            event_name, data = parse_chunk(chunk)
            self.handle_event(event_name, data)

    def run(self):
        '''Consume the stream forever, reconnecting with backoff.'''
        failures = 0
        while rclpy.ok():
            self._connected = False
            try:
                self.run_once()
                self.get_logger().warn('Relay stream ended')
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.HTTPError) as err:
                self.get_logger().warn('Relay connection failed: %s' % err)
            if self._connected:
                failures = 0  # made it onto the stream; back off from scratch next time
            delay = min(2 ** failures, 10)
            failures += 1
            self.get_logger().info('Reconnecting in %d s ...' % delay)
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default='http://localhost:8398/',
                        help="URL of the topic relay server")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    url = args.url if args.url.endswith('/') else args.url + '/'

    rclpy.init()
    node = TopicRelayClient(url)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
