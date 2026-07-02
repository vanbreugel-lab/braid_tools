#!/usr/bin/env python3
"""ROS 1 -> ROS 2 topic relay server (the ROS 1 half).

Subscribes to a configurable list of ROS 1 topics and re-serves every
message as JSON over an HTTP server-sent-event stream, the same way the
Braid software streams tracking data. The matching ROS 2 client
(topic_relay_client.py on the `ros2` branch of this repo) connects to this
stream and republishes the messages into the ROS 2 graph.

Runs on the ROS 1 machine with no build steps: stdlib + rospy/roslib only.

    rosrun braid_tools topic_relay_server.py \
        --config $(rospack find braid_tools)/relay_config/relay_topics.yaml

Wire format (one SSE event per HTTP/1.1 chunk; chunked transfer encoding is
required -- a close-delimited HTTP/1.0 body makes the client's
iter_content(chunk_size=None) block forever):

    event: ros1_relay_hello
    data: {"version": 1, "topics": {"/topic": "pkg/msg/Type", ...}}

    event: ros1_relay
    data: {"topic": "/topic", "type": "pkg/msg/Type", "t_wall": <float>, "msg": {...}}

    event: ros1_relay_heartbeat
    data: {"t_wall": <float>}

Notes:
- "type" is always the ROS 2 type string; this server does the ROS1->ROS2
  type-name mapping (config `type_map`, defaulting to pkg/Name -> pkg/msg/Name).
- Time/Duration fields are serialized with ROS 2 spelling:
  {"sec": int, "nanosec": int}.
- uint8[] fields become JSON lists of ints. Intended for telemetry-sized
  messages; do not relay images or other large arrays.
- NaN/Inf floats pass through as Python-JSON tokens (both ends are Python).
"""

from __future__ import print_function
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

import rospy
import roslib.message


def ros1_msg_to_dict(val):
    '''genpy message -> JSON-able dict, with ROS 2 field spelling for time.

    The Time/Duration check must come before the _slot_types check:
    genpy.Time has __slots__ but no _slot_types.
    '''
    if hasattr(val, 'secs') and hasattr(val, 'nsecs'):    # genpy.Time/Duration
        return {'sec': int(val.secs), 'nanosec': int(val.nsecs)}
    if hasattr(val, '_slot_types'):                       # any genpy message
        return {name: ros1_msg_to_dict(getattr(val, name)) for name in val.__slots__}
    if isinstance(val, bytes):                            # rospy's uint8[]
        return list(val)
    if isinstance(val, (list, tuple)):
        return [ros1_msg_to_dict(v) for v in val]
    return val                                            # int/float/str/bool


def ros1_type_to_ros2(ros1_type, type_map):
    '''Map a ROS 1 type string (pkg/Name) to a ROS 2 one (pkg/msg/Name).'''
    if ros1_type in type_map:
        return type_map[ros1_type]
    pkg, name = ros1_type.split('/')
    return '%s/msg/%s' % (pkg, name)


class Relay:
    '''Fans ROS 1 messages out to per-client queues; resolves topic types.'''

    def __init__(self, topics, type_map, heartbeat_s):
        self.type_map = type_map
        self.heartbeat_s = heartbeat_s
        self.lock = threading.Lock()
        self.client_queues = []
        self.pending_topics = list(topics)   # not yet type-resolved
        self.resolved = {}                   # ros1 topic -> ros2 type string
        self.drop_counts = {}

        t = threading.Thread(target=self._resolve_topics_loop, daemon=True)
        t.start()

    # -- topic resolution / subscription --------------------------------

    def _resolve_topics_loop(self):
        master = rospy.get_master()
        while not rospy.is_shutdown() and self.pending_topics:
            try:
                _, _, topic_types = master.getTopicTypes()
            except Exception as err:
                rospy.logwarn('Could not query master for topic types: %s' % err)
                time.sleep(2.0)
                continue
            published = dict(topic_types)
            for topic in list(self.pending_topics):
                if topic not in published:
                    continue
                ros1_type = published[topic]
                msg_class = roslib.message.get_message_class(ros1_type)
                if msg_class is None:
                    rospy.logerr('Cannot import message class for %s [%s]; '
                                 'is the package on this machine?' % (topic, ros1_type))
                    self.pending_topics.remove(topic)
                    continue
                ros2_type = ros1_type_to_ros2(ros1_type, self.type_map)
                rospy.Subscriber(topic, msg_class,
                                 callback=self._msg_callback,
                                 callback_args=(topic, ros2_type),
                                 queue_size=100)
                with self.lock:
                    self.resolved[topic] = ros2_type
                    hello = self.hello_event()
                    for q in self.client_queues:
                        self._put(q, hello)
                self.pending_topics.remove(topic)
                rospy.loginfo('Relaying %s [%s -> %s]' % (topic, ros1_type, ros2_type))
            if self.pending_topics:
                rospy.loginfo_throttle(10, 'Waiting for topics to appear: %s'
                                       % self.pending_topics)
                time.sleep(2.0)

    # -- events ----------------------------------------------------------

    @staticmethod
    def sse(event_name, data):
        return 'event: %s\ndata: %s\n\n' % (event_name, json.dumps(data))

    def hello_event(self):
        return self.sse('ros1_relay_hello', {'version': 1, 'topics': self.resolved})

    def heartbeat_event(self):
        return self.sse('ros1_relay_heartbeat', {'t_wall': time.time()})

    def _msg_callback(self, msg, args):
        topic, ros2_type = args
        event = self.sse('ros1_relay', {'topic': topic,
                                        'type': ros2_type,
                                        't_wall': time.time(),
                                        'msg': ros1_msg_to_dict(msg)})
        with self.lock:
            for q in self.client_queues:
                self._put(q, event)

    def _put(self, q, event):
        try:
            q.put_nowait(event)
        except queue.Full:
            try:
                q.get_nowait()   # drop oldest
                q.put_nowait(event)
            except (queue.Empty, queue.Full):
                pass
            self.drop_counts[id(q)] = self.drop_counts.get(id(q), 0) + 1
            rospy.logwarn_throttle(5, 'Slow relay client, dropping messages '
                                      '(%d dropped)' % self.drop_counts[id(q)])

    # -- client bookkeeping ----------------------------------------------

    def register_client(self):
        q = queue.Queue(maxsize=1000)
        with self.lock:
            q.put_nowait(self.hello_event())
            self.client_queues.append(q)
        return q

    def unregister_client(self, q):
        with self.lock:
            if q in self.client_queues:
                self.client_queues.remove(q)
            self.drop_counts.pop(id(q), None)


RELAY = None  # set in main(); BaseHTTPRequestHandler has no clean ctor hook


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass  # quiet; rospy handles logging

    def write_chunk(self, payload):
        data = payload.encode()
        self.wfile.write(('%x\r\n' % len(data)).encode() + data + b'\r\n')
        self.wfile.flush()

    def do_GET(self):
        if self.path != '/events':
            body = b'braid_tools topic relay server; stream at /events\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        q = RELAY.register_client()
        rospy.loginfo('Relay client connected: %s' % (self.client_address,))
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            while not rospy.is_shutdown():
                try:
                    event = q.get(timeout=RELAY.heartbeat_s)
                except queue.Empty:
                    event = RELAY.heartbeat_event()
                self.write_chunk(event)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            RELAY.unregister_client(q)
            rospy.loginfo('Relay client disconnected: %s' % (self.client_address,))


def main():
    global RELAY

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='yaml file with a `topics` list and optional `type_map`')
    parser.add_argument('--port', type=int, default=8398,
                        help='port to serve on, default 8398')
    parser.add_argument('--host', default='0.0.0.0',
                        help='address to bind, default 0.0.0.0')
    parser.add_argument('--heartbeat', type=float, default=2.0,
                        help='seconds between keepalive events when idle, default 2')
    args = parser.parse_args(rospy.myargv()[1:])

    with open(args.config) as f:
        config = yaml.safe_load(f)
    topics = config['topics']
    type_map = config.get('type_map') or {}

    rospy.init_node('topic_relay_server')
    RELAY = Relay(topics, type_map, args.heartbeat)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    rospy.loginfo('Topic relay server listening on http://%s:%d/events'
                  % (args.host, args.port))

    rospy.spin()
    server.shutdown()


if __name__ == '__main__':
    main()
