#!/usr/bin/env python3
"""Convert relayed ROS 1 volume-trigger messages into braid_tools/BraidTrigger.

The ROS 1 trigger node (wind_tunnel_openloop_trigger_in_volume.py, running on
the braid machine) publishes a std_msgs/Float64MultiArray with

    data = [flag, obj_id, framenumber, t_wall] + stimulus_values

which reaches this ROS 2 machine via the topic relay
(topic_relay_server.py on the ROS 1 machine -> topic_relay_client.py here)
under the same topic name. This node re-publishes each trigger as a
braid_tools/BraidTrigger on `braid_trigger`, which is what
braid_triggered_video_saver.py listens to.

The stamp is taken from t_wall (data[3]): it is the braid machine's wall
clock, the same clock domain as the tracking acquire_stamp, so the video
saver's pre/post window is evaluated correctly without any cross-machine
clock synchronization. The stimulus values (the arduino `bag_data`) are
preserved verbatim in the BraidTrigger metadata JSON and therefore end up in
each event's metadata.hdf5.

    ros2 run braid_tools braid_trigger_adapter.py --trigger-topic braid_trigger_topic
"""

import sys
import json
import argparse

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.utilities import remove_ros_args
from builtin_interfaces.msg import Time
from std_msgs.msg import Float64MultiArray

from braid_tools.msg import BraidTrigger


def unix_time_to_msg(t):
    '''Convert a float unix timestamp to a builtin_interfaces/Time message.'''
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


class BraidTriggerAdapter(Node):
    def __init__(self, trigger_topic):
        super().__init__('braid_trigger_adapter')
        self.trigger_topic = trigger_topic
        self.sub = self.create_subscription(Float64MultiArray, trigger_topic,
                                            self.callback, 10)
        self.pub = self.create_publisher(BraidTrigger, 'braid_trigger', 10)
        self.get_logger().info('forwarding %s (Float64MultiArray) -> '
                               'braid_trigger (BraidTrigger)' % trigger_topic)

    def callback(self, msg):
        data = list(msg.data)
        if len(data) < 2:
            self.get_logger().warn('trigger message with %d values (< 2), '
                                   'dropping: %s' % (len(data), data))
            return
        out = BraidTrigger()
        out.trigger = data[0] >= 0.5
        out.obj_id = int(data[1])
        if len(data) >= 4:
            out.stamp = unix_time_to_msg(data[3])
        # else: stamp stays zero -> video saver uses its receive time
        out.metadata = json.dumps({
            'framenumber': int(data[2]) if len(data) >= 3 else None,
            'extra': data[4:],
            'source_topic': self.trigger_topic,
            'raw_data': data,
        })
        self.pub.publish(out)
        self.get_logger().info('trigger forwarded: obj_id %d (trigger=%s, '
                               '%d extra values)'
                               % (out.obj_id, out.trigger, len(data) - 4
                                  if len(data) > 4 else 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-topic", default='braid_trigger_topic',
                        help="incoming Float64MultiArray topic (as named in "
                             "the ROS 1 trigger node's config yaml)")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = BraidTriggerAdapter(args.trigger_topic)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
