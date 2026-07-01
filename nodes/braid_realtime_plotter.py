#!/usr/bin/env python3
import sys
import os
import argparse
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.utilities import remove_ros_args
from ament_index_python.packages import get_package_share_directory

from braid_tools.msg import FlydraMainbrainSuperPacket


class RealTimePlotter(Node):
    def __init__(self, braid_topic, config_file):
        super().__init__('braid_realtime_plotter')
        self.config_file = config_file

        # load configuration file
        with open(config_file) as file:
            config = yaml.safe_load(file)
        self.xmin = float(config['xmin'])
        self.xmax = float(config['xmax'])
        self.ymin = float(config['ymin'])
        self.ymax = float(config['ymax'])
        self.zmin = float(config['zmin'])
        self.zmax = float(config['zmax'])
        self.tail = int(config['tail'])

        # example configuration file:
        '''
            xmin: -.3
            xmax: .3

            ymin: -.25
            ymax: .25

            zmin: 0
            zmax: .5

            tail: 300
        '''

        # initialize data
        self.x_vec = [0,]
        self.y_vec = [0,]
        self.z_vec = [0,]
        self.tcall = time.time()
        self.lock = threading.Lock()

        # initialize ROS stuff
        self.create_subscription(FlydraMainbrainSuperPacket, braid_topic,
                                 self.trigger_callback, 10)

        # initialize figure
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_zlim(self.zmin, self.zmax)
        self.ax.set_xlabel('x - axis')
        self.ax.set_ylabel('y - axis')
        self.ax.set_zlabel('z - axis')

        # initialize line
        self.line, = self.ax.plot(self.x_vec, self.y_vec, self.z_vec)

    def trigger_callback(self, super_packet):
        self.tcall = time.time()
        with self.lock:
            for packet in super_packet.packets:
                if len(packet.objects) > 0:
                    obj = packet.objects[0]
                    if isinstance(obj.position.x, float) and isinstance(obj.position.y, float) and isinstance(obj.position.z, float):
                        self.x_vec.append(obj.position.x)
                        self.y_vec.append(obj.position.y)
                        self.z_vec.append(obj.position.z)
                    if len(packet.objects) > 1:
                        print('WARNING: only plotting first object! More than 1 object not implemented')

    def main(self):

        def animate(i):
            with self.lock:
                # trim data to a short tail
                self.x_vec = self.x_vec[-1 * self.tail:]
                self.y_vec = self.y_vec[-1 * self.tail:]
                self.z_vec = self.z_vec[-1 * self.tail:]

                # update the line
                self.line.set_xdata(np.array(self.x_vec))
                self.line.set_ydata(np.array(self.y_vec))  # update the data
                self.line.set_3d_properties(np.array(self.z_vec))

            return self.line,

        # Init only required for blitting to give a clean slate.
        def init():
            self.line.set_xdata(np.array(self.x_vec))
            self.line.set_ydata(np.array(self.y_vec))  # update the data
            self.line.set_3d_properties(np.array(self.z_vec))
            return self.line,

        # spin ROS in a background thread; matplotlib needs the main thread
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        try:
            self.ani = animation.FuncAnimation(self.fig, animate, None, init_func=init,
                                               interval=15, blit=True, cache_frame_data=False)
            plt.show()
        finally:
            executor.shutdown()


def main():
    default_config = os.path.join(get_package_share_directory('braid_tools'),
                                  'plot_config', 'small_tunnel.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, dest="config", default=default_config,
                        help="Full path that points to a config.yaml file")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    real_time_plotter = RealTimePlotter("flydra_mainbrain/super_packets",
                                        os.path.expanduser(args.config))
    try:
        real_time_plotter.main()
    except KeyboardInterrupt:
        pass
    finally:
        real_time_plotter.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
