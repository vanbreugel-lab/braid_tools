#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from braid_tools.msg import flydra_mainbrain_super_packet, flydra_mainbrain_packet, flydra_object
import yaml
import time
from optparse import OptionParser

class RealTimePlotter:
    def __init__(self, braid_topic, config_file):
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

        # initialize ROS stuff
        rospy.init_node("braid_realtime_plotter", anonymous=True)
        rospy.Subscriber(braid_topic, flydra_mainbrain_super_packet, self.trigger_callback)
        
        # initialize figure
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        plt.style.use('seaborn-white')
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
        obj_ids = []
        for packet in super_packet.packets:
            if len(packet.objects) > 0:
                obj = packet.objects[0]
                #for obj in packet.objects:
                if isinstance(obj.position.x, float) and isinstance(obj.position.y, float) and isinstance(obj.position.z, float):
                    self.x_vec.append(obj.position.x)
                    self.y_vec.append(obj.position.y)
                    self.z_vec.append(obj.position.z)
                if len(packet.objects) > 1:
                    print('WARNING: only plotting first object! More than 1 object not implemented')

    def main(self):

        def animate(i):
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

        self.ani = animation.FuncAnimation(self.fig, animate, None, init_func=init,
                                            interval=15, blit=True)
        plt.show()




if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--config", type="str", dest="config", default='',
                      help="Full path that points to a config.yaml file")
    (options, args) = parser.parse_args()


    if 1:
        config_file = options.config

        real_time_plotter = RealTimePlotter("flydra_mainbrain/super_packets", config_file)
        real_time_plotter.main()

