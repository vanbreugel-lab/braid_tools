#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import rospy
from std_msgs.msg import Float32, Float32MultiArray
# from ros_flydra.msg import *
from braid_tools.msg import flydra_mainbrain_super_packet, flydra_mainbrain_packet, flydra_object
import yaml
import time
from optparse import OptionParser

fig = plt.figure()
ax = plt.axes(projection='3d')
x_vec = []
y_vec = []
z_vec = []

tcall = time.time()


def trigger_callback(super_packet):
    tcall = time.time()
    obj_ids = []
    for packet in super_packet.packets:
        for obj in packet.objects:
            obj_ids.append(obj.obj_id)
            x_vec.append(obj.position.x)
            y_vec.append(obj.position.y)
            z_vec.append(obj.position.z)
        # print(obj.obj_id)


#### Read in config yaml
'''
        config -- path to a .yaml file describing the parameters for triggering. see below for an example.
'''
config_file = options.config

with open(config_file) as file:
    config = yaml.safe_load(file)

# save config inputs to variables
xmin = config['xmin']
xmax = config['xmax']

ymin = config['ymin']
ymax = config['ymax']

zmin = config['zmin']
zmax = config['zmax']

tail = config['tail']


def braid_sub():
    rospy.init_node("Gimme_the_data", anonymous=True)
    rospy.Subscriber("/flydra_mainbrain/super_packets", flydra_mainbrain_super_packet, trigger_callback)
    # rospy.spin()
    plt.show(block=True)


def animate_(i, x_vec, y_vec, z_vec):
    plt.style.use('seaborn-white')
    x_vec = x_vec[-1 * tail:]
    y_vec = y_vec[-1 * tail:]
    z_vec = z_vec[-1 * tail:]
    ax.clear()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_zlim(zmin, zmax)
    ax.scatter(x_vec, y_vec, z_vec)
#test

ani = animate.FuncAnimation(fig, animate_, fargs=(x_vec, y_vec, z_vec), interval=10)
# plt.show()
if __name__ == "__main__":
    braid_sub()
    parser = OptionParser()
    parser.add_option("--config", type="str", dest="config", default='',
                      help="Full path that points to a config.yaml file')
                      (options, args) = parser.parse_args()


