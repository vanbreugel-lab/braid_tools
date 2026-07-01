#!/usr/bin/env python3
import sys
import argparse
import os
import time
import threading
import atexit

import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from braid_tools.msg import FlydraMainbrainSuperPacket


class DataListener(Node):
    def __init__(self, info='data', record_time_hrs=24, home_directory=''):
        super().__init__('save_braid_data_to_hdf5')

        self.subTrackedObjects = self.create_subscription(FlydraMainbrainSuperPacket,
            '/flydra_mainbrain/super_packets', self.tracked_object_callback, 500)

        experiment_basename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        filename = experiment_basename + '_braid_objects.hdf5'

        if home_directory == '':
            # set with: --ros-args -p data_directory:=/path/to/dir
            self.declare_parameter('data_directory', '~/Desktop/temp')
            home_directory = os.path.expanduser(self.get_parameter('data_directory').value)
        else:
            home_directory = os.path.expanduser(home_directory)

        os.makedirs(home_directory, exist_ok=True)
        filename = os.path.join(home_directory, filename)

        print('Saving hdf5 data to: ', filename)

        self.time_start = time.time()
        self.record_time_hrs = record_time_hrs

        self.array_buffer = []
        # set up thread locks
        self.lockBuffer = threading.Lock()

        self.saving_stopped = False

        self.chunk_size = 5000
        self.hdf5 = h5py.File(filename, 'w')
        #self.hdf5.swmr_mode = True # helps prevent file corruption if closed improperly
        self.hdf5.attrs.create("info", info)

        self.data_to_save = [   'frame_number',
                                'reconstruction_stamp_secs',
                                'reconstruction_stamp_nsecs',
                                'acquire_stamp_secs',
                                'acquire_stamp_nsecs',
                                'obj_id',
                                'position_x',
                                'position_y',
                                'position_z',
                                'velocity_x',
                                'velocity_y',
                                'velocity_z',
                                'posvel_covariance_d1',
                                'posvel_covariance_d2',
                                'posvel_covariance_d3',
                                'posvel_covariance_d4',
                                'posvel_covariance_d5',
                                'posvel_covariance_d6',
                                ]

        self.data_format = {    'frame_number': int,
                                'reconstruction_stamp_secs': int,
                                'reconstruction_stamp_nsecs': int,
                                'acquire_stamp_secs': int,
                                'acquire_stamp_nsecs': int,
                                'obj_id': int,
                                'position_x': float,
                                'position_y': float,
                                'position_z': float,
                                'velocity_x': float,
                                'velocity_y': float,
                                'velocity_z': float,
                                'posvel_covariance_d1': float,
                                'posvel_covariance_d2': float,
                                'posvel_covariance_d3': float,
                                'posvel_covariance_d4': float,
                                'posvel_covariance_d5': float,
                                'posvel_covariance_d6': float,
                            }

        self.dtype = [(data,self.data_format[data]) for data in self.data_to_save]

        self.hdf5.create_dataset('data', (self.chunk_size, 1), maxshape=(None,1), dtype=self.dtype)
        self.hdf5['data'].attrs.create('current_frame', 0)
        self.hdf5['data'].attrs.create('line', 0)
        self.hdf5['data'].attrs.create('length', self.chunk_size)

        self.timer = self.create_timer(0.05, self.check_buffer)

    def add_chunk(self):
        length = self.hdf5['data'].attrs.get('length')
        new_length = length + self.chunk_size
        self.hdf5['data'].resize(new_length, axis=0)
        self.hdf5['data'].attrs.modify('length', new_length)

    def save_array_data(self):
        newline = self.hdf5['data'].attrs.get('line') + 1
        nrows_to_add = len(self.array_buffer)

        self.hdf5['data'].attrs.modify('line', newline+nrows_to_add)
        if newline+nrows_to_add >= self.hdf5['data'].attrs.get('length')-50:
            self.hdf5.flush()
            self.add_chunk()

        self.hdf5['data'][newline:newline+nrows_to_add] = self.array_buffer
        self.array_buffer = []

    def tracked_object_callback(self, super_packet):
        with self.lockBuffer:
            for packet in super_packet.packets:
                for obj in packet.objects:
                    a = np.array([( packet.framenumber,
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
                                   )], dtype=self.dtype)
                    self.array_buffer.append(a)

    def check_buffer(self):
        t = time.time() - self.time_start
        if t > self.record_time_hrs*3600:
            self.stop_saving_data()
            raise SystemExit
        with self.lockBuffer:
            time_start = time.time()
            if len(self.array_buffer) > 0:
                self.save_array_data()
            pt = time.time() - time_start
            if pt > 0.05:
                self.get_logger().warn("Data saving processing time exceeds acquisition rate. Processing time: %f" % pt)

    def stop_saving_data(self):
        if self.saving_stopped:
            return
        self.saving_stopped = True
        with self.lockBuffer:
            if len(self.array_buffer) > 0:
                self.save_array_data()
            self.hdf5.close()
        print('shut down nicely')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-time-hrs", type=int, dest="record_time_hrs", default=24,
                        help="number of hours to record data for")
    parser.add_argument("--home-directory", type=str, dest="home_directory", default='',
                        help="directory for saving data to; if empty, uses the "
                             "'data_directory' ROS parameter (default ~/Desktop/temp)")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    datalistener = DataListener(record_time_hrs=args.record_time_hrs,
                                home_directory=args.home_directory)
    atexit.register(datalistener.stop_saving_data)
    try:
        rclpy.spin(datalistener)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        datalistener.stop_saving_data()
        datalistener.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
