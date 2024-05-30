#!/usr/bin/env python
from __future__ import division
from optparse import OptionParser
import roslib
import rospy
import os
import time
import threading

import numpy as np
from braid_tools.msg import flydra_mainbrain_super_packet, flydra_mainbrain_packet, flydra_object

import h5py

import atexit

class DataListener:
    def __init__(self, info='data', record_time_hrs=24, home_directory=''):
        self.subTrackedObjects = rospy.Subscriber('/flydra_mainbrain/super_packets', 
            flydra_mainbrain_super_packet, self.tracked_object_callback, queue_size=500)
        
        experiment_basename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
           
        filename = experiment_basename + '_braid_objects.hdf5'

        if home_directory == '':
            try:
                home_directory = os.path.expanduser( rospy.get_param('/braid/data_directory') )
            except:
                home_directory = os.path.expanduser('~/Desktop/temp')   
        else:
            home_directory = os.path.expanduser(home_directory)

        filename = os.path.join(home_directory, filename)
        
        print('Saving hdf5 data to: ', filename)
        
        self.time_start = time.time()
        self.record_time_hrs = record_time_hrs
        
        self.buffer = []
        self.array_buffer = []
        # set up thread locks
        self.lockParams = threading.Lock()
        self.lockBuffer = threading.Lock()
        
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
        rospy.init_node('save_braid_data_to_hdf5')
        
        self.hdf5.create_dataset('data', (self.chunk_size, 1), maxshape=(None,1), dtype=self.dtype)
        self.hdf5['data'].attrs.create('current_frame', 0)
        self.hdf5['data'].attrs.create('line', 0)
        self.hdf5['data'].attrs.create('length', self.chunk_size)
        
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
                                    packet.reconstruction_stamp.secs,
                                    packet.reconstruction_stamp.nsecs,
                                    packet.acquire_stamp.secs,
                                    packet.acquire_stamp.nsecs,
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

    def process_buffer(self):
        self.save_array_data()
            
    def main(self):
        atexit.register(self.stop_saving_data)
        while (not rospy.is_shutdown()):
            t = time.time() - self.time_start
            if t > self.record_time_hrs*3600:
                self.stop_saving_data()
                return
            with self.lockBuffer:
                time_now = rospy.Time.now()
                if len(self.array_buffer) > 0:
                    self.process_buffer()
                pt = (rospy.Time.now()-time_now).to_sec()
                if len(self.buffer) > 9:
                    rospy.logwarn("Data saving processing time exceeds acquisition rate. Processing time: %f, Buffer: %d", pt, len(self.buffer))
            
        
    def stop_saving_data(self):
        self.hdf5.close()
        print('shut down nicely')
        
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--record-time-hrs", type="int", dest="record_time_hrs", default=24,
                        help="number of hours to record data for")
    parser.add_option("--home-directory", type="str", dest="home_directory", default='~/Desktop/temp',
                        help="home directory for saving data to")
    (options, args) = parser.parse_args()
    
    datalistener = DataListener(record_time_hrs=options.record_time_hrs, 
                                home_directory=options.home_directory)
    datalistener.main()
