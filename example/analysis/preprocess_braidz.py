import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import argparse

from braid_analysis import braid_filemanager
from braid_analysis import braid_slicing
from braid_analysis import braid_analysis_plots
from braid_analysis import flymath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help="data directory")
    parser.add_argument('params', type=str, help="full path to yaml file with preprocessing parameters")
    args = parser.parse_args()
    
    data_directory = args.directory
    preprocessing_parameters_yaml_fname = args.params

    ##########################################################################################################
    # load preprocessing parameters
    with open(preprocessing_parameters_yaml_fname) as stream:
        preprocessing_params = yaml.safe_load(stream)

    # extract preprocessing params 
    preprocessed_data_subdir = preprocessing_params['preprocessed_data_subdir']
    
    # 1/framerate -- helpful for making nice timestamps
    dt = preprocessing_params['dt']

    # filtering parameters
    min_length = preprocessing_params['min_length']
    min_xdist_travelled = preprocessing_params['min_xdist_travelled']
    
    # filename for saving trajectory stats
    trajec_stats_yaml_filename = preprocessing_params['trajec_stats_yaml_filename']
    
    # preprocessed braid suffix
    preprocessed_suffix = '_preprocessed'
    ##########################################################################################################
    
    if braid_filemanager.preprocessed_braidz_exists(data_directory, sub_directory=preprocessed_data_subdir, suffix=preprocessed_suffix):
        print("----------------------------------------------------")
        print('WARNING: preprocessed data already exists.. exiting.')
        print('Clear out subdirectory and try running again')
        print('Subdir: ' + preprocessed_data_subdir )
        print("----------------------------------------------------")
        sys.exit(0)
    
    print("")
    print("----------------------------------------------------")
    print("Loading from directory: ")
    print(data_directory)
    
    # load braid as pandas dataframe
    braidz_filename = braid_filemanager.get_filename(data_directory, '.braidz')
    print(braidz_filename)
    print("")
    braid_df = braid_filemanager.load_filename_as_dataframe_3d(braidz_filename)
    
    # save some stats
    num_trajecs = len(braid_df.obj_id.unique())
    stats_raw = {'1_raw_braidz': 
                      {'num_trajecs': {'text': "Number of trajecs before filtering: ",
                                      'number': int(num_trajecs),},}
                 }
    # print stats
    print("----------------------------------------------------")
    print("Raw data stats: ")
    print( stats_raw['1_raw_braidz']['num_trajecs']['text'] + str(stats_raw['1_raw_braidz']['num_trajecs']['number']))
    print("----------------------------------------------------")
    
    # do some basic filtering of object length and xdist travelled
    print('Filtering for minimum trajec length and xdistance travelled')
    long_obj_ids = braid_slicing.get_long_obj_ids_fast_pandas(braid_df, length=min_length)
    braid_df_culled = braid_slicing.get_data_frame_slice_from_obj_ids(braid_df, long_obj_ids)
    long_xdist_objids = braid_slicing.get_trajectories_that_travel_far(braid_df_culled, 
                                                                       xdist_travelled=min_xdist_travelled)
    braid_df_culled = braid_slicing.get_data_frame_slice_from_obj_ids(braid_df_culled, long_xdist_objids)

    # assign unique obj id
    name = os.path.basename(braidz_filename).split('.')[0]
    braid_df_culled = braid_slicing.assign_unique_id(braid_df_culled, name)

    # calculate some useful things:
    print('Calculating speed (xy ground speed)')
    braid_df_culled['speed_xy'] = np.sqrt(braid_df_culled.xvel**2 + braid_df_culled.yvel**2)
    print('Calculating angular velocities')
    braid_df_culled = flymath.assign_course_and_ang_vel_to_dataframe(braid_df_culled)
    
    # save the preprocessed data
    fname = braid_filemanager.save_preprocessed_braidz(data_directory, braid_df_culled, suffix=preprocessed_suffix)
      
    # save some stats
    num_trajecs = len(braid_df_culled.obj_id.unique())
    stats_slightly_filtered = {'2_filtered_trajecs': 
                      {'num_trajecs': {'text': "Filtered number of trajecs: ",
                                      'number': int(num_trajecs),
                                      },
                      'filtering_protocol': {'min_length': min_length,
                                             'min_xdist_travelled': min_xdist_travelled},
                      'data_filename': fname}
                 }
    # print stats
    print("----------------------------------------------------")
    print("Slightly filtered stats: ")
    print( stats_slightly_filtered['2_filtered_trajecs']['num_trajecs']['text'] + str(stats_slightly_filtered['2_filtered_trajecs']['num_trajecs']['number']))
    print("----------------------------------------------------")
        
    # save the stats to a yaml file
    all_stats = {**stats_raw, **stats_slightly_filtered} # joins all stats dictionaries
    preprocessed_data_directory = os.path.join(data_directory, preprocessed_data_subdir)
    stats_yaml = os.path.join(preprocessed_data_directory, trajec_stats_yaml_filename)
    with open(stats_yaml, 'w') as yaml_file:
        yaml.dump(all_stats, yaml_file, default_flow_style=False)
        
    # instructions on how to load stats
    if 0:
        with open(stats_yaml) as stream:
            stats_yaml = yaml.safe_load(stream)
        print(stats_yaml)
