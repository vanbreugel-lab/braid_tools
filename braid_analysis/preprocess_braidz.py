import pandas as pd 
import os
import argparse
import numpy as np

from braid_analysis import braid_filemanager
from braid_analysis import braid_slicing
from braid_analysis import flymath

def load_braidz_and_filter(data_directory):
    braidz_filename = braid_filemanager.get_filename(data_directory, '.braidz')
    braid_df = braid_filemanager.load_filename_as_dataframe_3d(braidz_filename)
    braid_df_culled = braid_filter_objids(braid_df)
    name = os.path.basename(braidz_filename).split('.')[0]
    braid_df_culled = braid_slicing.assign_unique_id(braid_df_culled, name)

    return braid_df_culled

def braid_filter_objids(braid_df, length=50, xdist_travelled=0.1):
    long_obj_ids = braid_slicing.get_long_obj_ids_fast_pandas(braid_df, length=length)
    braid_df_culled = braid_slicing.get_data_frame_slice_from_obj_ids(braid_df, long_obj_ids)
    long_xdist_objids = braid_slicing.get_trajectories_that_travel_far(braid_df_culled, 
                                                                       xdist_travelled=xdist_travelled)
    braid_df_culled = braid_slicing.get_data_frame_slice_from_obj_ids(braid_df_culled, long_xdist_objids)
    
    return braid_df_culled

def save_preprocessed_braidz(data_directory, braid_df_culled):
    braidz_filename = braid_filemanager.get_filename(data_directory, '.braidz')
    preprocessed_data_dir = os.path.join(data_directory, 'preprocessed_data')
    if not os.path.isdir(preprocessed_data_dir):
        os.mkdir(preprocessed_data_dir)

    preprocessed_data_fname = os.path.basename(braidz_filename).split('.')[0] + '_preprocessed.hdf'
    fname = os.path.join(preprocessed_data_dir, preprocessed_data_fname)
    braid_df_culled.to_hdf(fname, 'DATA_' + os.path.basename(braidz_filename).split('.')[0] )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help="data directory")
    parser.add_argument('--length', type=int, default=50,
                        help="minimum trajectory length")
    parser.add_argument('--xdist', type=float, default=0.1,
                        help="minimum xdist travelled")
    args = parser.parse_args()

    braid_df_culled = load_braidz_and_filter(args.directory)
    braid_df_culled = braid_filter_objids(braid_df_culled, length=args.length, xdist_travelled=args.xdist)

    braid_df_culled = braid_df_culled[0:200000].copy()

    braid_df_culled['speed'] = np.sqrt(braid_df_culled.xvel**2 + braid_df_culled.yvel**2)
    braid_df_culled = flymath.assign_course_and_ang_vel_to_dataframe(braid_df_culled)

    save_preprocessed_braidz(args.directory, braid_df_culled)
