import pandas as pd
import numpy as np

def get_long_obj_ids_fast_pandas(df_3d, length=30):
    '''
    Use fancy pandas stuff to get a list of the object id numbers that are longer than the given length (in frames)

    Inputs
    ------
    df_3d  --------- (dataframe) pandas dataframe from braidz_filemanager.load_filename_as_dataframe_3d
    length --------- (int) minimum length for objects, in frames

    Returns
    -------
    obj_ids  ------- (list) of object id numbers

    '''
    try:
        number_frames_per_obj_id = df_3d[["frame", "obj_id"]].groupby(by=["obj_id"]).agg(["count"])
        obj_ids = number_frames_per_obj_id[  number_frames_per_obj_id[('frame', 'count')]  >  length  ].index.values
        return obj_ids
    except:
        number_frames_per_obj_id = df_3d[["frames", "objid"]].groupby(by=["objid"]).agg(["count"])
        obj_ids = number_frames_per_obj_id[  number_frames_per_obj_id[('frames', 'count')]  >  length  ].index.values
        return obj_ids

def get_middle_of_tunnel_obj_ids_fast_pandas(df_3d, zmin=0.1, zmax=0.4, ymin=-0.15, ymax=0.15, xmin=-0.5, xmax=0.5):
    '''
    Use fancy pandas stuff to get a list of the object id numbers for which the median x,y,z values lie within the indicate range. 

    Inputs
    ------
    df_3d  --------- (dataframe) pandas dataframe from braidz_filemanager.load_filename_as_dataframe_3d
    zmin   --------- (float) minimum zval
    zmax   --------- (float) maximum zval
    ymin, ymax  ---- (float)
    xmin, xmax  ---- (float)

    Returns
    -------
    obj_ids  ------- (list) of object id numbers

    '''

    # only trajectories that are on average in the middle of the z dimension
    median_z = df_3d[["frame", "obj_id", "z"]].groupby(by=["obj_id"]).agg(["median"])
    obj_ids = median_z['z'].query('median > ' + str(zmin) + ' and median < ' + str(zmax) ).index.values
    df_3d_filtered = df_3d[df_3d.obj_id.isin(obj_ids)]
    
    # only trajectories that are on average in the middle of the y dimension
    median_y = df_3d_filtered[["frame", "obj_id", "y"]].groupby(by=["obj_id"]).agg(["median"])
    obj_ids = median_y['y'].query('median > ' + str(ymin) + ' and median < ' + str(ymax) ).index.values
    df_3d_filtered = df_3d[df_3d.obj_id.isin(obj_ids)]

    # only trajectories that are on average in the middle of the x dimension
    median_x = df_3d_filtered[["frame", "obj_id", "x"]].groupby(by=["obj_id"]).agg(["median"])
    obj_ids = median_x['x'].query('median > ' + str(xmin) + ' and median < ' + str(xmax) ).index.values
    
    return obj_ids

def get_trajectories_that_travel_far(df_3d, xdist_travelled=0.1):
    '''
    Use fancy pandas stuff to get a list of the object id numbers where the xdistance travelled is large

    Inputs
    ------
    df_3d  ------------ (dataframe) pandas dataframe from braidz_filemanager.load_filename_as_dataframe_3d
    xdist_travelled  -- (float) minimum distance travelled

    Returns
    -------
    obj_ids  ------- (list) of object id numbers

    '''

    min_x = df_3d[["frame", "obj_id", "x"]].groupby(by=["obj_id"]).agg(["min"])
    max_x = df_3d[["frame", "obj_id", "x"]].groupby(by=["obj_id"]).agg(["max"])
    s = max_x[(    'x', 'max')][np.abs((max_x[(    'x', 'max')] - min_x[(    'x', 'min')]))>xdist_travelled]
    obj_ids = df_3d[df_3d.obj_id.isin(s.index.values)].obj_id.unique()
    return obj_ids

def get_data_frame_slice_from_obj_ids(df_3d, objids):
    df_3d_culled = df_3d[df_3d.obj_id.isin(objids)]
    return df_3d_culled

def assign_unique_id(braid_df, name):
    """This Function adds a unique ID column to the braid data frame"""
    braid_df['obj_id_unique'] = braid_df['obj_id'].apply(lambda x: name + '_' + str(x))
    return braid_df

def get_obj_ids_that_stay_in_volume(df_3d, obj_id_key='obj_id', zmin=0.1, zmax=0.4, ymin=-0.15, ymax=0.15, xmin=-0.5, xmax=0.5):
    '''
    Use fancy pandas stuff to get a list of the object id numbers for which the median x,y,z values lie within the indicate range. 

    Inputs
    ------
    df_3d  --------- (dataframe) pandas dataframe from braidz_filemanager.load_filename_as_dataframe_3d
    obj_id_key ----- (str) which column to use as obj_id, consider using obj_id_unique_event
    zmin   --------- (float) minimum zval
    zmax   --------- (float) maximum zval
    ymin, ymax  ---- (float)
    xmin, xmax  ---- (float)

    Returns
    -------
    obj_ids  ------- (list) of object id numbers

    '''

    min_z = df_3d[[obj_id_key, "z"]].groupby(by=[obj_id_key]).agg(["min"])
    obj_ids = min_z['z'][(min_z['z']['min'] > zmin)==True].index.values
    df_3d_filtered = df_3d[df_3d[obj_id_key].isin(obj_ids)]

    df_3d = df_3d_filtered
    max_z = df_3d[[obj_id_key, "z"]].groupby(by=[obj_id_key]).agg(["max"])
    obj_ids = max_z['z'][(max_z['z']['max'] < zmax)==True].index.values
    df_3d_filtered = df_3d[df_3d[obj_id_key].isin(obj_ids)]

    df_3d = df_3d_filtered
    min_y = df_3d[[obj_id_key, "y"]].groupby(by=[obj_id_key]).agg(["min"])
    obj_ids = min_y['y'][(min_y['y']['min'] > ymin)==True].index.values
    df_3d_filtered = df_3d[df_3d[obj_id_key].isin(obj_ids)]

    df_3d = df_3d_filtered
    max_y = df_3d[[obj_id_key, "y"]].groupby(by=[obj_id_key]).agg(["max"])
    obj_ids = max_y['y'][(max_y['y']['max'] < ymax)==True].index.values
    df_3d_filtered = df_3d[df_3d[obj_id_key].isin(obj_ids)]

    df_3d = df_3d_filtered
    min_x = df_3d[[obj_id_key, "x"]].groupby(by=[obj_id_key]).agg(["min"])
    obj_ids = min_x['x'][(min_x['x']['min'] > xmin)==True].index.values
    df_3d_filtered = df_3d[df_3d[obj_id_key].isin(obj_ids)]

    df_3d = df_3d_filtered
    max_x = df_3d[[obj_id_key, "x"]].groupby(by=[obj_id_key]).agg(["max"])
    obj_ids = max_x['x'][(max_x['x']['max'] < xmax)==True].index.values
    df_3d_filtered = df_3d[df_3d[obj_id_key].isin(obj_ids)]
    
    return obj_ids