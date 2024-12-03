import sys
import zipfile
from braid_analysis import bag2hdf5_python3 as bag2hdf5
import h5py
import os

try:
    import urllib.request # requires Python 3
    urlparse = urllib.parse.urlparse
except:
    import urllib2 as urllib
    print('Load braidz files using python 3!!')

import io
import pandas as pd

###################################################################################
# Generic helpful functions

def get_filenames(path, contains, does_not_contain=['~', '.pyc']):
    cmd = 'ls ' + '"' + path + '"'
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
    filelist = []
    for i, filename in enumerate(all_filelist):
        if contains in filename:
            fileok = True
            for nc in does_not_contain:
                if nc in filename:
                    fileok = False
            if fileok:
                filelist.append( os.path.join(path, filename) )
    return filelist
    
def get_filename(path, contains, does_not_contain=['~', '.pyc']):
    filelist = get_filenames(path, contains, does_not_contain)
    if len(filelist) == 1:
        return filelist[0]
    elif len(filelist) > 0 and 'bgimg' in contains:
        pick = sorted(filelist)[-1]
        print('Found multiple background images, using ' + str(pick))
        return pick
    else:
        print (filelist)
        print ('Found too many, or too few files')
    return None
            
def load_bag_as_hdf5(bag_filename, skip_messages=[]):
    output_fname = bag_filename.split('.')[0] + '.hdf5'
    print (output_fname)
    if not os.path.exists(output_fname):
        bag2hdf5.bag2hdf5(   bag_filename,
                             output_fname,
                             max_strlen=200,
                             skip_messages=skip_messages)    
    metadata = h5py.File(output_fname, 'r')
    return metadata

def get_pandas_dataframe_from_uncooperative_hdf5(filename, key='first_key'):
    '''

    '''
    f = h5py.File(filename,'r')
    all_keys = list(f.keys())
    if key == 'first_key':
        if len(all_keys) > 1:
            print('all keys: ')
            print(all_keys)
            print('WARNING: loading first key only')
        key = all_keys[0]
    data = f[key][()]
    dic = {}
    for column_label in data.dtype.fields.keys():
        dic.setdefault(column_label, data[column_label])
    df = pd.DataFrame(dic)
    return df

def save_preprocessed_braidz(   data_directory, braid_df_culled, 
                                sub_directory='preprocessed_data', suffix='_preprocessed'):
    braidz_filename = get_filename(data_directory, '.braidz')
    preprocessed_data_dir = os.path.join(data_directory, sub_directory)
    if not os.path.isdir(preprocessed_data_dir):
        os.mkdir(preprocessed_data_dir)

    preprocessed_data_fname = os.path.basename(braidz_filename).split('.')[0] + suffix + '.hdf'
    fname = os.path.join(preprocessed_data_dir, preprocessed_data_fname)
    braid_df_culled.to_hdf(fname, 'DATA_' + os.path.basename(braidz_filename).split('.')[0] )
    
    return fname

def preprocessed_braidz_exists(   data_directory, 
                                sub_directory='preprocessed_data', suffix='_preprocessed'):
    braidz_filename = get_filename(data_directory, '.braidz')
    preprocessed_data_dir = os.path.join(data_directory, sub_directory)

    preprocessed_data_fname = os.path.basename(braidz_filename).split('.')[0] + suffix + '.hdf'
    fname = os.path.join(preprocessed_data_dir, preprocessed_data_fname)
    
    return os.path.exists(fname)

###################################################################################

###################################################################################
# Braid specific functions

def open_filename_or_url(filename_or_url):
    '''
    Opens filename as object for reading.

    Inputs
    ------
    filename_or_url -- (str) filename

    '''
    parsed = urlparse(filename_or_url)
    is_windows_drive = len(parsed.scheme) == 1
    if is_windows_drive or parsed.scheme=='':
        # no scheme, so this is a filename.
        fileobj_with_seek = open(filename_or_url,mode='rb')
    else:
        # Idea for one day: implement HTTP file object reader that implements
        # seek using HTTP range requests.
        fileobj = url_request.urlopen(filename_or_url)
        fileobj_with_seek = io.BytesIO(fileobj.read())
    return fileobj_with_seek

def load_filename_as_dataframe_2d(filename_or_url, frame_range=None):
    '''
    Returns the 2d data from a braidz file as a pandas dataframe.

    Inputs
    ------
    filename_or_url ------ (str) filename
    frame_range  --------- (list of ints) [first frame, last frame]

    Returns
    -------
    data2d_distorted_df -- (dataframe)
    
    '''
    fileobj = open_filename_or_url(filename_or_url)

    with zipfile.ZipFile(file=fileobj, mode='r') as archive:
        cam_info_df = pd.read_csv(
            archive.open('cam_info.csv.gz'),
            comment="#",
            compression='gzip')

        camn2camid = {}
        for i, row in cam_info_df.iterrows():
            camn2camid[row['camn']] = row['cam_id']

        cam_ids = list(cam_info_df['cam_id'].values)
        cam_ids.sort()
        data2d_distorted_df = pd.read_csv(
            archive.open('data2d_distorted.csv.gz'),
            comment="#",
            compression='gzip')
    
    if frame_range is None:
        return data2d_distorted_df
    else:
        return data2d_distorted_df.query('frame > ' + str(frame_range[0]) + \
                                         ' and frame < ' + str(frame_range[-1]))

def load_filename_as_dataframe_3d(filename_or_url, frame_range=None):
    '''
    Returns the 3d data from a braidz file as a pandas dataframe.

    Inputs
    ------
    filename_or_url ------ (str) filename
    frame_range  --------- (list of ints) [first frame, last frame]

    Returns
    -------
    df -- (dataframe)
    
    '''

    fileobj = open_filename_or_url(filename_or_url)

    with zipfile.ZipFile(file=fileobj, mode='r') as archive:
        df = pd.read_csv(
            archive.open('kalman_estimates.csv.gz'),
            comment="#",
            compression='gzip')
    
    if frame_range is None:
        return df
    else:
        return df.query('frame > ' + str(frame_range[0]) + \
                        ' and frame < ' + str(frame_range[-1]))

###################################################################################

