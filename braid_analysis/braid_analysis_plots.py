import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pynumdiff

from matplotlib import patches
from matplotlib.collections import PatchCollection

from braid_analysis import flymath

import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

def plot_3d_trajectory(df_3d, 
                       obj_ids=None,
                       frame_range=None,
                       mode='time', 
                       color=None,
                       xlim=[-0.5, 0.5], ylim=[-0.25, 0.25], zlim=[0, 0.5]):
    '''
    either obj_ids or frame_range must not be none
    
    obj_ids can be a list, in which case, multiple trajecs will be plotted
    frame_range should be a list with first and last frame of interest
    
    mode: 
       time = plot x,y,z,speed vs. time
       3d   = plot x vs y, x vs z
       
    '''
    if obj_ids is None:
        assert frame_range != None
        df_3d_q = df_3d.query('frame > ' + str(frame_range[0]) + ' and frame < ' + str(frame_range[-1]) )
        obj_ids = df_3d_q.obj_id.unique()
    
    try: 
        _ = obj_ids[0]
    except:
        obj_ids = [obj_ids]

    fig = plt.figure()
    
    # XY XZ planes
    if mode == '3d':
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for oid in obj_ids:
            trajec = df_3d[df_3d.obj_id==oid]
            ax1.plot(trajec.x, trajec.y)
            ax2.plot(trajec.x, trajec.z)

        for i, ax in enumerate([ax1, ax2]):
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            if i == 0:
                ax.set_ylabel('y')
            else:
                ax.set_ylabel('z')
    
    # X,Y,Z,speed vs time
    elif mode == 'time':
        ax1 = fig.add_subplot(411)
        ax1.set_ylim(*xlim)
        ax1.set_xlim(*frame_range)
        ax1.set_ylabel('x pos')
        ax2 = fig.add_subplot(412)
        ax2.set_ylim(*ylim)
        ax2.set_xlim(*frame_range)
        ax2.set_ylabel('y pos')
        ax3 = fig.add_subplot(413)
        ax3.set_ylim(*zlim)
        ax3.set_xlim(*frame_range)
        ax3.set_ylabel('z pos')
        ax4 = fig.add_subplot(414)
        ax4.set_ylim(0, 1.5)
        ax4.set_xlim(*frame_range)
        ax4.set_ylabel('speed')
        ax4.set_xlabel('Frames')
        
        for oid in obj_ids:
            trajec = df_3d[df_3d.obj_id==oid]
            line, = ax1.plot(trajec.frame, trajec.x)
            ax1.fill_between(trajec.frame, trajec.x+trajec.P00**0.5, trajec.x-trajec.P00**0.5, 
                             color=line.get_color(), alpha=0.3)
            
            ax2.plot(trajec.frame, trajec.y)
            ax2.fill_between(trajec.frame, trajec.y+trajec.P11**0.5, trajec.y-trajec.P11**0.5, 
                             color=line.get_color(), alpha=0.3)
            
            ax3.plot(trajec.frame, trajec.z)
            ax3.fill_between(trajec.frame, trajec.z+trajec.P22**0.5, trajec.z-trajec.P22**0.5, 
                             color=line.get_color(), alpha=0.3)
            
            speed = np.sqrt(trajec.xvel**2 + trajec.yvel**2 + trajec.zvel**2)
            ax4.plot(trajec.frame, speed)
            
def plot_2d_data(df_2d, df_3d, frame_range=None):
    first_frame, last_frame = frame_range
    
    
    if first_frame is None:
        first_frame = df_2d.frame.values[0]
    if last_frame is None:
        last_frame = df_2d.frame.values[-1]
        
    frames = np.arange(first_frame, last_frame+1)
    df_3d_traj = df_3d[df_3d['frame'].isin(frames)]
    
    color = np.array([0.5]*len(frames))
    color[df_3d_traj.frame.values - frames[0]] = 1

    # get 2d data 
    df_2d_traj = df_2d[df_2d['frame'].isin(frames)]

    # Find all cameras
    camns = df_2d_traj.camn.unique()
    camns = np.sort(camns)

    # plot the data
    fig = plt.figure(figsize=(10,10))
    xticks = [first_frame, last_frame]
    for camn in camns:
        ax = fig.add_subplot(len(camns), 2, 2*camn+1)
        df_2d_traj_camn_frames = df_2d_traj[df_2d_traj['camn']==camn].frame.values
        ax.scatter( df_2d_traj[df_2d_traj['camn']==camn].frame, 
                    df_2d_traj[df_2d_traj['camn']==camn].x,
                    c=color[df_2d_traj_camn_frames-first_frame], vmin=0.5, vmax=1)
        ax.set_ylabel('Cam: ' + str(camn))

        # plot labels
        #ax.set_xlim(first_frame, last_frame)
        ax.set_xticks(xticks)
        ax.set_ylim(0, 800)
        ax.set_yticks([0, 800])
        if camn==0:
            ax.set_title('x pixel')
        if camn != camns[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Frames')

        ax = fig.add_subplot(len(camns), 2, 2*camn+2)
        ax.scatter(df_2d_traj[df_2d_traj['camn']==camn].frame, 
                   df_2d_traj[df_2d_traj['camn']==camn].y,
                   c=color[df_2d_traj_camn_frames-first_frame], vmin=0.5, vmax=1)

        # plot labels
        #ax.set_xlim(first_frame, last_frame)
        ax.set_xticks(xticks)
        ax.set_ylim(0, 800)
        ax.set_yticks([0, 800])
        if camn==0:
            ax.set_title('y pixel')
        if camn != camns[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Frames')

def plot_2d_datums_per_camera(df_2d, df_3d, frame_range=None):
    first_frame, last_frame = frame_range
    
    
    if first_frame is None:
        first_frame = df_2d.frame.values[0]
    if last_frame is None:
        last_frame = df_2d.frame.values[-1]
        
    frames = np.arange(first_frame, last_frame+1)
    df_3d_traj = df_3d[df_3d['frame'].isin(frames)]
    
    color = np.array([0.5]*len(frames))
    color[df_3d_traj.frame.values - frames[0]] = 1

    # get 2d data 
    df_2d_traj = df_2d[df_2d['frame'].isin(frames)]

    # Find all cameras
    camns = df_2d_traj.camn.unique()
    camns = np.sort(camns)

    # plot the data
    fig = plt.figure(figsize=(15,10))
    xticks = [first_frame, last_frame]
    datums_per_camera = []
    for camn in camns:
        ax = fig.add_subplot(len(camns)+2, 1, camn+1)
        
        df_2d_traj_camn_frames = df_2d_traj[df_2d_traj['camn']==camn].frame.unique()
        df_2d_traj['x_not_nan'] = ~np.isnan(df_2d_traj.x)
        n_datums = df_2d_traj[df_2d_traj['camn']==camn].groupby('frame').sum().x_not_nan
        
        ax.plot( df_2d_traj_camn_frames, 
                    n_datums, c='black')
        ax.set_ylabel('Cam: ' + str(camn))

        # plot labels
        #ax.set_xlim(first_frame, last_frame)
        ax.set_xticks(xticks)
        ax.set_xlim(*frame_range)
        ax.set_ylim(0, 3)
        ax.set_yticks([0, 1, 2, 3])
        if camn==0:
            ax.set_title('number of 2d datums')
        if camn != camns[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([])
            #ax.set_xlabel('Frames')

        datums_per_camera.append(n_datums)
        
    # how many cameras total?
    ax_all = fig.add_subplot(len(camns)+2, 1, len(camns)+1)
    ax_all.plot(df_2d_traj_camn_frames, np.sum(datums_per_camera, axis=0))
    ax_all.set_ylabel('N cams')
    ax_all.set_ylim(0,9)
    ax_all.set_yticks([0,3,6,9])
    ax_all.set_xlim(*frame_range)
            
    # now plot x position of all the objects to see ghosts
    df_3d_q = df_3d.query('frame > ' + str(frame_range[0]) + ' and frame < ' + str(frame_range[-1]) )
    obj_ids = df_3d_q.obj_id.unique()
    
    ax_x = fig.add_subplot(len(camns)+2, 1, len(camns)+2)
    for oid in obj_ids:
        trajec = df_3d[df_3d.obj_id==oid]
        ax_x.plot(trajec.frame, trajec.x)
        ax_x.set_xlim(*frame_range)
    
    ax_x.set_xticks(xticks)
    ax_x.set_ylim(-0.3)
    ax_x.set_yticks([0.3])
    ax_x.set_xlabel('Frames')
    ax_x.set_ylabel('x pos')

def plot_occupancy_heatmaps(df_3d, xmin, xmax, ymin, ymax, zmin, zmax, 
                            resolution=0.01, ax_xz=None, ax_xy=None, cmap='magma',
                            log=True, vmin=0, vmax=0.001, norm_by_frames=True):

    res = resolution
    binx = np.arange(xmin, xmax+res, res)
    biny = np.arange(ymin, ymax+res, res)
    binz = np.arange(zmin, zmax+res, res)

    # for log
    eps = 1e-12 # for log
    if log:
        vmin = np.log(eps)
        vmax = np.log(vmax)
    
    if ax_xz is None or ax_xy is None:
        fig = plt.figure(figsize=(10,5))
        ax_xz = fig.add_subplot(121)
        ax_xy = fig.add_subplot(122) 

    # xz
    Hxz, xedges, zedges = np.histogram2d(df_3d['x'], df_3d['z'], bins=[binx, binz])
    if norm_by_frames:
        Hxz /= np.sum(Hxz)
    if log:
        Hxz = np.log(Hxz + eps)
    ax_xz.imshow(Hxz.T, origin="lower", 
               extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.set_xlim(xmin, xmax)
    ax_xz.set_ylim(zmin, zmax)
    ax_xz.set_aspect('equal')

    ax_xz.set_xlabel('x position')
    ax_xz.set_ylabel('z position')



    # xy
    Hxy, xedges, yedges = np.histogram2d(df_3d['x'], df_3d['y'], bins=[binx, biny])
    if norm_by_frames:
        Hxy /= np.sum(Hxy)
    if log:
        Hxy = np.log(Hxy + eps)
    ax_xy.imshow(Hxy.T, origin="lower", 
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)
    ax_xy.set_aspect('equal')

    ax_xy.set_xlabel('x position')
    ax_xy.set_ylabel('y position')

def plot_starting_and_ending_points(df_3d, obj_id_key,
                                    xmin, xmax, ymin, ymax, zmin, zmax, 
                                    start_or_end='start',
                                    padding=0.1, ax_xz=None, ax_xy=None,
                                    dot_size=1):

    df_3d = df_3d[~df_3d.xvel.isna()].copy()

    if start_or_end == 'start':
        key_frames = df_3d.loc[df_3d.groupby(obj_id_key).frame.idxmin()]
        color = 'green'
    elif start_or_end == 'end':
        key_frames = df_3d.loc[df_3d.groupby(obj_id_key).frame.idxmax()]
        color = 'red'

    if ax_xy is None or ax_xz is None:
        fig = plt.figure(figsize=(10,5))
        ax_xz = fig.add_subplot(121)
        ax_xy = fig.add_subplot(122) 

    ax_xz.scatter(key_frames.x.values, key_frames.z.values, c=color, s=dot_size)
    ax_xz.set_xlim(xmin - padding, xmax + padding)
    ax_xz.set_ylim(zmin - padding, zmax + padding)
    ax_xz.set_aspect('equal')
    ax_xz.set_xticks(np.arange(xmin, xmax+0.2, 0.2))
    ax_xz.set_yticks(np.arange(zmin, zmax+0.2, 0.2))
    ax_xz.set_xlabel('x position, m')
    ax_xz.set_ylabel('z position, m')

    ax_xy.scatter(key_frames.x.values, key_frames.y.values, c=color, s=dot_size)
    ax_xy.set_xlim(xmin - padding, xmax + padding)
    ax_xy.set_ylim(ymin - padding, ymax + padding)
    ax_xy.set_aspect('equal')
    ax_xy.set_xticks(np.arange(xmin, xmax+0.2, 0.2))
    ax_xy.set_yticks(np.arange(ymin, ymax+0.2, 0.2))
    ax_xy.set_xlabel('x position, m')
    ax_xy.set_ylabel('y position, m')

def plot_speed_xy_histogram(df_3d, ax=None, bins=None, speed_key='speed_xy'):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if bins is None:
        bins = np.arange(0, 1.5, 0.01)

    results = ax.hist(df_3d[speed_key], bins)

    ax.set_xlabel(speed_key)
    ax.set_ylabel('Count')

def plot_length_of_trajectories_histogram(df_3d, obj_id_key="obj_id", ax=None, dt=0.01, bins=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    number_frames_per_obj_id = df_3d[["frame", obj_id_key]].groupby(by=[obj_id_key]).agg(["count"])
    n_frames_per_trajec = number_frames_per_obj_id.frame.values

    if bins is None:
        bins = np.arange(0, 20, 0.3)
    
    results = ax.hist(n_frames_per_trajec*dt, bins)

    ax.set_xlabel('trajectory length, sec')
    ax.set_ylabel('Count')

def plot_xy_trajectory_with_color_overlay(df_3d_trajec_slice, 
                                          obj_id_key='obj_id_unique',
                                          column_for_color='ang_vel_smoother',
                                          plane = 'xy', # xy or xz or yz
                                          cmap='seismic', 
                                          vmin=-50, vmax=50,
                                          dot_size=2,
                                          ax=None):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.set_aspect('equal')

    if plane == 'xy' or 'xz':
        xval = df_3d_trajec_slice.x.values
    elif plane == 'yz':
        yval = df_3d_trajec_slice.y.values

    if plane == 'xy':
        yval = df_3d_trajec_slice.y.values
    elif plane == 'xz' or plane == 'yz':
        yval = df_3d_trajec_slice.z.values

    ax.plot(xval, yval, color='gray', alpha=0.3)
    ax.scatter(xval, yval, c=df_3d_trajec_slice[column_for_color].values, s=dot_size, cmap=cmap,
               vmin=vmin, vmax=vmax)

    obj_id = df_3d_trajec_slice[obj_id_key].values[0]
    ax.set_title(obj_id)

def plot_column_vs_time(df_3d,
                        column='course_smoothish',
                        time_key='time_relative_to_flash',
                        norm_columns_to_min_max=False,
                        norm_columns_to_sum=False,
                        norm_columns_to_min_max_smoothing=0,
                        cmap='bone_r',
                        vmin=0, vmax=1,
                        bin_y=None,
                        bin_x=None,
                        res_y=0.01,
                        res_x=0.01,
                        ax=None,
                        interpolation='nearest',
                        return_array=False,
                          ):
    '''
    norm_columns_to_min_max -- make sure that each column in time has a minimum of 0 and a maxmimum of 1
    norm_columns_to_sum -- make sure that each column in time sums to 1
    norm_columns_to_min_max_smoothing -- if using norm_columns_to_min_max, smooth the min and max to remove aliasing artifacts. 
                                         Integer value for window size in a sliding window smoother corresponding to pynumdiff.meandiff

    '''

    if norm_columns_to_sum and norm_columns_to_min_max:
        raise ValueError('Choose at most one norm option, do not set both to True')

    if bin_y is None:
        ymin = df_3d[column].min()
        ymax = df_3d[column].max()
        bin_y = np.arange(ymin, ymax+res_y, res_y)

    if bin_x is None:
        xmin = df_3d[time_key].min()
        xmax = df_3d[time_key].max()
        bin_x = np.arange(xmin+res_x/2, xmax+res_x+res_x/2, res_x)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    M, time_edges, column_edges = np.histogram2d(df_3d[time_key], df_3d[column], 
                                     bins=[bin_x, bin_y])

    if norm_columns_to_min_max:
        norm_min = np.min(M, axis=1) 
        
        if norm_columns_to_min_max_smoothing > 0:
            norm_min, _ = pynumdiff.meandiff(norm_min,1,[norm_columns_to_min_max_smoothing])

        M_min = M - norm_min[:,None]
        norm_max = np.nanmax(M_min, axis=1)

        if norm_columns_to_min_max_smoothing > 0:
            norm_max, _ = pynumdiff.meandiff(norm_max,1,[norm_columns_to_min_max_smoothing])

        M_min_max = M_min / norm_max[:,None]
        M = M_min_max

    if norm_columns_to_sum:
        norm_sum = np.nansum(M, axis=1) 
        M_norm_sum = M / norm_sum[:,None]
        M = M_norm_sum

    ax.imshow(M.T, origin="lower", 
           extent=[time_edges[0], time_edges[-1], column_edges[0], column_edges[-1]], 
           cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)

    ax.set_aspect('auto')

    ax.set_ylabel(column)
    ax.set_xlabel(time_key)

    ax.set_ylim(column_edges[0], column_edges[-1])
    ax.set_xlim(time_edges[0], time_edges[-1])

    if return_array:
        return M.T, time_edges, column_edges

def add_stimulus_shading_to_ax(ax, df_3d, stimulus_column, 
                               ymax, ymin,
                               time_key='time_relative_to_flash',
                               obj_id_key='obj_id_unique_event', cmap='Reds',
                               vmin=0, vmax=1, alpha=0.3, zorder=1,
                               ):
    '''
    Show stimulus as a gradient polygon on the provided ax. 
    This function will find a "demo" trajectory and grab the stimulus_column key and use that to generate the shading.
    It assumes that all trajectories in the df_3d provided have the same stimulus_column values.

    '''
    obj_id_demo = df_3d[obj_id_key].unique()[0]
    trajec = df_3d[df_3d[obj_id_key]==obj_id_demo]
    x = trajec[time_key].values
    polygon = ax.fill_between(x, ymin*np.ones_like(x), ymax*np.ones_like(x), lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    gradient = ax.imshow(trajec.lights_on.values.reshape(1, -1), cmap=cmap, aspect='auto',
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()], 
                         vmin=vmin, vmax=vmax, 
                         zorder=zorder, alpha=alpha)
    #gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData) # <<< this causes issues when writing to an svg

#############################################################################################################
# Arrow head trajectories

def get_wedges_for_heading_plot(x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None, size_radius_range=(0.01,.1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True, deg=True, nskip=0, center_offset_fraction=0.75):
    '''
    Returns a Patch Collection of Wedges, with arbitrary color and orientation
    
    Outputs:
    Patch Collection
    
    Inputs:
    x, y        - x and y positions (np.array or list, each of length N)
    color       - values to color wedges by (np.array or list, length N), OR color string. 
       colormap - specifies colormap to use (string, eg. 'jet')
       norm     - specifies range you'd like to normalize to, 
                  if none, scales to min/max of color array (2-tuple, eg. (0,1) )
    orientation - angles are in degrees, use deg=False to convert radians to degrees
    size_radius - radius of wedge, in same units as x, y. Can be list or np.array, length N, for changing sizes
       size_radius_norm - specifies range you'd like to normalize size_radius to, if size_radius is a list/array
                  should be tuple, eg. (0.01, .1)
    size_angle  - angular extent of wedge, degrees. Can be list or np.array, length N, for changing sizes
    edgecolor   - color for lineedges, string or np.array of length N
    alpha       - transparency (single value, between 0 and 1)
    flip        - flip orientations by 180 degrees, default = True
    nskip       - allows you to skip between points to make the points clearer, nskip=1 skips every other point
    center_offset_fraction  - (float in range (0,1) ) - 0 means (x,y) is at the tip, 1 means (x,y) is at the edge
    '''
    cmap = plt.get_cmap(colormap)
    
    # norms
    if colornorm is None and type(color) is not str:
        colornorm = plt.Normalize(np.min(color), np.max(color))
    elif type(color) is not str:
        colornorm = plt.Normalize(colornorm[0], colornorm[1])
    if size_radius_norm is None:
        size_radius_norm = plt.Normalize(np.min(size_radius), np.max(size_radius), clip=True)
    else:
        size_radius_norm = plt.Normalize(size_radius_norm[0], size_radius_norm[1], clip=True)
        
    indices_to_plot = np.arange(0, len(x), nskip+1)
        
    # fix orientations
    if type(orientation) is list:
        orientation = np.array(orientation)
    if deg is False:
        orientation = orientation*180./np.pi
    if flip:
        orientation += 180
    
    flycons = []
    n = 0
    for i in indices_to_plot:
        # wedge parameters
        if type(size_radius) is list or type(size_radius) is np.array or type(size_radius) is np.ndarray: 
            r = size_radius_norm(size_radius[i])*(size_radius_range[1]-size_radius_range[0]) + size_radius_range[0] 
        else: r = size_radius
        
        if type(size_angle) is list or type(size_angle) is np.array or type(size_angle) is np.ndarray: 
            angle_swept = size_radius[i]
        else: angle_swept = size_radius
        theta1 = orientation[i] - size_angle/2.
        theta2 = orientation[i] + size_angle/2.
        
        center = [x[i], y[i]]
        center[0] -= np.cos(orientation[i]*np.pi/180.)*r*center_offset_fraction
        center[1] -= np.sin(orientation[i]*np.pi/180.)*r*center_offset_fraction
        
        wedge = patches.Wedge(center, r, theta1, theta2)
        flycons.append(wedge)
        
    # add collection and color it
    pc = PatchCollection(flycons, cmap=cmap, norm=colornorm)
    
    # set properties for collection
    pc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        if type(color) is list:
            color = np.asarray(color)
        pc.set_array(color[indices_to_plot])
    else:
        pc.set_facecolors(color)
    pc.set_alpha(alpha)
    
    return pc

def plot_arrowhead_trajectory(x, y, color='black', arrow_length=0.05, arrow_angle=30, ax=None, linewidth=1):

    # ignore nans
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    xvel = pynumdiff.finite_difference.second_order(x, 1)[1]
    yvel = pynumdiff.finite_difference.second_order(y, 1)[1]
    orientations = np.arctan2(yvel, xvel)

    last_orientation = flymath.mean_angle(orientations[-3:])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
    ax.plot(x,y, color=color, linewidth=linewidth)

    np.isnan(x)==False

    wedge = get_wedges_for_heading_plot([x[-1],],[y[-1],], color, [last_orientation*180/np.pi,], 
                                        size_radius=arrow_length, size_angle=arrow_angle)
    ax.add_collection(wedge)


def make_lights_time_cmap(lights_on, time):
    """
    Build a colormap and feature array that encode two experimental conditions:

    - ``time < 0``              → gray   (pre-stimulus baseline)
    - ``lights_on != 0``        → red    (lights-on period)
    - ``lights_on == 0, time >= 0`` → black  (lights-off, post-stimulus)

    The returned ``color_feature`` is an integer array with three sentinel
    values (0, 1, 2) that index into a ``ListedColormap``.  Pass both outputs
    directly to ``plot_arrowhead_trajectory_scaled`` via the ``cmap`` and
    ``color_feature`` arguments.

    Parameters
    ----------
    lights_on : array-like
        Per-point indicator of whether the lights are on (non-zero = on).
    time : array-like
        Per-point time values; negative values indicate the pre-stimulus period.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    color_feature : np.ndarray of int
        Integer array (same length as inputs) with values in {0, 1, 2}.
    """
    lights_on = np.asarray(lights_on)
    time      = np.asarray(time)

    if lights_on.shape != time.shape:
        raise ValueError("`lights_on` and `time` must have the same shape.")

    # Sentinel values: 0 = gray, 1 = red, 2 = black
    color_feature = np.full(len(time), 2, dtype=int)   # default: black
    color_feature[time < 0]       = 0                  # gray (overrides black)
    color_feature[lights_on != 0] = 1                  # red  (overrides gray/black)

    cmap = mcolors.ListedColormap(['gray', 'red', 'black'])

    return cmap, color_feature


def plot_arrowhead_trajectory_scaled(x, y, color='black', arrow_length=0.05, arrow_angle=30,
                                     ax=None, linewidth=1, scale_bar=False, units='',
                                     flow_direction=None, fontsize=5,
                                     flow_arrow_length=0.05, flow_arrow_angle=30,
                                     flow_arrow_size=0.08, padding=0.2,
                                     flow_column_width=0.2,
                                     color_feature=None, cmap=None,
                                     color_feature_min=None, color_feature_max=None,
                                     n_segments=None, rasterized=False):
    """
    Plot a trajectory with arrowheads, with optional per-point colormap coloring.

    Parameters
    ----------
    x, y : array-like
        Trajectory coordinates.
    color : str or color
        Fallback solid color when `cmap` / `feature` are not provided.
    color_feature : array-like, optional
        Scalar values (same length as x/y) used to color the trajectory via `cmap`.
        Typical choices: time index, speed, a sensor reading, etc.
    cmap : str or Colormap, optional
        A matplotlib colormap name (e.g. ``'viridis'``) or Colormap object.
        Required when `color_feature` is supplied; ignored otherwise.
    color_feature_min, color_feature_max : float, optional
        Clipping / normalisation bounds for `color_feature`.  Defaults to the data
        min / max when not provided.
    n_segments : int, optional
        Number of uniformly-coloured segments used to approximate the
        continuous colormap along the trajectory.  Defaults to
        ``max(50, len(x) // 4)``.  Finer values give smoother colour gradients
        but add more artists to the axes.
    arrow_length, arrow_angle, ax, linewidth, scale_bar, units,
    flow_direction, fontsize, flow_arrow_length, flow_arrow_angle,
    flow_arrow_size, padding, flow_column_width
        Same as before.
    """

    # ------------------------------------------------------------------ #
    # Resolve colormap mode vs solid-color mode
    # ------------------------------------------------------------------ #
    use_cmap = (color_feature is not None) and (cmap is not None)

    if use_cmap:
        color_feature = np.asarray(color_feature, dtype=float)
        if len(color_feature) != len(x):
            raise ValueError("`color_feature` must have the same length as `x` and `y`.")

        cmap_obj = mcm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        f_min = color_feature_min if color_feature_min is not None else np.nanmin(color_feature)
        f_max = color_feature_max if color_feature_max is not None else np.nanmax(color_feature)
        norm  = mcolors.Normalize(vmin=f_min, vmax=f_max)

        if n_segments is None:
            n_segments = max(50, len(x) // 4)

    has_flow = flow_direction is not None and flow_arrow_length is not None

    # ------------------------------------------------------------------ #
    # Physical axis size → coordinate space
    # ------------------------------------------------------------------ #
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    ax_width_in  = bbox.width
    ax_height_in = bbox.height

    coord_width  = ax_width_in
    coord_height = ax_height_in

    flow_col  = flow_column_width * coord_width if has_flow else 0.0
    left_x_min = padding * coord_width
    left_x_max = coord_width - flow_col - padding * coord_width
    y_min_pad  = padding * coord_height
    y_max_pad  = coord_height - padding * coord_height

    # Scale trajectory
    x_norm = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    y_norm = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
    x_scaled = x_norm * (left_x_max - left_x_min) + left_x_min
    y_scaled = y_norm * (y_max_pad  - y_min_pad)  + y_min_pad

    ax.set_xlim(0, coord_width)
    ax.set_ylim(0, coord_height)
    ax.set_aspect('equal')

    # ------------------------------------------------------------------ #
    # Draw trajectory
    # ------------------------------------------------------------------ #
    if use_cmap:
        # Interpolate trajectory and feature onto a uniform index grid so that
        # segments have consistent arc-length regardless of the original
        # sampling density.
        idx_orig = np.arange(len(x_scaled), dtype=float)
        idx_fine = np.linspace(0, len(x_scaled) - 1, n_segments + 1)
        xs_fine  = np.interp(idx_fine, idx_orig, x_scaled)
        ys_fine  = np.interp(idx_fine, idx_orig, y_scaled)
        feat_fine = np.interp(idx_fine, idx_orig, color_feature)

        # Draw each segment with the colour of its midpoint feature value.
        # We call plot_arrowhead_trajectory only on the last segment so that
        # a single arrowhead appears at the trajectory's tip.
        for i in range(n_segments):
            seg_x = xs_fine[i : i + 2]
            seg_y = ys_fine[i : i + 2]
            mid_feat = (feat_fine[i] + feat_fine[i + 1]) / 2.0
            seg_color = cmap_obj(norm(mid_feat))

            if i < n_segments - 1:
                ax.plot(seg_x, seg_y, color=seg_color, linewidth=linewidth,
                        solid_capstyle='round', rasterized=rasterized)
            else:
                # Last segment — use plot_arrowhead_trajectory for the tip arrowhead.
                # Pass a few extra points so the arrowhead direction is stable.
                tip_x = xs_fine[max(0, i - 1) :]
                tip_y = ys_fine[max(0, i - 1) :]
                _artists_before = set(ax.lines + ax.patches + list(ax.collections))
                plot_arrowhead_trajectory(
                    tip_x, tip_y,
                    color=seg_color,
                    arrow_length=arrow_length,
                    arrow_angle=arrow_angle,
                    ax=ax,
                    linewidth=linewidth,
                )
                if rasterized:
                    for _a in set(ax.lines + ax.patches + list(ax.collections)) - _artists_before:
                        _a.set_rasterized(True)
    else:
        _artists_before = set(ax.lines + ax.patches + list(ax.collections))
        plot_arrowhead_trajectory(
            x_scaled, y_scaled,
            color=color,
            arrow_length=arrow_length,
            arrow_angle=arrow_angle,
            ax=ax,
            linewidth=linewidth,
        )
        if rasterized:
            for _a in set(ax.lines + ax.patches + list(ax.collections)) - _artists_before:
                _a.set_rasterized(True)

    for collection in ax.collections:
        collection.set_clip_on(False)

    # ------------------------------------------------------------------ #
    # Re-read limits after set_aspect
    # ------------------------------------------------------------------ #
    ax.figure.canvas.draw()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_extent_ax = xlim[1] - xlim[0]
    y_extent_ax = ylim[1] - ylim[0]

    label_gap = 0.03 * y_extent_ax
    offset_x  = 0.02 * x_extent_ax

    right_x_min = coord_width - flow_col
    right_x_max = coord_width

    traj_hull    = MultiPoint(list(zip(x_scaled, y_scaled))).convex_hull.buffer(offset_x * 2)
    occupied_geom = traj_hull

    def find_best_position(size, occupied, x_min, x_max, y_min, y_max):
        n = 30
        xs_grid = np.linspace(x_min + size, x_max - size, n)
        ys_grid = np.linspace(y_min + size, y_max - size, n)
        best_pt, best_dist = None, -1
        for cx_ in xs_grid:
            for cy_ in ys_grid:
                pt = Point(cx_, cy_)
                d = pt.distance(occupied) if not occupied.is_empty else 1e9
                box_fits = (cx_ - size > x_min and cx_ + size < x_max and
                            cy_ - size > y_min and cy_ + size < y_max)
                if box_fits and d > best_dist:
                    best_dist = d
                    best_pt = (cx_, cy_)
        return best_pt

    # ------------------------------------------------------------------ #
    # Scale bar
    # ------------------------------------------------------------------ #
    if scale_bar:
        x_extent_original = np.nanmax(x) - np.nanmin(x)
        max_bar_original   = 0.3 * x_extent_original
        magnitude          = 10 ** np.floor(np.log10(max_bar_original))
        nice_steps         = [1, 2, 5]
        bar_size_original  = magnitude
        for step in nice_steps:
            candidate = step * magnitude
            if candidate <= max_bar_original:
                bar_size_original = candidate

        scale_factor   = (left_x_max - left_x_min) / x_extent_original
        bar_size_scaled = bar_size_original * scale_factor
        bar_half        = bar_size_scaled / 2

        pos = find_best_position(max(bar_half, label_gap * 2), occupied_geom,
                                 xlim[0], right_x_min, ylim[0], ylim[1])
        if pos is not None:
            cx, cy = pos
            bar_x_start = cx - bar_half
            bar_x_end   = cx + bar_half

            if cy < (ylim[0] + ylim[1]) / 2:
                text_y, va = cy + label_gap, 'bottom'
            else:
                text_y, va = cy - label_gap, 'top'

            ax.plot([bar_x_start, bar_x_end], [cy, cy],
                    color=color, linewidth=0.5)
            ax.text(cx, text_y, f'{bar_size_original:g} {units}',
                    ha='center', va=va, fontsize=fontsize, color=color)

            bar_geom = MultiPoint([
                (bar_x_start, cy), (bar_x_end, cy), (cx, text_y)
            ]).convex_hull.buffer(label_gap * 2)
            occupied_geom = unary_union([occupied_geom, bar_geom])

    # ------------------------------------------------------------------ #
    # Flow direction arrow
    # ------------------------------------------------------------------ #
    if has_flow:
        arrow_size_scaled = flow_arrow_size * x_extent_ax

        cx = (right_x_min + right_x_max) / 2
        cy = (ylim[0] + ylim[1]) / 2

        dx = np.cos(flow_direction) * arrow_size_scaled / 2
        dy = np.sin(flow_direction) * arrow_size_scaled / 2

        n_points = 10
        t = np.linspace(-0.5, 0.5, n_points)
        arrow_x = cx + t * dx * 2
        arrow_y = cy + t * dy * 2

        plot_arrowhead_trajectory(
            arrow_x, arrow_y,
            color=color,
            arrow_length=flow_arrow_length,
            arrow_angle=flow_arrow_angle,
            ax=ax,
            linewidth=linewidth,
        )

        for collection in ax.collections:
            collection.set_clip_on(False)

        text_angle_deg = np.degrees(flow_direction)
        if 90 < text_angle_deg % 360 < 270:
            text_angle_deg += 180

        perp_dx = -np.sin(flow_direction) * label_gap * 3
        perp_dy =  np.cos(flow_direction) * label_gap * 3
        cx_mid  = (right_x_min + right_x_max) / 2
        cy_mid  = (ylim[0] + ylim[1]) / 2
        if (cx + perp_dx - cx_mid)**2 + (cy + perp_dy - cy_mid)**2 < \
           (cx - perp_dx - cx_mid)**2 + (cy - perp_dy - cy_mid)**2:
            perp_dx, perp_dy = -perp_dx, -perp_dy

        ax.text(cx + perp_dx, cy + perp_dy, 'flow',
                ha='center', va='center', fontsize=fontsize, color=color,
                rotation=text_angle_deg, rotation_mode='anchor')

def make_trajectory_plot_page(braid_df, 
                              n_col = 8,
                              n_row = 10,
                              starting_objid_ix = 0,
                              fig_width=8,
                              fig_height=10,
                              show_obj_id=True):
    '''
    Function for plotting lots of trajectories nicely on a page -- useful for picking a good demo trajectory.
    This function is quite slow because it uses plot_arrowhead_trajectory_scaled.

    If you want to view a new set of trajectories, run the same function with starting_objid_ix set to where the last page left off.
    '''

    fig = plt.figure(figsize=(fig_width,fig_height))
    
    for i in range(starting_objid_ix, starting_objid_ix + n_col*n_row):
        ax = fig.add_subplot(n_row, n_col, i-starting_objid_ix+1)
        
        # get a single trajectory
        obj_id_key = 'obj_id_unique_event'
        if i < len(braid_df[obj_id_key].unique()):
            obj_id = braid_df[obj_id_key].unique()[i] # << 36 is a good demo, 40 is beautiful, 41 is nice
            trajec = braid_df[braid_df[obj_id_key]==obj_id]
            trajec = trajec.dropna()
        
            x_pos = trajec.x.values
            y_pos = trajec.y.values
            cmap, color_feature = make_lights_time_cmap(trajec.lights_on, trajec.time_relative_to_flash)
            plot_arrowhead_trajectory_scaled(x_pos, y_pos, color='black', arrow_length=0.1, arrow_angle=30,
                                             ax=ax, linewidth=0.75, scale_bar=True, units='m', fontsize=6, 
                                             padding=0.05,
                                             flow_direction=None, flow_column_width=0,
                                             cmap=cmap,
                                             color_feature=color_feature,
                                             color_feature_min=0,   # pin the norm so sentinel values map correctly
                                             color_feature_max=2,
                                             rasterized=True)
            ax.set_title(obj_id, fontsize=4)
            
        fifi.mpl_functions.adjust_spines(ax, [])

    return fig
        