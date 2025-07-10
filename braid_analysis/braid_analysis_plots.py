import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pynumdiff

from matplotlib import patches
from matplotlib.collections import PatchCollection

from braid_analysis import flymath

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
        bin_x = np.arange(xmin+res/2, xmax+res+res/2, res)

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
    xvel = pynumdiff.finite_difference.second_order(x, 1)[1]
    yvel = pynumdiff.finite_difference.second_order(y, 1)[1]
    orientations = np.arctan2(yvel, xvel)

    last_orientation = flymath.mean_angle(orientations[-3:])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
    ax.plot(x,y, color=color, linewidth=linewidth)
    wedge = get_wedges_for_heading_plot([x[-1],],[y[-1],], color, [last_orientation*180/np.pi,], 
                                        size_radius=arrow_length, size_angle=arrow_angle)
    ax.add_collection(wedge)
