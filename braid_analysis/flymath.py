import numpy as np
import pynumdiff
import cvxpy
import pandas as pd
import multiprocessing
import scipy.signal
import matplotlib.pyplot as plt
from braid_analysis import braid_analysis_plots

########################################################################################################
## Course direction and angular velocity functions

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def median_angle(angle):
    return np.arctan2(np.median(np.sin(angle)), np.median(np.cos(angle)))

def mean_angle(angle):
    return np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))

def subtract_angles(angle1, angle2, radians=True):
    '''
    return signed smallest difference between (angle1 - angle2)
    '''
    a = angle1 - angle2
    if radians:
        a = a*180/np.pi

    d = (a+180) % 360 - 180

    if radians:
        return d*np.pi/180
    else:
        return d

def unwrap_angle(z, correction_window_for_2pi=5, n_range=2, plot=False):
    # automatically scales n_range to most recent value, and maybe faster
    smooth_zs = np.array(z[0:2])
    for i in range(2, len(z)):
        first_ix = np.max([0, i-correction_window_for_2pi])
        last_ix = i
        
        nbase = np.round( (smooth_zs[-1] - z[i])/(2*np.pi) )
        
        candidates = []
        for n in range(-1*n_range, n_range):
            candidates.append(n*2*np.pi+nbase*2*np.pi+z[i])
        error = np.abs(candidates - np.mean(smooth_zs[first_ix:last_ix])) 
        smooth_zs = np.hstack(( smooth_zs, [candidates[np.argmin(error)]] ))
    if plot:
        plt.plot(smooth_zs, '.', color='black', markersize=1)
    return smooth_zs

def get_course_and_raw_angular_velocity(xvel, yvel, dt=0.01, correction_window_for_2pi=5, butter_filter_params=[2,0.5]):
    '''
    get course, and do some very minimal smoothing to get a raw angular velocity and smoothish course direction.
    '''
    course = np.arctan2(yvel, xvel)
    course_unwrapped = unwrap_angle(course, correction_window_for_2pi=correction_window_for_2pi)
    course_smoothish, ang_vel_smoothish = pynumdiff.smooth_finite_difference.butterdiff(course_unwrapped, dt, butter_filter_params)
    
    return course, course_smoothish, ang_vel_smoothish

def diff_angle(angles, dt, params, 
               derivative_method='smooth_finite_difference.butterdiff', 
               correction_window_for_2pi=5):
    '''
    Take a filtered derivative of an angle
    '''
    family, method = derivative_method.split('.')
    unwrapped_angle = unwrap_angle(angles, correction_window_for_2pi=correction_window_for_2pi, n_range=5)
    angles_smooth, angles_dot = pynumdiff.__dict__[family].__dict__[method](unwrapped_angle, dt, params, {})

    return wrap_angle(angles_smooth), angles_dot

def get_convex_smoothed_course_and_ang_vel(course, xvel, yvel, dt=0.01, butter_filter_params=[1,0.1], correction_window_for_2pi=5, gamma=0.01, correction_gamma=0.1):
    '''
    Generates a "very" smooth angular velocity and angle. Robust to slow and noisy trajectories.
    '''
    theta_meas = course 
    
    # Get a smoothed estimate of tan(course)
    tan_theta_cvx = cvxpy.Variable(len(theta_meas))
    vx = xvel
    vy = yvel
    loss = cvxpy.norm(vy - cvxpy.multiply(tan_theta_cvx, vx), 2) + gamma*cvxpy.tv(tan_theta_cvx) 
    obj = cvxpy.Minimize( loss )
    prob = cvxpy.Problem(obj) 
    prob.solve(solver='MOSEK')
    theta_cvx = np.arctan(tan_theta_cvx.value)
    #theta_cvx = -1*theta_cvx - np.pi/2.
    
    # atan is accurate to +/- pi, so find the correction factor
    correction = []
    for i in range(len(theta_cvx)):
        if theta_meas[i] - theta_cvx[i] > 2:
            correction.append(1)
        elif theta_meas[i] - theta_cvx[i] < -2:
            correction.append(-1)
        else:
            correction.append(0)
            
    # smooth the correction factor
    correction_cvx = cvxpy.Variable(len(correction))
    loss = cvxpy.norm(correction_cvx - correction, 1) + correction_gamma*cvxpy.tv(correction_cvx)
    obj = cvxpy.Minimize( loss )
    prob = cvxpy.Problem(obj) 
    prob.solve(solver='MOSEK')
    
    theta_cvx_corrected = theta_cvx + correction_cvx.value*np.pi
    
    theta_smooth, thetadot_smooth = diff_angle(theta_cvx_corrected, dt, butter_filter_params,
                                       correction_window_for_2pi=correction_window_for_2pi)
    
    return theta_smooth, thetadot_smooth

def assign_course_and_ang_vel_to_trajec(trajec, 
                                        dt=0.01, 
                                        correction_window_for_2pi=5, 
                                        rough_butter_filter_params=[2,0.5],
                                        smooth_butter_filter_params=[1, 0.1],
                                        gamma = 0.01,
                                        correction_gamma = 0.1,
                                        do_cvx_smoother=False,
                                        ):
    xvel = trajec.xvel.values
    yvel = trajec.yvel.values

    course, course_smoothish, ang_vel_smoothish = get_course_and_raw_angular_velocity(xvel, yvel, 
                                                                                        dt=dt, 
                                                                                        correction_window_for_2pi=correction_window_for_2pi, 
                                                                                        butter_filter_params=rough_butter_filter_params)
    trajec.loc[:,'course'] = course
    trajec.loc[:,'course_smoothish'] = wrap_angle(course_smoothish)
    trajec.loc[:,'ang_vel_smoothish'] = ang_vel_smoothish
    
    if do_cvx_smoother:
        theta_smooth, thetadot_smooth = get_convex_smoothed_course_and_ang_vel(course, xvel, yvel, 
		                                                                    dt=dt, 
		                                                                    correction_window_for_2pi=correction_window_for_2pi, 
		                                                                    butter_filter_params=smooth_butter_filter_params,
		                                                                    gamma = gamma,
		                                                                    correction_gamma = correction_gamma)
        trajec.loc[:,'course_smoother'] = wrap_angle(theta_smooth)
        trajec.loc[:,'ang_vel_smoother'] = thetadot_smooth
    
    
    return trajec

def assign_course_and_ang_vel_to_dataframe(df, 
                                            dt=0.01, 
                                            correction_window_for_2pi=5, 
                                            rough_butter_filter_params=[2,0.5],
                                            smooth_butter_filter_params=[1, 0.1],
                                            do_cvx_smoother=False,
                                            ):
    
    df_obj_vec =[]
    for objid in df['obj_id_unique'].unique():
        trajec = df[df['obj_id_unique']==objid].copy()
        trajec = assign_course_and_ang_vel_to_trajec(trajec, 
                                        dt=dt, 
                                        correction_window_for_2pi=correction_window_for_2pi, 
                                        rough_butter_filter_params=rough_butter_filter_params,
                                        smooth_butter_filter_params=smooth_butter_filter_params,
                                        do_cvx_smoother=do_cvx_smoother,
                                        )
        df_obj_vec.append(trajec)

    fdf = pd.concat(df_obj_vec)

    return fdf

########################################################################################################

def get_continuous_chunks(array, array2=None, jump=1, return_index=False):
    """
    Splits array into a list of continuous chunks. Eg. [1,2,3,4,5,7,8,9] becomes [[1,2,3,4,5], [7,8,9]]
    
    array2  -- optional second array to split in the same way array is split
    jump    -- specifies size of jump in data to create a break point
    """
    diffarray = diffa(array)
    break_points = np.where(np.abs(diffarray) > jump)[0]
    break_points = np.insert(break_points, 0, 0)
    break_points = np.insert(break_points, len(break_points), len(array))
    
    chunks = []
    array2_chunks = []
    index = []
    for i, break_point in enumerate(break_points):
        if break_point >= len(array):
            break
        chunk = array[break_point:break_points[i+1]]
        if type(chunk) is not list:
            chunk = chunk.tolist()
        chunks.append(chunk)
        
        if array2 is not None:
            array2_chunk = array2[break_point:break_points[i+1]]
            if type(array2_chunk) is not list:
                array2_chunk = array2_chunk.tolist()
            array2_chunks.append(array2_chunk)
        
        if return_index:
            indices_for_chunk = np.arange(break_point,break_points[i+1])
            index.append(indices_for_chunk)
            
    if type(break_points) is not list:
        break_points = break_points.tolist()
        
    if return_index:
        return index
    
    if array2 is None:
        return chunks, break_points
    
    else:
        return chunks, array2_chunks, break_points

##########################################################################################################
# Saccade detector (modified Geometric Saccade Detector)

def get_score_amp(args):
    traj, t, delta_frames, time_key, dt = args
    sub_slice = traj[traj[time_key].between(t - delta_frames*dt, 
                                       t + delta_frames*dt)].copy()
    ref_ind = np.where(sub_slice[time_key] == t)[0][0]
    ref_x = sub_slice['x'].iloc[ref_ind]
    ref_y = sub_slice['y'].iloc[ref_ind]
    sub_slice['norm_x'] = sub_slice['x'].sub(ref_x)
    sub_slice['norm_y'] = sub_slice['y'].sub(ref_y)
    
    
    sub_slice['alpha_before'] = np.arctan2(sub_slice['norm_y'], sub_slice['norm_x'])
    sub_slice['alpha_after'] = np.arctan2(-1*sub_slice['norm_y'], -1*sub_slice['norm_x'])
    sub_slice['r'] = np.sqrt( sub_slice['norm_y']**2 + sub_slice['norm_x']**2 )
    
    theta_before = median_angle(sub_slice.loc[sub_slice[time_key] < t, 'alpha_before'])
    theta_after = median_angle(sub_slice.loc[sub_slice[time_key] > t, 'alpha_after'])
    amp = np.abs(np.arccos( np.cos(theta_before)*np.cos(theta_after) + np.sin(theta_before)*np.sin(theta_after) ))
    
    disp = sub_slice['r'].sum()
    
    return t, disp, amp

def assign_saccade_info_with_modified_gsd(traj, delta_frames=5, time_key='timestamp', obj_id_key='obj_id'):
    #traj = traj[~traj.ang_vel_smoother.isna()].copy()
    traj[time_key] = traj[time_key].interpolate()
    traj['ang_vel_smoother'] = traj['ang_vel_smoother'].interpolate()

    objid = traj[obj_id_key].iloc[0]
    dt = 1/int(np.round(1/np.median(np.diff(traj[time_key]))))
    disps, amps, frames = [], [], []
    unique_times = traj[time_key].unique()
    args =[(traj, t, delta_frames, time_key, dt) for t in unique_times]
  
    with multiprocessing.Pool() as pool:
        out_array =pool.map(get_score_amp, args)
        #disps.append(disp)
       # amps.append(amp)
        #frames.append(q)
    for e in out_array:
        frames.append(e[0])
        disps.append(e[1])
        amps.append(e[2])

    
    traj.loc[:,'saccade_gsd_score'] = np.nan_to_num(np.array(amps)**2*np.array(disps)*np.sign(traj.ang_vel_smoother), 0)
    traj.loc[:,'saccade_gsd_amp'] = np.nan_to_num(amps, 0)
    traj.loc[:,'saccade_gsd_disp'] = np.nan_to_num(disps, 0)
    
    return traj


def get_saccade_indices(trajec, height=0.001, distance=10, width=1, plot=False):

    prominence = height*2
    
    score = np.abs(trajec['saccade_gsd_score'].values)
    peaks = scipy.signal.find_peaks(score, height=height, distance=distance, prominence=prominence, width=width)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        braid_analysis_plots.plot_xy_trajectory_with_color_overlay(trajec, column_for_color='saccade_gsd_score',
                                                                   obj_id_key='obj_id', vmin=-0.01, vmax=0.01, ax=ax)
        plt.gca().set_prop_cycle(None)
        for ix in peaks[0]:
            ax.plot(trajec.iloc[ix]['x'], trajec.iloc[ix]['y'], 'x')

    return peaks[0]

###############################################################################################################################
