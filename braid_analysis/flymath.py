import numpy as np
import pynumdiff
import cvxpy
import pandas as pd

########################################################################################################
## Course direction and angular velocity functions

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

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
                                        ):
    xvel = trajec.xvel.values
    yvel = trajec.yvel.values

    course, course_smoothish, ang_vel_smoothish = get_course_and_raw_angular_velocity(xvel, yvel, 
                                                                                        dt=dt, 
                                                                                        correction_window_for_2pi=correction_window_for_2pi, 
                                                                                        butter_filter_params=rough_butter_filter_params)

    theta_smooth, thetadot_smooth = get_convex_smoothed_course_and_ang_vel(course, xvel, yvel, 
                                                                            dt=dt, 
                                                                            correction_window_for_2pi=correction_window_for_2pi, 
                                                                            butter_filter_params=smooth_butter_filter_params,
                                                                            gamma = gamma,
                                                                            correction_gamma = correction_gamma)
    
    trajec.loc[:,'course'] = course
    trajec.loc[:,'course_smoothish'] = wrap_angle(course_smoothish)
    trajec.loc[:,'ang_vel_smoothish'] = ang_vel_smoothish

    trajec.loc[:,'course_smoother'] = wrap_angle(theta_smooth)
    trajec.loc[:,'ang_vel_smoother'] = thetadot_smooth

    return trajec

def assign_course_and_ang_vel_to_dataframe(df, 
                                            dt=0.01, 
                                            correction_window_for_2pi=5, 
                                            rough_butter_filter_params=[2,0.5],
                                            smooth_butter_filter_params=[1, 0.1],
                                            ):
    
    df_obj_vec =[]
    for objid in df['obj_id_unique'].unique():
        trajec = df[df['obj_id_unique']==objid].copy()
        trajec = assign_course_and_ang_vel_to_trajec(trajec, 
                                        dt=dt, 
                                        correction_window_for_2pi=correction_window_for_2pi, 
                                        rough_butter_filter_params=rough_butter_filter_params,
                                        smooth_butter_filter_params=smooth_butter_filter_params,
                                        )
        df_obj_vec.append(trajec)

    fdf = pd.concat(df_obj_vec)

    return fdf

########################################################################################################
