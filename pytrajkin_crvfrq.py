'''
Frequency analysis and reconstruction of curvature features.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft, fftfreq
import scipy.ndimage as spi

import utils

def get_curvature_of_t_indexed_curve(crv):
    '''
    Input:
    crv: a 2D-array representing a curve indexed by t, each row denotes a 2D coordinate

    Output:
    curvature: curvature indexed by t
    '''
    #try another way to get velocity profile
    dx_dt = np.gradient(crv[:, 0])
    dy_dt = np.gradient(crv[:, 1])
    vel = np.vstack((dx_dt, dy_dt)).T

    #line speed
    ds_dt = np.sqrt(dx_dt*dx_dt + dy_dt*dy_dt)

    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    return curvature

def find_monotonic_sections(seq):
    '''
    for a given sequence, find monotonic sections
    '''
    grad = np.gradient(seq)
    #go through gradient to check monoticity
    idx = 0
    ptr = 1
    sections = []
    while ptr < len(grad):
        if grad[ptr] * grad[ptr-1] < 0:
            #change the signal
            sections.append(seq[idx:ptr])
            idx = ptr
        ptr+=1
    if idx != ptr:
        sections.append(seq[idx:ptr])

    return sections

def get_ang_indexed_curvature_of_t_indexed_curve(crv, interp_kind='linear', n_samples_per_2pi=100):
    '''
    Input:
    crv:                    a 2D-array representing a curve indexed by t, each row denotes a 2D coordinate
    n_samples_per_2pi:      how many sample points are used for 2 PI section
    Output:
    curvature: curvature indexed by angular position
    note this would be an array of sections, each of which is for a monotonical part
    '''
    #<hyin/Sep-23rd> it is necessary to have ang_t with the same length as the crv
    #this seems crucial to improve the reconstruction accuracy
    ang_t = get_continuous_ang(crv)
    curvature_t = get_curvature_of_t_indexed_curve(crv)

    ang_sections = find_monotonic_sections(ang_t)

    #visualize for debugging
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hold(True)

    curvature = []
    idx = 0
    for ang_stn in ang_sections:
        #for each monotonical part
        #<hyin/Sep-29th-2015> ignore the tiny part that contains only one point...
        if len(ang_stn) >= 2:
            #fit and resample with equal angular position interval
            #interpolation for this section
            #<hyin/Sep-23rd-2015> Cubic type is much slower than linear though not sure
            #if it was necessary for nontrivial angular position profile
            s = interp1d(x=ang_stn, y=curvature_t[idx:(idx+len(ang_stn))], kind=interp_kind)

            n_samples_for_section = int(np.abs(ang_stn[-1] - ang_stn[0])/(2*np.pi) * n_samples_per_2pi)
            #<hyin/Sep-25th-2015> for tiny or straight stroke where the difference of angular position is small
            #return a fixed amount of sample points - 100
            if n_samples_for_section < 2:
                n_samples_for_section = 100
            ang_sample_pnts = np.linspace(ang_stn[0], ang_stn[-1], n_samples_for_section)
            curvature_section = s(ang_sample_pnts)
            curvature.append(np.copy(curvature_section))

        # ax.plot(ang_stn, curvature_t[idx:idx+len(ang_stn)])
        # ax.plot(ang_sample_pnts, curvature_section)
        # plt.draw()

        idx+=len(ang_stn)

    return curvature, ang_sections

def get_ang_indexed_velocity_of_t_indexed_curve(crv, interp_kind='linear', n_samples_per_2pi=100):
    '''
    Input:
    crv:                    a 2D-array representing a curve indexed by t, each row denotes a 2D coordinate
    n_samples_per_2pi:      how many sample points are used for 2 PI section
    Output:
    velocity: linear velocity indexed by angular position
    note this would be an array of sections, each of which is for a monotonical part
    '''
    #<hyin/Sep-23rd> it is necessary to have ang_t with the same length as the crv
    #this seems crucial to improve the reconstruction accuracy
    ang_t = get_continuous_ang(crv)
    velocity_t = get_vel_profile(crv)

    ang_sections = find_monotonic_sections(ang_t)

    velocity = []
    idx = 0
    for ang_stn in ang_sections:
        #for each monotonical part
        #for each monotonical part
        #<hyin/Sep-29th-2015> ignore the tiny part that contains only one point...
        if len(ang_stn) >= 2:
            #fit and resample with equal angular position interval
            #interpolation for this section
            #<hyin/Sep-23rd-2015> Cubic type is much slower than linear though not sure
            #if it was necessary for nontrivial angular position profile
            s = interp1d(x=ang_stn, y=velocity_t[idx:(idx+len(ang_stn))], kind=interp_kind)

            n_samples_for_section = int(np.abs(ang_stn[-1] - ang_stn[0])/(2*np.pi) * n_samples_per_2pi)
            #<hyin/Sep-25th-2015> for tiny or straight stroke where the difference of angular position is small
            #return a fixed amount of sample points - 100
            if n_samples_for_section < 2:
                n_samples_for_section = 100
            ang_sample_pnts = np.linspace(ang_stn[0], ang_stn[-1], n_samples_for_section)
            vel_section = s(ang_sample_pnts)
            velocity.append(np.copy(vel_section))

        idx+=len(ang_stn)

    return velocity, ang_sections

def get_trajectory_from_ang_curvature_parameterization(ang, curvature, dt=0.01):
    '''
    Return a 2D trajectory given angular position and curvature parameterization
    Input:
    ang:                angular position
    curvature:          curvature array with the same length as ang

    Output:
    crv                 2D curve
    '''
    ang_vel = np.gradient(ang) / dt
    lin_vel = ang_vel / curvature
    vel_x = lin_vel * np.cos(ang)
    vel_y = lin_vel * np.sin(ang)
    vel = np.vstack([vel_x, vel_y]).T
    crv = np.cumsum(vel, axis=0) * dt
    return crv

def get_curvature_frequency_based_curve(freq, amplitude=1.5, ang_s=0, ang_e=16*np.pi, n_samples=800):
    '''
    pure frequency curve, note the frequency is defined with the coordinate of angular position
    '''
    ang = np.linspace(ang_s,ang_e,n_samples)
    curvature = np.exp(amplitude * np.sin(freq*ang))

    res = get_trajectory_from_ang_curvature_parameterization(ang=ang, curvature=curvature, dt=0.005)
    return res

def get_curvature_fft_transform(curvature, n=None):
    '''
    the curvature is indexed by angular position, return the corresponding FFT transformation
    for analysis in frequency domain
    '''
    if n is None:
        return fft(curvature)/len(curvature)
    else:
        return fft(curvature, n)/len(curvature)
def get_curvature_inv_fft_transform(curvature_freq, n=None):
    '''
    return the curvature profile in coordinate domain (angular position) given the representation
    in the frequency domain
    '''
    if n is None:
        n = len(curvature_freq)
    return ifft(curvature_freq*len(curvature_freq), n)

def display_frequency(sp, freq):
    '''
    draw frequency bin chart
    '''
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(freq, np.abs(sp))
    ax.hold(True)
    ax.set_ylabel('Normed Amplitude')
    ax.set_xlabel('Frequency')
    ax.set_xlim([0, 8])
    plt.draw()
       
    return ax

def curvature_freq_test():

    crv = get_curvature_frequency_based_curve(32.0/2.0)

    # get curvature
    curvature, ang_sections = get_ang_indexed_curvature_of_t_indexed_curve(crv, interp_kind='linear')
    ang = np.linspace(0.0, 16*np.pi, len(curvature[0]))

    ###
    ### debug
    ###
    curvature_t = get_curvature_of_t_indexed_curve(crv)
    ang_t = get_continuous_ang(crv)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(np.arange(len(ang_t)), ang_t, '-*')
    # ax.plot(np.arange(len(curvature_t)), curvature_t)
    ax.plot(ang, np.log(curvature[0]))
    ax.hold(True)
    plt.draw()
    ###
    ###
    ###
    crv_reconstruct = get_trajectory_from_ang_curvature_parameterization(ang, curvature[0], dt=0.005)

    ax_original = utils.display_data([[crv]])
    ax_reconstruct = utils.display_data([[crv_reconstruct]])

    #see the frequency analysis and a reconstruction with perturbation in frequency domain
    freq_bins = fftfreq(len(ang), ang[1]-ang[0])*2*np.pi
    log_curvature_freq = get_curvature_fft_transform(np.log(curvature[0]))
    
    ax_freq = display_frequency(sp=log_curvature_freq, freq=freq_bins)

    #corrupt frequency basis coefficients
    log_curvature_freq[2] += np.random.randn() * 0.2
    ax_freq.plot(freq_bins, log_curvature_freq)
    #reconstruct from the corrupted frequency
    corrupt_log_curvature = get_curvature_inv_fft_transform(log_curvature_freq)
    ax.plot(ang, corrupt_log_curvature)

    curvature_recons = np.exp(corrupt_log_curvature)
    crv_corrupt_reconstruct = get_trajectory_from_ang_curvature_parameterization(ang[0:200], curvature_recons[0:200], dt=0.005)
    ax_corrupt_reconstruct = utils.display_data([[crv_corrupt_reconstruct]])

    return

def curvature_and_ang_frequency_for_t_indexed_curve(crv, remove_lowfrq=False):
    if remove_lowfrq:
        time_len = crv.shape[0]
        if time_len > 3:
            #smooth the data, gaussian filter
            smoothed_crv = np.array([spi.gaussian_filter(dim, 3) for dim in crv.transpose()]).transpose()
            work_crv=crv-smoothed_crv
        else:
            print 'Too short stroke, use the original data'
            work_crv=crv
    else:
        work_crv=crv
    curvature = get_curvature_of_t_indexed_curve(work_crv)
    ang = get_continuous_ang(work_crv)
    t_idx = np.arange(len(work_crv))
    freq_bins = fftfreq(len(t_idx), t_idx[1]-t_idx[0])*2*np.pi
    #fft
    crv_frq = get_curvature_fft_transform(curvature)
    ang_frq = get_curvature_fft_transform(ang)
    display_frequency(sp=crv_frq, freq=freq_bins)
    display_frequency(sp=ang_frq, freq=freq_bins)

    return

def get_vel_profile(stroke):
    """
    get the velocity profile for a stroke
    input is an array of position coordinates
    """
    vel_x = np.gradient(stroke[:, 0])
    vel_y = np.gradient(stroke[:, 1])
    vel_prf = np.sqrt(vel_x**2+vel_y**2)
    return vel_prf

def analyze_letter(letter_trajs):
    """
    analyze a given letter...
    """
    res_data = dict()
    res_data['sections'] = []
    res_data['crvfrq'] = []
    res_data['velfrq'] = [] 
    res_data['ang_sections'] = []
    for strk in letter_trajs:
        ang = get_continuous_ang(np.array(strk))
        curvature, ang_sections = get_ang_indexed_curvature_of_t_indexed_curve(np.array(strk))
        velocity, _ = get_ang_indexed_velocity_of_t_indexed_curve(np.array(strk))
        tmp_sctn = []
        tmp_crvt = []
        tmp_velt = []
        tmp_ang_sctn = []
        idx = 0
        for crvt, velt, sctn in zip(curvature, velocity, ang_sections):
            #for each section and corresponding log-curvature profile
            #ignore short tiny sections...
            tmp_sctn+=[strk[idx:(idx+len(sctn))]]
            tmp_crvt+=[get_curvature_fft_transform(np.log(crvt))]
            tmp_velt+=[get_curvature_fft_transform(np.log(velt))]
            tmp_ang_sctn+=[np.linspace(sctn[0], sctn[-1], len(tmp_crvt[-1]))]
            idx+=len(sctn)

        res_data['sections'] += [tmp_sctn[:]]
        res_data['crvfrq'] += [tmp_crvt[:]]
        res_data['velfrq'] += [tmp_velt[:]]
        res_data['ang_sections']+=[tmp_ang_sctn[:]]
    return res_data

def display_analyze_results(data):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig_crvfrq = plt.figure()
    ax_crvfrqplt = fig_crvfrq.add_subplot(111)
    #prepare color
    tol_nr_sctn = np.sum([len(strk) for strk in data['sections']])
    color_lettertraj=iter(cm.rainbow(np.linspace(0,1,tol_nr_sctn)))
    color_crvfrq=iter(cm.rainbow(np.linspace(0,1,tol_nr_sctn)))
    #plot letter sections in canvas
    letter_trajs = []
    for strk in data['sections']:
        for sctn in strk:
            c = next(color_lettertraj)
            tmp_letter_traj, = ax.plot(np.array(sctn)[:, 0], -np.array(sctn)[:, 1], linewidth=2.0, color=c)
            letter_trajs+=[tmp_letter_traj]
            ax.hold(True)
    ax.hold(False)
    #frequency analysis
    for strk_crv, strk_vel, strk_ang in zip(data['crvfrq'], data['velfrq'], data['ang_sections']):
        for crvt_sctn, vel_sctn, ang_sctn in zip(strk_crv, strk_vel, strk_ang):
            freq_bins = fftfreq(n=len(ang_sctn), d=ang_sctn[1]-ang_sctn[0])
            #cut to show only low frequency part
            n_freq = len(ang_sctn)/2+1
            #<hyin/Sep-25th-2015> need more investigation to see the meaning of fftfreq
            #some threads on stackoverflow suggests the frequency should be normalized by the length of array
            #but the result seems weird...                
            #power spectrum coefficient, see Huh et al, Spectrum of power laws for curved hand movements
            freq = np.abs(freq_bins[0:n_freq]*2*np.pi)
            beta = 2.0/3.0 * (1+freq**2/2.0)/(1+freq**2+freq**4/15.0)
            # print 'new section'
            # print freq
            # print beta
            # print np.abs(crvt_sctn[0:n_freq])
            c = next(color_crvfrq)
            ax_crvfrqplt.plot(freq, np.abs(crvt_sctn[0:n_freq])*beta, color=c)
            ax_crvfrqplt.hold(True)
            ax_crvfrqplt.plot(freq, np.abs(vel_sctn[0:n_freq]), color=c, linestyle='dashed')
            ax_crvfrqplt.set_ylabel('Normed Amplitude')
            ax_crvfrqplt.set_xlabel('Frequency')
            #cut the xlim
            ax_crvfrqplt.set_xlim([0, 8])
            ax_crvfrqplt.set_ylim([0.0, 1.0])
    ax_crvfrqplt.hold(False)
    plt.draw()
    return


def get_continuous_ang(stroke):
    """
    get continous angle profile
    see Continous-Angle-Time-Curve
    different from the utilities by using gradient
    """
    vel_x = np.gradient(stroke[:, 0])
    vel_y = np.gradient(stroke[:, 1])
    vel_vec = np.vstack([vel_x, vel_y]).T
    ang = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])
    #compute the change of ang
    ang_diff = np.diff(ang)
    #x and y:
    #x = cos(ang_diff)
    #y = sin(ang_diff)
    x = np.cos(ang_diff)
    y = np.sin(ang_diff)
    thres = 1e-14
    #update delta diff
    for idx in range(len(ang_diff)):
        if x[idx] > thres:
            ang_diff[idx] = np.arctan2(y[idx], x[idx])
        elif x[idx] < -thres and y[idx] > 0:
            ang_diff[idx] = np.arctan2(y[idx], x[idx]) + np.pi
        elif x[idx] < -thres and y[idx] < 0:
            ang_diff[idx] = np.arctan2(y[idx], x[idx]) - np.pi
        elif np.abs(x[idx]) < thres and y[idx] > 0:
            ang_diff[idx] = np.pi/2
        elif np.abs(x[idx]) < thres and y[idx] < 0:
            ang_diff[idx] = -np.pi/2
    cont_ang_prf = ang[0] + np.cumsum(ang_diff)
    return np.concatenate([[ang[0]], cont_ang_prf])

if __name__ == '__main__':
    curvature_freq_test()
