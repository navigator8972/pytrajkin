import sys, os
from collections import defaultdict
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import pytrajkin_randemb as pytkre
import utils

def get_dataset_images(data, char, output_folder='figs/dataset', animation=True):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    #prepare the output folder
    output_path = os.path.join(curr_dir, output_folder)
    output_path_char = os.path.join(output_path, char[0])
    if not os.path.exists(output_path_char):
        os.makedirs(output_path_char)

    #make a figure without frame
    fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    if not animation:
        for sample_idx, d in enumerate(data[char]):
            #for each letter
            ax = image_for_char(ax, d)
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            #prepare file name
            fname = '{0}_human_sample_{1:03d}.png'.format(char[0], sample_idx)
            fpath = os.path.join(output_path_char, fname)
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            fig.savefig(fpath, bbox_inches=extent, dpi=100)
    else:
        for sample_idx, d in enumerate(data[char]):
            print 'Processing {0}-th sample of letter {1}'.format(sample_idx, char)
            anim = animation_for_char(fig, ax, np.array(d))
            #prepare file name
            fname = '{0}_human_sample_{1:03d}_animated.gif'.format(char[0], sample_idx)
            fpath = os.path.join(output_path_char, fname)
 
            anim.save(fpath, writer='imagemagick', fps=30);
    return


def get_synthetic_data_images(char, mdl_folder='char_mdls', n_samples=1, output_folder='figs/synthetic_samples', animation=True):

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    mdl_dir = os.path.join(curr_dir, mdl_folder)

    data_files = [ f for f in os.listdir(mdl_dir) 
        if os.path.isfile(os.path.join(mdl_dir, f)) and os.path.join(mdl_dir, f).endswith(".p") ]

    data_files_sorted  = sorted(data_files)

    trajkin_mdl = pytkre.TrajKinMdl(use_kin=True)

    success = False

    for f in data_files_sorted:
        #find the file to load the model
        if f[0] == char[0]:
            if trajkin_mdl.load(os.path.join(mdl_dir, f)):
                print 'Successfully loaded model from file {0}'.format(f)
                success = True
                break
            else:
                print 'Failed to load model from file {0}'.format(f)

    if not success:
        print 'No model is successfully loaded'
        return

    #prepare the output folder
    output_path = os.path.join(curr_dir, output_folder)
    output_path_char = os.path.join(output_path, char[0])
    if not os.path.exists(output_path_char):
        os.makedirs(output_path_char)

    #make a figure without frame
    fig_static = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax_static = plt.Axes(fig_static, [0., 0., 1., 1.])
    ax_static.set_axis_off()
    fig_static.add_axes(ax_static)
    ax_static.set_aspect('equal')
    ax_static.set_xlim([-1.5, 1.5])
    ax_static.set_ylim([-1.5, 1.5])

    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    #generate samples...
    if not animation:
        for sample_idx in range(n_samples):
            sample_data = trajkin_mdl.sample()
            ax_static = image_for_char(ax_static, sample_data)
            extent = ax_static.get_window_extent().transformed(fig_static.dpi_scale_trans.inverted())
            #prepare file name
            fname = '{0}_synthetic_sample_{1:03d}.png'.format(char[0], sample_idx)
            fpath = os.path.join(output_path_char, fname)
            ax_static.set_xlim([-1.5, 1.5])
            ax_static.set_ylim([-1.5, 1.5])
            print 'Saving {0}'.format(fpath)
            fig_static.savefig(fpath, bbox_inches=extent, dpi=100)
    else:
        #make a figure without frame
        fig_dyn = plt.figure(frameon=False, figsize=(4,4), dpi=100)
        ax_dyn = plt.Axes(fig_dyn, [0., 0., 1., 1.])
        ax_dyn.set_axis_off()
        fig_dyn.add_axes(ax_dyn)
        ax_dyn.set_aspect('equal')
        ax_dyn.set_xlim([-1.5, 1.5])
        ax_dyn.set_ylim([-1.5, 1.5])

        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

        mean_dict, std_dict = prepare_stroke_timelen_stat(trajkin_mdl)
        for sample_idx in range(n_samples):
            sample_data = trajkin_mdl.sample()

            #prepare static ones...
            ax_static = image_for_char(ax_static, sample_data)
            extent = ax_static.get_window_extent().transformed(fig_static.dpi_scale_trans.inverted())
            #prepare file name
            fname = '{0}_synthetic_sample_{1:03d}.png'.format(char[0], sample_idx)
            fpath = os.path.join(output_path_char, fname)
            ax_static.set_xlim([-1.5, 1.5])
            ax_static.set_ylim([-1.5, 1.5])
            fig_static.savefig(fpath, bbox_inches=extent, dpi=100)

            #also prepare static ones...
            timelen_array = []
            #get averaged time length for each stroke
            for strk_idx, strk_data in enumerate(sample_data):
                timelen_mean = mean_dict[len(sample_data)][strk_idx]
                timelen_std = std_dict[len(sample_data)][strk_idx]
                #sample a time len...
                timelen = np.random.randn() * timelen_std * 0.1 + timelen_mean
                timelen_array.append([timelen])
            sample_data = np.concatenate([sample_data, np.array(timelen_array)], axis=1)
            anim = animation_for_char(fig_dyn, ax_dyn, sample_data, rate=30)

            #prepare file name
            fname = '{0}_synthetic_sample_{1:03d}_animated.gif'.format(char[0], sample_idx)
            fpath = os.path.join(output_path_char, fname)
 
            anim.save(fpath, writer='imagemagick', fps=30);

    # plt.show()

    return

def image_for_char(ax, sample_data):
    if ax is None:
        return None

    ax.hold(False)

    for stroke in sample_data:
        #plot sample data for each stroke
        if len(stroke.shape) > 1:
            #original form...
            ax.plot(stroke[:, 0], -stroke[:, 1], 'b', linewidth=4.0)
        else:
            #flattened form...
            #check length
            if len(stroke) % 2 == 0:
                #no time parameter
                ax.plot(stroke[0:len(stroke)/2], -stroke[len(stroke)/2:len(stroke)], 'b', linewidth=4.0)
            else:
                #need to remove time parameter
                ax.plot(stroke[0:(len(stroke)-1)/2], -stroke[(len(stroke)-1)/2:(len(stroke)-1)], 'b', linewidth=4.0)
        ax.hold(True)
    
    ax.hold(False)

    return ax

def prepare_stroke_timelen_stat(trajmdl):
    #prepare stroke time length statistics: Gaussian...
    mean_dict = defaultdict(list)
    std_dict = defaultdict(list)
    for strk_num, dict_data in trajmdl.data_dict_.iteritems():
        if len(dict_data[0]) != strk_num:
            print 'Unmatched data for stroke number {0}'.format(strk_num)
            continue
        else:
            print '============================================='
            print 'For characters consisted of {0} stroke(s)....'.format(strk_num)
            print '============================================='
            for strk_idx in range(strk_num):
                print 'Processing {0}-th stroke'.format(strk_idx)
                strk_data = np.array([dict_data[i][strk_idx] for i in range(len(dict_data))])
                mean_dict[strk_num].append(np.mean(strk_data[:, -1]))
                std_dict[strk_num].append(np.std(strk_data[:, -1]))
    return mean_dict, std_dict

def animation_for_char(fig, ax, sample_data, rate=30):
    if ax is None:
        return None

    #process the data, rescale time length...
    timelen_array = sample_data[:, -1]
    nframes_array = timelen_array * rate * 3
    motion_data = []
    motionframenum_array = []
    for strk_idx in range(len(nframes_array)):
        strk_data = np.reshape(sample_data[strk_idx][ 0:-1], (2, -1)).T
        # print len(strk_data), int(nframes_array[strk_idx])
        resample_strk_data = utils.interp_traj(strk_data, int(nframes_array[strk_idx]))
        motion_data.append(resample_strk_data)
        motionframenum_array.append(len(resample_strk_data))
    motionframenum_array = np.cumsum(motionframenum_array)
    motion_data_stacked = np.vstack(motion_data)
    # print sample_data
    # print timelen_array
    # print motion_data

    tol_nframes = motionframenum_array[-1]
    frame_data_array = []
    last_frame_data = [[]]
    for frame_idx in range(tol_nframes):
        new_start_pnt = np.any(motionframenum_array==frame_idx)
        if new_start_pnt:
            last_frame_data.append([])
        #insert to the active stroke from the last frame data
        last_frame_data[len(last_frame_data)-1].append(motion_data_stacked[frame_idx, :])
        frame_data_array.append(copy.deepcopy(last_frame_data))

    def init_animation():
        global line
        for strk_data in frame_data_array[0]:
            strk_data_array = np.array(strk_data)
            line, = ax.plot(strk_data_array[:, 0], -strk_data_array[:, 1], 'b', linewidth=4.0)
            ax.hold(True)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

    def animate(nframe):
        for strk_data in frame_data_array[nframe]:
            strk_data_array = np.array(strk_data)
            line, = ax.plot(strk_data_array[:, 0], -strk_data_array[:, 1], 'b', linewidth=4.0)
            ax.hold(True)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.hold(False)

    anim = animation.FuncAnimation(fig, animate, init_func=init_animation, frames=tol_nframes)
    
    return anim