
"""
Module for utilities
"""

import numpy as np
import copy
from collections import defaultdict
import matplotlib.pyplot as plt 

import scipy
from scipy import interpolate
import scipy.ndimage as spi
from scipy.stats import itemfreq

from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn import decomposition

uji_data_set_file = 'data/ujipenchars2.txt'

def generate_data_files():
    uji_data = parse_data_set()
    uji_data_normed = normalize_data(uji_data)
    uji_data_interp = interp_data(uji_data)
    #smooth data
    uji_data_smoothed = smoothed_data(uji_data_normed)
    uji_data_interp_num_100 = interp_data_fixed_num(uji_data_smoothed)

    return uji_data_interp_num_100

def parse_data_set():
    """
    parse data set to read letters 
    """
    uji_data = dict()
    f = open(uji_data_set_file, 'r')
    
    #extract coordinates
    start_idx = 3       #the first 3 rows are comments
    for i in range(start_idx):
        dumps = f.readline()
    #for each speciman
    while 1:
        line = f.readline()
        if not line:
            break
        #check if this is the start of another session
        title = line.split()
        if title[0] == '//' and len(title) == 1:
            #print 'start a new session'
            #read another three lines
            for j in range(2):
                dumps = f.readline()
        else:
            #ignore current one, as it is a comment
            for j in range(2):
                #two duplicates for each subject && letter
                line = f.readline()
                input_text = line.split(' ')
                #check if this is a letter among 'a' ~ 'z' or 'A' ~ 'Z'
                # letter_code = ord(input_text[1])
                # valid_letter = False
                # if letter_code <= ord('z') and letter code >= ord('a'):
                #   valid_letter = True
                # elif letter_code <= ord('Z') and letter_code >= ord('A'):
                #   valid_letter = True
                # else:
                #   valid_letter = False
                #check if this is a new record
                letter = input_text[1]
                if letter not in uji_data:
                    uji_data[letter] = []

                #read this letter, we should do this for
                #line indicating strokes
                line = f.readline()
                input_text = line.split()
                num_strokes = int(input_text[1])
                letter_coords = []
                for k in range(num_strokes):
                    line = f.readline()
                    input_text = line.split()
                    #the 2nd indicates number of data points
                    num_pnts = int(input_text[1])
                    #prepare
                    stroke_traj = np.zeros([num_pnts, 2])
                    for l in range(num_pnts):
                        stroke_traj[l, 0] = int(input_text[3 + l * 2])
                        stroke_traj[l, 1] = int(input_text[3 + l * 2 + 1])
                    letter_coords.append(stroke_traj)
                uji_data[letter].append(letter_coords)

    f.close()
    return uji_data

def normalize_data(data_set):
    """
    normalize data:
    1. center all characters 
    2. scale the size
    """
    normed_data = copy.deepcopy(data_set)
    for key in normed_data:
        for l in normed_data[key]:
            #merge strokes to find center & size
            tmp_traj = np.concatenate(l, axis=0)
            # print tmp_traj
            traj_center = np.mean(tmp_traj, axis=0)
            tmp_traj -= traj_center
            traj_max = np.amax(tmp_traj, axis=0)
            scale = np.amax(traj_max)

            for s in l:
                #for each stroke
                s -= traj_center
                s /= scale
    return normed_data

def interp_data(data_set):
    """
    interpolate data
    """
    interp_data = dict()
    for key in data_set:
        interp_data[key] = []
        for l in data_set[key]:
            interp_letter = []
            for s in l:
                time_len = s.shape[0]
                if time_len > 3:
                    #interpolate each dim, cubic
                    t = np.linspace(0, 1, time_len)
                    spl_x = interpolate.splrep(t, s[:, 0])
                    spl_y = interpolate.splrep(t, s[:, 1])
                    #resample, 4 times more, vel is also scaled...
                    t_spl = np.linspace(0, 1, 4 * len(t))
                    x_interp = interpolate.splev(t_spl, spl_x, der=0)
                    y_interp = interpolate.splev(t_spl, spl_y, der=0)
                    # #construct new stroke
                    interp_letter.append(np.concatenate([[x_interp], [y_interp]], axis=0).transpose())
                else:
                    #direct copy if no sufficient number of points
                    interp_letter.append(s)
            interp_data[key].append(interp_letter)
    return interp_data

def interp_data_fixed_num(data_set, num=100):
    """
    interpolate data with fixed number of points
    """
    interp_data = dict()
    for key in data_set:
        interp_data[key] = []
        for l in data_set[key]:
            interp_letter = []
            for s in l:
                time_len = s.shape[0]
                if time_len > 3:
                    #interpolate each dim, cubic
                    t = np.linspace(0, 1, time_len)
                    spl_x = interpolate.splrep(t, s[:, 0])
                    spl_y = interpolate.splrep(t, s[:, 1])
                    #resample, 4 times more, vel is also scaled...
                    t_spl = np.linspace(0, 1, num)
                    x_interp = interpolate.splev(t_spl, spl_x, der=0)
                    y_interp = interpolate.splev(t_spl, spl_y, der=0)
                    # #construct new stroke
                    data = np.concatenate([x_interp, y_interp])
                    dt = float(time_len)/num
                    interp_letter.append(np.concatenate([data, [dt]]))
                else:
                    #direct copy if no sufficient number of points
                    interp_letter.append(s)
            interp_data[key].append(interp_letter)
    return interp_data


def smooth_data(data_set):
    """
    smooth data
    """
    smoothed_data = dict()
    for key in data_set:
        smoothed_data[key] = []
        for l in data_set[key]:
            smoothed_letter = []
            for s in l:
                time_len = s.shape[0]
                if time_len > 3:
                    #smooth the data, gaussian filter
                    filtered_stroke = np.array([spi.gaussian_filter(dim, 3) for dim in s.transpose()]).transpose()
                    smoothed_letter.append(filtered_stroke)
                else:
                    #direct copy if no sufficient number of points
                    smoothed_letter.append(s)
            smoothed_data[key].append(smoothed_letter)
    return smoothed_data

def normalize_trajs_helper(trajs):
    #merge strokes to find center & size
    tmp_traj = np.concatenate(trajs, axis=0)
    # print tmp_traj
    traj_center = np.mean(tmp_traj, axis=0)
    tmp_traj -= traj_center
    traj_max = np.amax(tmp_traj, axis=0)
    scale = np.amax(traj_max)

    normalized_trajs = [(s-traj_center)/scale for s in trajs]

    return normalized_trajs

def smooth_and_interp_trajs_helper(traj, filter_width=3, num=100):
    if len(traj) < filter_width:
        #too short, send a warning and return the original one...
        print 'The stroke is with less than ', filter_width, ' sample points, retain the original one'
        return traj
    else:
        smoothed_stroke = np.array([spi.gaussian_filter(dim, filter_width) for dim in np.array(traj).transpose()]).transpose()
        #interpolation
        t = np.linspace(0, 1, len(smoothed_stroke))
        spl_x = interpolate.splrep(t, smoothed_stroke[:, 0])
        spl_y = interpolate.splrep(t, smoothed_stroke[:, 1])
        #resample, 4 times more, vel is also scaled...
        t_spl = np.linspace(0, 1, num)
        x_interp = interpolate.splev(t_spl, spl_x, der=0)
        y_interp = interpolate.splev(t_spl, spl_y, der=0)
        # #construct new stroke
        processed_traj = np.vstack((x_interp, y_interp)).transpose()
    
    return processed_traj

def smooth_and_interp_trajs(trajs, filter_width=3, num=100):
    """
    a helper to smooth and interp with specified number of samples given a letter instance
    which is possibly consisted of multiple stroke trajectories
    """
    normalized_trajs = normalize_trajs_helper(trajs)
    processed_trajs = [smooth_and_interp_trajs_helper(stroke, filter_width=filter_width, num=num) for stroke in normalized_trajs]
    
    return processed_trajs

def interp_traj(traj, num=100):
    """
    traj is a 2D trajectory, tray to interpolate it and resample with given number of sample points...
    """
    t = np.linspace(0, 1, len(traj))
    spl_x = interpolate.splrep(t, traj[:, 0])
    spl_y = interpolate.splrep(t, traj[:, 1])
    #resample, 4 times more, vel is also scaled...
    t_spl = np.linspace(0, 1, num)
    x_interp = interpolate.splev(t_spl, spl_x, der=0)
    y_interp = interpolate.splev(t_spl, spl_y, der=0)
    # #construct new stroke
    processed_traj = np.vstack((x_interp, y_interp)).transpose()
    return processed_traj
    
def display_data(letters):
    """
    display data: the input will be a list of coordinates, could be multiple strokes...
    """
    #prepare figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()       #for adding multiple letters
    for l in letters:
        for s in l:
            if len(s.shape) == 2:
                #original format
                ax.plot(s[:, 0], -s[:, 1], linewidth=3.5)
            else:
                #compact format, extract data
                if len(s) % 2 == 1:
                    x_data = s[0:(len(s)-1)/2]
                    y_data = s[(len(s)-1)/2:(len(s)-1)]
                    ax.plot(x_data, -y_data, linewidth=3.5)
                else:
                    x_data = s[0:len(s)/2]
                    y_data = s[len(s)/2:]
                    ax.plot(x_data, -y_data, linewidth=3.5)
    ax.set_xlabel('X Coordinate (Unit)')
    ax.set_ylabel('Y Coordinate (Unit)')
    ax.set_aspect('equal')
    
    plt.draw()
    return ax

def plot_trajs_helper(ax, trajs, linespec='-.g'):
    tmp_lines = []
    for traj in trajs:
        tmp_line, = ax.plot(traj[:, 0], traj[:, 1], linespec, linewidth=3.0)
        tmp_lines.append(tmp_line)

    return tmp_lines

def display_vel_profile(letters):
    """
    display the velocity profile for each stroke 
    input must be a set of letters consisted of strokes
    """
    #prepare figure
    max_num_strokes = max([len(l) for l in letters])
    fig = plt.figure()
    axes = []
    plt.ion()
    for i in range(max_num_strokes):
        axes.append(fig.add_subplot(max_num_strokes, 1, i))

    for l in letters:
        for i in range(len(l)):
            #vel profile
            vel_prf = get_vel_profile(l[i]) * len(l[i]) / 100
            idx = np.linspace(0, 1, len(vel_prf))
            axes[i].plot(idx, vel_prf)
    plt.ioff()
    plt.show()
    return

def get_vel_profile(stroke):
    """
    get the velocity profile for a stroke
    input is an array of position coordinates
    """
    vel = np.diff(stroke, axis=0)
    vel_prf = np.sum(vel**2, axis=-1)**(1./2)
    return vel_prf

def get_continuous_ang(stroke):
    """
    get continous angle profile
    see Continous-Angle-Time-Curve
    """
    vel_vec = np.diff(stroke, axis=0)
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

def random_pick(choices, probs):
    """
    a subroutine to select a categorization from choices, given probs of each element
    """
    '''
    >>> a = ['Hit', 'Out']
    >>> b = [.3, .7]
    >>> random_pick(a,b)
    '''
    cutoffs = np.cumsum(probs)
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))
    return choices[idx]

def generate_item_freq(item_lst):
    """
    a subroutine to generate frequency of each items given the list of item data
    """
    model = itemfreq(item_lst)
    #separate them
    items = model[:, 0].astype(int)  #the input is assumed to be integer type array
    freq = model[:, 1]

    return items, freq

def do_pca(data):
    """
    subroutine to do pca:
    data is a 2D array with each row as data instance
    scaling and centering will be done here
    """

    valid_data = np.array(data)
    valid_data = valid_data[~np.isnan(valid_data).any(axis=1)]
    valid_data = valid_data[~np.isinf(valid_data).any(axis=1)]
    mean_data = np.mean(valid_data, axis=0)
    col_std_data = np.array([np.std(d) for d in valid_data.transpose()])
    if len(valid_data) > 1:
        new_data = np.array([(d - mean_data)/col_std_data for d in valid_data])
        corr_mat = new_data.transpose().dot(new_data)
        W, V = np.linalg.eig(corr_mat)
    else:
        #only one data instance
        V = np.zeros((len(col_std_data), len(col_std_data)))
        W = np.zeros(len(col_std_data))

    return W, V, mean_data, col_std_data

def random_forest_embedding(data, n_estimators=400, random_state=0, max_depth=5, min_samples_leaf=1):
    """
    learn a density with random forest representation
    """
    """
    scikit-learn only supports axis-align sepration, let's first stick to this and see how it works
    """
    # n_estimators = 400
    # random_state = 0
    # max_depth = 5
    rf_mdl = RandomTreesEmbedding(
        n_estimators=n_estimators, 
        random_state=random_state, 
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf)
    rf_mdl.fit(data)
    
    # forestClf.fit(trainingData, trainingLabels)
    # indices = forestClf.apply(trainingData)
    # samples_by_node = defaultdict(list)
    # for est_ind, est_data in enumerate(indices.T):
    # for sample_ind, leaf in enumerate(est_data):
    # samples_by_node[ est_ind, leaf ].append(sample_ind)
    # indexOfSamples = samples_by_node[0,10]
    # # samples_by_node[treeIndex, leafIndex within that tree]
    # leafNodeSamples = trainingAngles[indexOfSamples]
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(leafNodeSamples)

    indices = rf_mdl.apply(data)
    samples_by_node = defaultdict(list)
    idx_by_node = defaultdict(list)
    kde_by_node = defaultdict(KernelDensity)

    for idx, sample, est_data in zip(range(len(data)), data, indices):
        for est_ind, leaf in enumerate(est_data):
            samples_by_node[ est_ind, leaf ].append(sample)
            idx_by_node[ est_ind, leaf ].append(idx)

        
    #Kernel Density Estimation for each leaf node
    # for k,v in samples_by_node.iteritems():
    #     est_ind, leaf = k
          # params = {'bandwidth': np.logspace(-1, 1, 20)}
          # grid = GridSearchCV(KernelDensity(), params)
          # grid.fit(v)

    #     kde_by_node[ est_ind, leaf ] = grid.best_estimator_

    res_mdl = dict()
    res_mdl['rf_mdl'] = rf_mdl
    res_mdl['samples_dict'] = samples_by_node
    res_mdl['idx_dict'] = idx_by_node
    # res_mdl['kde_dict'] = kde_by_node
    return res_mdl

def sample_from_rf_mdl(mdl, n_samples=1):

    n_trees = len(mdl['rf_mdl'].estimators_)

    def samle_from_rf_mdl_helper():
        tree_idx = int(np.random.uniform(low=0, high=n_trees))
        tmp_tree = mdl['rf_mdl'].estimators_[tree_idx].tree_
        #print 'sampling from the {0}-th tree'.format(tree_idx)
        def recurse(tree, curr_node_idx):
            #check left and right children
            left_child_node_idx = tree.children_left[curr_node_idx]
            right_child_node_idx = tree.children_right[curr_node_idx]

            if left_child_node_idx == -1 and right_child_node_idx == -1:
                #leaf, return it
                return curr_node_idx
            elif left_child_node_idx == -1 and right_child_node_idx != -1:
                #expand right side
                return recurse(tree, right_child_node_idx)
            elif left_child_node_idx != -1 and right_child_node_idx == -1:
                #expand left side
                return recurse(tree, left_child_node_idx)
            else:
                #make a random decision based number of samples
                n_samples_left = tree.n_node_samples[left_child_node_idx]
                n_samples_right = tree.n_node_samples[right_child_node_idx]
                #binomial
                decision = np.random.binomial(1, n_samples_left/float(n_samples_left+n_samples_right))
                if decision > 0.5:
                    #expand left side
                    return recurse(tree, left_child_node_idx)
                else:
                    #expand right side
                    return recurse(tree, right_child_node_idx)
        
        sample_leaf_idx = recurse(tmp_tree, 0)
        #local gaussian for leaf samples
        samples = mdl['samples_dict'][tree_idx, sample_leaf_idx]
        if len(samples) == 1:
            #local perturbation...
            print 'only one at the leaf node...'
            sample = samples[0]
        else:
            #mean+std
            print 'leaf node contains {0} samples'.format(len(samples))
            sample = np.mean(samples, axis=0)
            #t0 is quite sensitive
            # sample[2:-1:6] = samples[1][2:-1:6]
            # sample[3:-1:6] = samples[1][3:-1:6]
            # # sample[4:-1:6] = samples[1][4:-1:6]
            # # sample[5:-1:6] = samples[1][5:-1:6]
            # sample[6:-1:6] = samples[1][6:-1:6]
            # sample[7:-1:6] = samples[1][7:-1:6]
            # print samples
            # sample = samples[0]

        return sample

    res_samples = np.array([samle_from_rf_mdl_helper() for i in range(n_samples)])
    
    return res_samples

def random_forest_embedding_test(letters):
    samples_by_num_stroke = defaultdict(list)
    for letter in letters:
        samples_by_num_stroke[len(letter)-1].append(letter)
    
    #try letters with only one stroke
    data = np.array(samples_by_num_stroke[0])[:, 0, 0:-1]

    #random embedding
    rf_mdl = random_forest_embedding(data)
    ax = None
    #sample...
    while 1:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            plt.show()
            ax.hold(False)

        sample = sample_from_rf_mdl(rf_mdl)[0]
        ax.plot(sample[0:len(sample)/2], -sample[len(sample)/2:len(sample)])
        plt.draw()
       
        key = raw_input('any key to sample next or \'q\' to exit...')
        if key == 'q':
            break
    plt.ioff()

    return None

def random_forest_embedding_clustering(rf_mdl, template_idx):
    max_idx = 0
    #find largest index
    for k, d in rf_mdl['idx_dict'].iteritems():
        if max_idx < np.max(d):
            max_idx = np.max(d)
    res_vec = np.zeros(max_idx+1)
    #collect clustered data for each leaf
    for k, d in rf_mdl['idx_dict'].iteritems():
        #tree_idx, leaf_idx = k
        if template_idx in d:
            for elem in d:
                res_vec[elem]+=1
    res_vec = res_vec/np.max(res_vec)
    return res_vec

def random_forest_embedding_clustering_test(letters):
    samples_by_num_stroke = defaultdict(list)
    for letter in letters:
        samples_by_num_stroke[len(letter)-1].append(letter)
    
    #try letters with only one stroke
    data = np.array(samples_by_num_stroke[0])[:, 0, 0:-1]
    #<hyin/Feb-18th-2015> velocity seems a bit confusing...
    #vel_data = np.concatenate([np.diff(data[:, 0:data.shape[1]],axis=1), np.diff(data[:, data.shape[1]:], axis=1)], axis=1)
    #random embedding
    #<hyin/Feb-19th-2015> PCA?
    use_pca = True
    train_data = np.array(data)

    if use_pca:
        pca = decomposition.PCA(n_components=10)
        train_data = pca.fit_transform(data)

    rf_mdl = random_forest_embedding(data)

    template_idx = 0
    res_vec = random_forest_embedding_clustering(rf_mdl, template_idx)

    ax = None
    ax_sim = []
    ax_nsim = []

    #sample...
    while 1:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            plt.show()
            ax.hold(True)
        for idx in range(len(res_vec)):
            if res_vec[idx] > 0.05:
                sample = data[idx, :]
                ax.plot(sample[0:len(sample)/2], -sample[len(sample)/2:len(sample)], 'b', alpha=res_vec[idx]**1.05)
        templ_sample = data[template_idx, :]        
        ax.plot(templ_sample[0:len(templ_sample)/2], -templ_sample[len(templ_sample)/2:len(templ_sample)], 'k', linewidth=2.0)
        plt.draw()
        
        n_axes=3
        if 2*n_axes+1 < len(res_vec):
            #there are enough samples to do that
            #draw the most similar and not similar three
            sort_arg_lst = np.argsort(res_vec)
            for i in range(n_axes):
                tmp_fig_sim = plt.figure()
                tmp_fig_nsim = plt.figure()

                sim_sample = data[sort_arg_lst.tolist().index(len(res_vec)-2-i), :]    #the most similar one is it self...
                nsim_sample = data[sort_arg_lst.tolist().index(i), :]

                tmp_ax_sim = tmp_fig_sim.add_subplot(111)
                tmp_ax_nsim = tmp_fig_nsim.add_subplot(111)

                tmp_ax_sim.plot(sim_sample[0:len(sim_sample)/2], -sim_sample[len(sim_sample)/2:len(sim_sample)])
                tmp_ax_nsim.plot(nsim_sample[0:len(nsim_sample)/2], -nsim_sample[len(nsim_sample)/2:len(nsim_sample)])

                plt.draw()

        key = raw_input('any key to sample next or \'q\' to exit...')
        if key == 'q' or True:
            break
    plt.ioff()
    ax.hold(False)

    return res_vec

def registration_points_from_parms(parms):
    '''
    estimate the five registration points from the given parameter component
    '''
    D, t0, mu, sigma, theta_s, theta_e = parms

    a_array = np.array([3*sigma, 1.5*sigma**2+sigma*np.sqrt(0.25*sigma**2+1),
                sigma**2, 1.5*sigma**2-sigma*np.sqrt(0.25*sigma**2+1), -3*sigma])

    t_array = t0 + np.exp(mu)*np.exp(-a_array)
    return t_array


def hausdorff_distance(C1, C2):
    """
    helper function to evaluate the hausdorff distance between two curves
    """
    D = scipy.spatial.distance.cdist(C1, C2, 'euclidean')

    #none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return (H1 + H2) / 2.

def get_mean_letters(data):
    #for a group of letters, get mean letter for each case of stroke numbers
    res = []
    ddict = defaultdict(list)
    for d in data:
        ddict[len(d)].append(np.array(d))
    for n_strk in ddict.keys():
        print 'Getting mean letters with {0} strokes.'.format(n_strk)
        mean_data = None
        for letter in ddict[n_strk]:
            if mean_data is None:
                mean_data = copy.copy(letter)
            else:
                mean_data = mean_data + letter
        mean_data = mean_data / len(ddict[n_strk])
        res.append(mean_data)

    return ddict, res

def difference_of_two_letters(d1, d2):
    #euclidean distance between the two letters
    #they must have same number of strokes...
    if len(d1) != len(d2):
        print 'The number of strokes are not equal!'
        return None
    err = 0.0
    #use the first point as the anchor

    for i_strk in range(len(d1)):
        displacement = d2[i_strk][0] - d1[i_strk][0]
        err+=np.sum(np.sum((d2[i_strk] - d1[i_strk] - displacement)**2))/float(len(d1[i_strk]))
        print 'Accumulated error till stroke {0}:'.format(i_strk+1), err

    return err/float(len(d1))

def from_flatten_to_2D(letters):
    #for a given letter, convert it from the flatten form to 2D
    convt_letters=[]
    for l in letters:
        tmp_letter = []
        for s in l:
            if len(s) % 2 == 1:
                #ignore the fixed time horizon
                tmp_letter.append(np.reshape(s[0:-1], (2, -1)).T)
            else:
                #compact format, extract data
                tmp_letter.append(np.reshape(s, (2, -1)).T)
        convt_letters.append(tmp_letter)
    return convt_letters