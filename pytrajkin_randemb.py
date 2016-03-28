"""
Random embedding of trajectories to learn local models...
"""
import cPickle as cp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt 

import scipy.optimize as sciopt

from sklearn import decomposition
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neighbors import KernelDensity

import utils
import pytrajkin_rxzero as pytkrxz

class TrajKinMdl:

    def __init__(self, data=None, use_pca=False, use_kin=False):
        self.data_          = data
        self.data_dict_     = None
        self.use_pca_       = use_pca
        self.use_kin_       = use_kin
        self.model_         = None
        self.num_strks_     = []
        self.num_strk_occ_  = []

    def populate_data_dict(self):
        """
        populate data dict from model character data
        """
        if self.data_ is None:
            print 'No data-set was found.'
            return
        else:
            """
            the data is a set for a series of characters
            format:
            data_ is a list, each element is a list whose element is a trajectory representation for each stroke
            the trajectory is 1D, but actually represents 2D (x-y) coordinates, it is structured like
            [x0,x1,x2, ..., xn, y0, y1, y2, ..., yn, dt]
            dt is a ratio to indicate the sampling time rate, the actual interval is dt * 0.01
            note, each stroke would have same length of trajectory so the dt might vary, this is different from
            operating on original data
            """
            """
            group data according to number of strokes, this is necessary as a same character might be written with
            different style
            """
            self.data_dict_ = defaultdict(list)
            for d in self.data_:
                #1-based index
                self.data_dict_[len(d)].append(d)
            #distribution of number of strokes
            for k, d in self.data_dict_.iteritems():
                self.num_strks_.append(k)
                self.num_strk_occ_.append(len(d))

            self.num_strk_occ_ = np.array(self.num_strk_occ_, dtype=float)/np.sum(self.num_strk_occ_)
        return

    def train(self, train_kin_var=True):
        #populate data dict if necessary
        if self.data_dict_ is None:
            self.populate_data_dict()

        mdl_dict = defaultdict(list)
        #for each group of data
        for strk_num, dict_data in self.data_dict_.iteritems():
            if len(dict_data[0]) != strk_num:
                print 'Unmatched data for stroke number {0}'.format(strk_num)
                continue
            else:
                print '============================================='
                print 'For characters consisted of {0} stroke(s)....'.format(strk_num)
                print '============================================='
                for strk_idx in range(strk_num):
                    print 'Training {0}-th stroke'.format(strk_idx)
                    strk_data = np.array([dict_data[i][strk_idx] for i in range(len(dict_data))])
                    train_data = strk_data[:, 0:-1]
                    if self.use_pca_:
                        train_data, pca = self.train_pca_preprocessing(strk_data[:, 0:-1])

                    tmp_rf_mdl = self.random_forest_embedding(train_data)  #ignore the last column: dt

                    if self.use_pca_:
                        tmp_rf_mdl['pca'] = pca

                    if self.use_kin_:
                        self.train_leaf_kin_mdl(tmp_rf_mdl)

                    mdl_dict[strk_num].append(tmp_rf_mdl)

        #train kin var if kin is trained
        self.model_ = mdl_dict
        if train_kin_var:
            self.train_leaf_kin_mdl_var()

        return      

    def train_leaf_kin_mdl(self, tmp_rf_mdl):
        #construct kinematics model for each leaf, then the synthesis can be done on the kinematics parm space
        #prepare a model for each leaf node
        kinparms_by_node = defaultdict(list)

        for k, d in tmp_rf_mdl['samples_dict'].iteritems():
            tree_idx, leaf_idx = k
            print 'The leaf contains {0} samples'.format(len(d))
            print 'Training kinematics model for tree {0} and leaf node {1}'.format(tree_idx, leaf_idx)
            if len(d) > 1:
                #extract mean trajectory
                mean_sample = np.mean(d, axis=0)
                mean_traj = np.reshape(mean_sample, (2, -1)).transpose()
                kin_parms = pytkrxz.rxzero_train(mean_traj, global_opt_itrs=1, verbose=False)
                kinparms_by_node[tree_idx, leaf_idx] = [mean_traj[0, :], kin_parms]

        tmp_rf_mdl['kinparms_dict'] = kinparms_by_node
        #to learn the variability of kinematics parameters if possible
        return

    def train_leaf_kin_mdl_single(self, num_strk, strk_idx, tree_idx, leaf_idx):
        """
        A tool method to retrain kinematics parameters for a given node
        For fixing some invalid parameters from the bug of rxzero_train after batch training
        """
        #extract tmp_mdl
        tmp_mdl = None
        if self.model_ is not None:
            if num_strk in self.model_:
                if strk_idx < len(self.model_[num_strk]):
                    tmp_mdl = self.model_[num_strk][strk_idx]

        if tmp_mdl is not None:
            if 'samples_dict' in tmp_mdl and 'kinparms_dict' in tmp_mdl:
                if (tree_idx, leaf_idx) in tmp_mdl['samples_dict']:
                    d = tmp_mdl['samples_dict'][tree_idx, leaf_idx]
                    if len(d) > 1:
                        mean_sample = np.mean(d, axis=0)
                        mean_traj = np.reshape(mean_sample, (2, -1)).transpose()
                        kin_parms = pytkrxz.rxzero_train(mean_traj, verbose=False)
                        tmp_mdl['kinparms_dict'][tree_idx, leaf_idx] = [mean_traj[0, :], kin_parms]
                    else:
                        print 'Only one sample, no need to train.'
                else:
                    print 'No specified tree or leaf index, please check the structure.'
            else:
                print 'No samples or kinematics parameters found... Not trained yet?'
        return


    #<hyin/Feb-22nd-2015> this method is deprecated.
    def kinparm_mdl_var_inference(self, start_pnt, parms, samples):
        """
        P(x^*|\theta, \Sigma_{\theta}) = P(x^*|x_{ref})P(x_{ref}|\theta)P(\theta|\Sigma_{\theta})
        assuming \mu_{\theta} is given
        """
        def obj_func_theta(x, *args):
            """
            objective function to infer the latent parameter
            x:      unknown parameters theta
            args:   a tuple of
            (
                start_pnt
                samples
                mu_theta
                sigma
            )
            """
            start_pnt, samples, mu_theta, sigma = args
            #evaluate theta
            parm = np.reshape(x, (-1, 6))
            t_array = np.arange(0.0, 1.0, 0.01)
            eval_traj, eval_vel = pytkrxz.rxzero_traj_eval(parms, t_array, start_pnt[0], start_pnt[1])
            
            #regularization term
            val = 0.5 * np.linalg.norm(sigma) + 0.5*np.sum((x-mu_theta)*(x-mu_theta)*1./(sigma+1e-5))
            #cost term
            ref_sample = eval_traj.transpose().flatten()
            err = samples - ref_sample      #broadcasting

            val = val + np.sum(np.sum(err * err))

            return val

        def obj_func_sigma(x, *args):
            """
            objective function to infer the parameter variability
            x:      unknown parameters variability
            args:   a tuple of
            (
                mu_theta
                theta
            )
            """
            mu_theta, theta = args

            #regularization term
            val = 0.5 * np.linalg.norm(x)
            #cost term
            val = val + 0.5*np.sum((theta-mu_theta)*(theta-mu_theta)*1./(x+1e-5))

            return val

        def infer_theta(theta_0, start_pnt, mu_theta, sigma, samples):
            #this inference needs optimization
            #construct bounds for theta optimization
            parms = np.reshape(theta_0, (-1, 6))
            bounds_theta = []
            for parm in parms:
                parm_bounds = [ 
                                (0.5*parm[0], 1.5*parm[0]), #D
                                (parm[1] - 0.5*np.abs(parm[1]), parm[1] + 0.5*np.abs(parm[1])), #t0
                                (None, None),               #mu
                                (0.0,  3.0),                 #sigma
                                (-np.pi, np.pi),            #theta_s
                                (-np.pi, np.pi)             #theta_e
                                ]
                bounds_theta = bounds_theta + parm_bounds
            #print 'inferring theta'
            args = (start_pnt, samples, mu_theta, sigma, )
            opt_res = sciopt.minimize(obj_func_theta, theta_0, args=args, bounds=bounds_theta)
            print bounds_theta
            print opt_res.message
            print opt_res.x
            #raw_input()
            return opt_res.x

        def infer_sigma(sigma_0, theta, mu_theta):
            #this should be a convex optimization...

            #construct bounds for sigma optimization
            bounds_sigma = [(0.0, None) for i in range(len(theta))]
            #print 'inferring sigma'
            args = (mu_theta, theta, )
            opt_res = sciopt.minimize(obj_func_sigma, sigma_0, args=args, bounds=bounds_sigma)
            
            return opt_res.x

        mu_theta = parms.flatten()

        theta_0 = parms.flatten()
        sigma_0 = np.ones(len(theta_0))*0.05

        iter_err = 1e4
        iter_tol = 1e-3
        itr = 0
        max_itr = 5

        while iter_err > iter_tol and itr < max_itr:

            #infer theta
            theta_new = infer_theta(theta_0, start_pnt, mu_theta, sigma_0, samples)
            #infer sigma
            sigma_new = infer_sigma(sigma_0, theta_new, mu_theta)

            #update error
            iter_err = np.linalg.norm(theta_new - theta_0)/len(theta_0) + np.linalg.norm(sigma_new-sigma_0)/len(sigma_0)

            #refresh theta_0, sigma_0 and itr count
            theta_0 = theta_new
            sigma_0 = sigma_new
            itr+=1

        return theta_0, sigma_0

    def train_leaf_kin_mdl_var(self):
        if self.model_ is None:
            print 'Need pretrained kinematics models.'
            return
        #train variability for each leaf kinematics model, provide statistics for synthesis of char kin parameters
        for num_strk, mdls in self.model_.iteritems():
            for tmp_rf_mdl in mdls:
                #check if their is field for kinematic parms
                if 'kinparms_dict' in tmp_rf_mdl:
                    kinparms_var_by_node = defaultdict(list)
                    for k, d in tmp_rf_mdl['kinparms_dict'].iteritems():
                        tree_idx, leaf_idx = k
                        print 'Training Var model for tree {0} and leaf node {1}'.format(tree_idx, leaf_idx)
                        start_pnt = d[0]
                        parms = d[1]
                        
                        #leaf samples
                        samples = tmp_rf_mdl['samples_dict'][tree_idx, leaf_idx]
                        #learn local MaxEnt model, concerning the variability of kin parameters
                        #<hyin/Feb-22nd-2015> use a new way to infer the variability of kinematics parameters
                        #kinparm_mdl_var_inference is deprecated
                        #latent_parm, latent_sigma = self.kinparm_mdl_var_inference(start_pnt, parms, samples)

                        #covar in cartesian space, note the correlation between x and y is also taken into account
                        #use block cov for each dimension if one expected independent analysis
                        #extract diagonal covariance matrix
                        #cart_cov = [np.cov(np.array(samples)[:, col_idx]) for col_idx in range(len(samples[0]))]
                        #cart_cov = np.diag(cart_cov)
                        cart_cov = np.cov(np.array(samples).transpose())
                        # print np.diag(cart_cov)
                        #eval trajectory at mean parms
                        t_array = np.arange(0.0, 1.0, 0.01)
                        eval_traj, eval_vel = pytkrxz.rxzero_traj_eval(parms, t_array, start_pnt[0], start_pnt[1])
                        #gradient, use finite difference
                        delta_p = 1e-4
                        grad_mat = []
                        parms_flatten = np.concatenate([start_pnt, parms.flatten()])
                        parms_flatten = np.nan_to_num(parms_flatten)

                        for dim_idx in range(len(parms_flatten)):
                            #evaluate central difference
                            perturb_parm_1 = np.array(parms_flatten)
                            perturb_parm_2 = np.array(parms_flatten)
                            perturb_parm_1[dim_idx] += delta_p * 0.5
                            perturb_parm_2[dim_idx] -= delta_p * 0.5
                            perturb_traj_1, perturb_vel_1 = pytkrxz.rxzero_traj_eval(np.reshape(perturb_parm_1[2:], (-1, 6)), t_array, perturb_parm_1[0], perturb_parm_1[1])
                            perturb_traj_2, perturb_vel_2 = pytkrxz.rxzero_traj_eval(np.reshape(perturb_parm_2[2:], (-1, 6)), t_array, perturb_parm_2[0], perturb_parm_2[1])
                            delta_f = perturb_traj_1.transpose().flatten() - perturb_traj_2.transpose().flatten()

                            grad_mat.append(delta_f/delta_p)

                        grad_mat = np.array(grad_mat)
                        grad_mat = np.nan_to_num(grad_mat)
                        #svd, s is 1-d array
                        U, s, V = np.linalg.svd(grad_mat)
                        #print 'singular values:', s
                        #regularizer for inversion
                        reg_lambda = 5
                        invS = np.zeros((V.shape[1], U.shape[0]))
                        invS[0:len(s), 0:len(s)] = np.diag(1./(s+reg_lambda))

                        grad_mat_reg_inverse = V.dot(invS).dot(U.transpose())


                        cov_parm = grad_mat_reg_inverse.transpose().dot(cart_cov).dot(grad_mat_reg_inverse)
                        # eig_cov, eig_vec = np.linalg.eig(cov_parm)

                        #tmp_rf_mdl['kinlat_dict'] = [ np.reshape(latent_parm, (-1, 6)), latent_sigma ]
                        kinparms_var_by_node[tree_idx, leaf_idx] = np.diag(cov_parm)
                        # print np.diag(cov_parm)
                        # <hyin/Feb-22nd-2015> TODO: need some careful investigation on the scale
                        # print 'variance:', latent_sigma
                    tmp_rf_mdl['kinparm_cov_dict'] = kinparms_var_by_node
        return

    def train_pca_preprocessing(self, data, n_comps=10):
        #pca truncation
        pca = decomposition.PCA(n_components=n_comps)
        prcs_data = pca.fit_transform(data)
        return prcs_data, pca

    def sample_tree_data(self, strk_mdl):
        #sample a tree data for a stroke, which actually belongs to the training data/some local statistics
        sample_data = self.sample_from_rf_mdl(strk_mdl, 1)

        #recover if pca is used
        #note, sample_from_rf_mdl take average in reduced space...
        #<hyin/Feb-19th-2015> this seems not a good idea, average in reduced space give 
        #weird reconstruction. Also, note that, the correlation between strokes are actually
        #not learned, so this sample is independent without considering previous strokes
        #this might result in some issue
        #the first field is data, others are tree_idx, leaf_idx and noise
        if self.use_pca_ and 'pca' in strk_mdl:
            sample_data = strk_mdl['pca'].components_.transpose().dot(sample_data[0][0])
        else:
            #take the first as only one is sampled...
            sample_data = sample_data[0][0]
        return sample_data

    def sample_strk_num(self):
        #sample stroke number
        if not self.num_strk_occ_.size:
            print 'No statistics about number of strokes...'
            return 0
        else:
            sample_array = np.random.multinomial(1, self.num_strk_occ_)

        return sample_array.tolist().index(1)

    def sample(self):
        #sample from the learned mdl, possibly in kinmematics space
        #first decide the number of strokes of the character
        strk_num = self.sample_strk_num()
        sample_data = []
        if strk_num is not None:
            print 'Generate a sample consisted of {0} strokes'.format(strk_num+1)
            #select for the model list
            mdl_lst = self.model_[strk_num+1]
            char_data = []
            for mdl in mdl_lst:
                #for each stroke...
                tmp_sample_data = self.sample_tree_data(mdl)
                sample_data.append(tmp_sample_data)
        return sample_data

    def save(self, fname):
        #save learned model
        #for all fields
        tmp_save_dict = defaultdict(list)
        tmp_save_dict['data']               = self.data_
        tmp_save_dict['data_dict']          = self.data_dict_
        tmp_save_dict['use_pca']            = self.use_pca_
        tmp_save_dict['use_kin']            = self.use_kin_
        tmp_save_dict['model']              = self.model_
        tmp_save_dict['num_strks']          = self.num_strks_
        tmp_save_dict['num_strk_occ']       = self.num_strk_occ_

        cp.dump(tmp_save_dict, open(fname, 'wb'))
        return True

    def load(self, fname):
        tmp_load_dict = cp.load(open(fname, 'rb'))

        success = True

        if tmp_load_dict is not None:
            self.data_          = tmp_load_dict['data']
            self.data_dict_     = tmp_load_dict['data_dict']
            self.use_pca_       = tmp_load_dict['use_pca']
            self.use_kin_       = tmp_load_dict['use_kin']
            self.model_         = tmp_load_dict['model']
            self.num_strks_     = tmp_load_dict['num_strks']
            self.num_strk_occ_  = tmp_load_dict['num_strk_occ']
        else:
            success = False

        return success

    def random_forest_embedding(self, data, n_estimators=50, random_state=0, max_depth=3, min_samples_leaf=1):
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
        
        indices = rf_mdl.apply(data)
        samples_by_node = defaultdict(list)
        idx_by_node = defaultdict(list)
        #kde_by_node = defaultdict(KernelDensity)

        for idx, sample, est_data in zip(range(len(data)), data, indices):
            for est_ind, leaf in enumerate(est_data):
                samples_by_node[ est_ind, leaf ].append(sample)
                idx_by_node[ est_ind, leaf ].append(idx)
          
        res_mdl = dict()
        res_mdl['rf_mdl'] = rf_mdl
        res_mdl['samples_dict'] = samples_by_node
        res_mdl['idx_dict'] = idx_by_node
        # res_mdl['kde_dict'] = kde_by_node
        return res_mdl

    def sample_from_rf_mdl(self, mdl, n_samples=1):

        if not self.use_kin_:
            #no kinematics, sample from raw parameters
            sample_from_kin = False
        else:
            if 'kinparms_dict' in mdl.keys():
                sample_from_kin = True
            else:
                sample_from_kin = False

        n_trees = len(mdl['rf_mdl'].estimators_)

        def samle_from_rf_mdl_helper(from_kin=False):
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
                perturb_parm = 0
                mean_parm = 1
            else:
                #mean+std
                print 'In the {0}-th tree, leaf node {1} contains {2} samples'.format(tree_idx, sample_leaf_idx, len(samples))
                if from_kin:
                    if not mdl['kinparms_dict'][tree_idx, sample_leaf_idx]:
                        sample = np.mean(samples, axis=0)
                    else:
                        #evaluate kinematics model
                        start_pnt = mdl['kinparms_dict'][tree_idx, sample_leaf_idx][0]
                        parms = mdl['kinparms_dict'][tree_idx, sample_leaf_idx][1]
                        mean_parm = np.concatenate([start_pnt, parms.flatten()])
                        print mean_parm
                        #apply a noise if necessary
                        if 'kinparm_cov_dict' in mdl:
                            noise_scale = 0.02
                            # print len(mean_parm), len(mdl['kinparm_cov_dict'][tree_idx, sample_leaf_idx])
                            # print 'covar:', mdl['kinparm_cov_dict'][tree_idx, sample_leaf_idx]
                            perturb_parm = np.random.multivariate_normal(mean_parm,noise_scale * np.diag(mdl['kinparm_cov_dict'][tree_idx, sample_leaf_idx]))
                            #apply perturbed parm
                            applied_start_pnt = perturb_parm[0:2]
                            # applied_start_pnt = start_pnt
                            applied_parms = np.reshape(perturb_parm[2:], (-1, 6))
                            print 'Apply a noise:'
                            print perturb_parm - mean_parm
                            print 'to the parameters.'
                        else:
                            applied_start_pnt = start_pnt
                            applied_parms = parms
                            perturb_parm = mean_parm


                        t_array = np.arange(0.0, 1.0, 0.01)
                        eval_traj, eval_vel = pytkrxz.rxzero_traj_eval(applied_parms, t_array, 
                            applied_start_pnt[0], applied_start_pnt[1])
                        sample = eval_traj.transpose().flatten()
                else:
                    #use mean...
                    sample = np.mean(samples, axis=0)
                    #no parm
                    perturb_parm = 0
                    mean_parm = 1

            return sample, tree_idx, sample_leaf_idx, perturb_parm - mean_parm

        res_samples = np.array([samle_from_rf_mdl_helper(sample_from_kin) for i in range(n_samples)])
        
        return res_samples

def trajkinsyn_test(data, fname=None):
    trajkin_mdl = TrajKinMdl(data, use_pca=False, use_kin=True)
    if fname is None:
        print 'training...'
        trajkin_mdl.train(train_kin_var=False)
        print 'Saving the model'
        trajkin_mdl.save('e_mdl.p')
        print 'train kinvar...'
        trajkin_mdl.train_leaf_kin_mdl_var()
        print 'Saving the model'
        trajkin_mdl.save('e_mdl.p')
    else:
        print 'Loading the model'
        trajkin_mdl.load(fname)
        
    ax = None
    #sample...
    while 1:
        if ax is None:   
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            plt.show()
            ax.hold(False)
    
        sample_data = trajkin_mdl.sample()

        for sample in sample_data:
            #plot sample data for each stroke
            ax.plot(sample[0:len(sample)/2], -sample[len(sample)/2:len(sample)], 'b', linewidth=4.0)
            ax.set_xlabel('X Coordinate', fontsize=16)
            ax.set_ylabel('Y Coordinate', fontsize=16)
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='x', pad=8)
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.hold(True)
            plt.draw()

        ax.hold(False)
        key = raw_input('any key to sample next or \'q\' to exit...')
        if key == 'q':
            break
    plt.ioff()

    return trajkin_mdl

def trajkinsyn_var_test(fname):
    """
    test to learn variability of kin parameters within a leaf node
    """
    #load data
    trajkin_mdl = TrajKinMdl(data=None, use_pca=False, use_kin=True)
    print 'Loading the model'
    trajkin_mdl.load(fname)

    #train a local model to infer latent variables
    #two strokes, the first 
    # rf_mdl = trajkin_mdl.model_[2][0]

    trajkin_mdl.train_leaf_kin_mdl_var()

    ax = None
    #sample...
    while 1:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            plt.show()
            ax.hold(False)
    
        sample_data = trajkin_mdl.sample()
        for sample in sample_data:
            #plot sample data for each stroke
            ax.plot(sample[0:len(sample)/2], -sample[len(sample)/2:len(sample)], 'b', linewidth=4.0)
            ax.hold(True)
            plt.draw()

        ax.hold(False)
        key = raw_input('any key to sample next or \'q\' to exit...')
        if key == 'q':
            break
    plt.ioff()

    return trajkin_mdl

import timeit

def training_time_test(data, number=5):
    time_cost = []
    for i in range(number):
        #create a TrajKinMdl
        test_mdl =  TrajKinMdl(data=data, use_kin=True)
        start_time = timeit.default_timer()
        test_mdl.train()
        elapsed = timeit.default_timer() - start_time
        time_cost.append(elapsed)

    return time_cost

def sampling_time_test(mdl, number=100):
    time_cost = []
    for i in range(number):
        start_time = timeit.default_timer()
        sample_data = mdl.sample()
        elapsed = timeit.default_timer() - start_time
        time_cost.append(elapsed)

    return time_cost