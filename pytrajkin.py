import numpy as np 
from sklearn.mixture import DPGMM, GMM
import pylab
from scipy.special import erf
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import copy

import pytrajkin_rxzero as pyrzx

"""
Module of supporting kinematic trajectory encoding
See Plamondon, A kinematic theory of rapid human movements, Biological Cybernetics, 1995
    Plamondon et al, Recent Developments in the study of rapid human movements
    O'Reilly and Plamondon, Development of a Sigma-Lognormal representation for on-line signatures 
"""

def fit_vel_profile_dpgmm(vel_profile, n_comps=5, dp=False):
    """
    fit a velocity profile with DP-GMM
    """
    N = 1000    # 1000 samples to fit
    integral = np.sum(vel_profile)
    #vel_profile is a 1D array, try to convert it to samples
    t = np.linspace(0, 1, len(vel_profile))
    data = np.array([])
    for i in range(len(t)):
        n_samples = vel_profile[i] / integral * N
        if n_samples > 0:
            #add samples
            samples = np.ones(n_samples) * t[i]
            #add noise
            data = np.concatenate([data, samples])
    fit_data = np.array([data]).transpose()
    #fit Dirichlet-Process Gaussian Mixture Model, 
    #something wrong with the module? The clusters seem merged...
    if dp:
        model = DPGMM(n_components=n_comps, n_iter=1000, alpha=10)
    else:
        model = GMM(n_components=n_comps)
    
    model.fit(fit_data)

    return model

def fit_vel_profile_gmm_test(vel_profile):
    n_comps = 3
    gmm = fit_vel_profile_dpgmm(vel_profile, n_comps)
    dpgmm = fit_vel_profile_dpgmm(vel_profile, n_comps=n_comps, dp=True)
    #show profile and fit from GMM
    t = np.linspace(0, 1, len(vel_profile))
    pylab.plot(t, vel_profile)
    #eval gmm
    logprob, resp = gmm.score_samples(np.array([t]).transpose())
    logprob_dp, resp_dp = dpgmm.score_samples(np.array([t]).transpose())
    pylab.plot(t, np.exp(logprob) / 40)
    pylab.plot(t, np.exp(logprob_dp))

    return

def vel_profile_registration(vel_profile):
    """
    function to generate a sequence of five characteristic points to for stroke vel_profile_registration
    for heuristics to generate initial guess of siglognormal parameters
    """
    #i guess here the inflexion point is specified with respect to velocity profile
    #otherwise the inflexion point of position profile should just be derivative of velocity
    #so crossing zero means the local maximum/minimum of velocity
    #however, this is not for sure, need to check the comprehensive one
    acc_profile = np.diff(np.append([0], vel_profile))
    acc_profile = np.diff(np.append([0], acc_profile))
    #search local maximum
    #the profile should be well smoothed to prevent identify strokes from noisy profile
    width = 5
    t3_array = []
    #sliding over the profile...
    for start_idx in range(len(vel_profile) - width + 1):
        sub_prf = vel_profile[start_idx:(start_idx+width-1)]
        if np.argmax(sub_prf) == (width-1)/2:   #center point is larger than adjacent ones
            t3_array.append(start_idx + (width-1)/2)
    if not t3_array:
        #no local maximum found, invalid profile
        print 'Invalid profile - velocity profile must have at least one local maximum...'
        return None
    #for each t3, search t_1, t_2, t_4, t_5 (See Ref.):
    #t_1 - first point preceding p3_n which is a local minimum or has a magnitude or less than 1% of v3_n
    #t_2 - first inflexion point before t_3 dot{v} = 0
    #t_4 - first inflexion point after t_3 dot{v} = 0
    #t_5 - first point following p3_n which is a local minimum or has a magnitude of less than 1% of v3_n
    t1_array = []
    t2_array = []
    t4_array = []
    t5_array = []

    for t3 in t3_array:
        #from t3 search along the two directions...
        t1_found, t2_found, t4_found, t5_found = False, False, False, False
        #for t1 and t2
        for i in range(t3):
            if t3-i < (width - 1)/2:
                break
            sub_prf = vel_profile[(t3-i-(width-1)/2):(t3-i+(width-1)/2)]
            if not t1_found:
                if np.argmin(sub_prf) == (width-1)/2 or vel_profile[t3-i] < 0.01 * vel_profile[t3]:
                    t1_array.append(t3-i)
                    t1_found = True
            if not t2_found:
                if acc_profile[t3-i+1] * acc_profile[t3-i-1] < 0:
                    t2_array.append(t3-i)
                    t2_found = True
            if t1_found and t2_found:
                break
        if not t1_found:
            #it is unlikely to happen, but for the first stroke, probably we need start of stroke as time index 0
            t1_array.append(0)
        if not t2_found:
            t2_array.append(0)
        #for t4 and t5
        for i in range(len(vel_profile) - t3):
            i_in_array = i+t3+1
            if i_in_array+(width-1)/2 >= len(vel_profile):
                break
            sub_prf = vel_profile[(i_in_array-(width-1)/2):(i_in_array+(width-1)/2)]
            if not t4_found:
                if acc_profile[i_in_array+1] * acc_profile[i_in_array-1] < 0:
                    t4_array.append(i_in_array)
                    t4_found = True
            if not t5_found:
                if np.argmin(sub_prf) == (width-1)/2 or vel_profile[i_in_array] < 0.01 * vel_profile[t3]:
                    t5_array.append(i_in_array)
                    t5_found = True

            if t4_found and t5_found:
                break
        if not t4_found:
            t4_array.append(len(vel_profile)-1)
        if not t5_found:
            t5_array.append(len(vel_profile)-1)
    res = np.array([[t1_array[i], t2_array[i], t3_array[i], t4_array[i], t5_array[i]] for i in range(len(t3_array))])

    return res

def vel_profile_reg_test(vel_profile):
    #first plot this vel profile
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(range(len(vel_profile)), vel_profile)
    plt.ion()
    #ready for get registration points...
    reg_pnts = vel_profile_registration(vel_profile)
    #for each registration points...
    colors = ['r', 'y', 'k', 'g', 'w']
    for reg_pnt_idx in range(5):
        ax.plot(reg_pnts[:, reg_pnt_idx].transpose(), 
            vel_profile[reg_pnts[:, reg_pnt_idx].transpose()],
            colors[reg_pnt_idx]+'o')
    plt.ioff()
    #also plot acc profile
    acc_profile = np.diff(np.append([0], vel_profile))
    acc_profile = np.diff(np.append([0], acc_profile))
    ax_acc = fig.add_subplot(212)
    ax_acc.plot(range(len(acc_profile)), acc_profile)
    plt.show()
    return

class TrajKinMdl:
    """
    class definition for kinematic trajectory model
    Support type of component type:
    siglognormal
    """
    def __init__(self, mdl_type='siglognormal', mdl_parms=None):
        self.mdl_type_ = mdl_type
        self.mdl_parms_ = mdl_parms
        self.x0 = 0.0
        self.y0 = 0.0
        #not sure if the interval should be fixed, let it be now although t_idx is exact time, confusing, uh?
        self.dt = 0.01

    def eval(self, t_idx, parms=None):
        """
        evaluate model with given time indexes
        """
        if parms is None:
            parms = self.mdl_parms_

        res_pos = None
        res_vel = None
        if self.mdl_type_ == 'siglognormal':
            """
            for siglognormal model, each parm should be a tuple: (D, t0, mu, sigma, theta_s, theta_e)
            """
            """
            The following seems not a good implementation, note the division of zero
            The problem is that the need to evaluate the limit of (sin(Phi(t)) - sin(theta_s))/(theta_e - theta_s)
            when theta_s -> theta_e
            so let's try the velocity profile implementation, this seems not that compact though...
            """
            # x_comps = np.array([ comp_parm[0] / (comp_parm[5] - comp_parm[4]) * (np.sin(self.siglog_normal_Phi(comp_parm, t_idx)) - np.sin(comp_parm[4])) for comp_parm in self.mdl_parms_ ])
            # y_comps = np.array([ comp_parm[0] / (comp_parm[5] - comp_parm[4]) * (-np.cos(self.siglog_normal_Phi(comp_parm, t_idx)) + np.cos(comp_parm[4])) for comp_parm in self.mdl_parms_])

            # res = np.concatenate([ [np.sum(x_comps, axis=0)], 
            #   [np.sum(y_comps, axis=0)]], axis=0).transpose()
            # v_amp_array = np.array([self.siglog_vel_amp(parm, t_idx) for parm in parms])
            # phi_array = np.array([self.siglog_normal_Phi(parm, t_idx) for parm in parms])   

            # v_x = np.sum(np.abs(v_amp_array) * np.cos(phi_array), axis=0)
            # v_y = np.sum(np.abs(v_amp_array) * np.sin(phi_array), axis=0)
            # v_vec = np.concatenate([[v_x], [v_y]], axis=0).transpose()

            # #more consideration is needed for this dt...
            # res_pos = np.array([self.x0, self.y0]) + np.cumsum(v_vec, axis=0) * self.dt
            # res_vel = v_vec
            res_pos, res_vel = pyrzx.rxzero_traj_eval(parms, t_idx, self.x0, self.y0)
        else:
            print 'Invalid or unsupported model type'
            
        return res_pos, res_vel

    def train(self, pos_traj):
        #extract parameters from given position trajectory
        #pos_traj:      an array of 2D position coordinates
        #get a series of t idx
        # vel_profile = self.get_vel_profile(pos_traj)
        # reg_pnts = vel_profile_registration(vel_profile)
        #<hyin/Feb-09-2015> use rxzero, though a not complete version...
        parms, reg_pnts = pyrzx.rxzero_train(pos_traj, getregpnts=True)
        self.mdl_parms_ = parms
        return parms, reg_pnts

    def get_vel_profile(self, stroke):
        """
        get the velocity profile for a stroke
        input is an array of position coordinates
        """
        vel = np.diff(stroke, axis=0)
        vel_prf = np.sum(vel**2, axis=-1)**(1./2)
        return vel_prf

    def siglog_normal_init_guess(self, reg_pnts_array, pos_traj):
        """
        get initial guess from registration points for the parameters of siglognormal model
        """

        def make_sigma_equation(reg_pnts):
            """
            input is a series of registration points t1 - t5
            though only t1, t3, t5 are useful
            """
            def sigma_estimator(sigma):
                #do float here, otherwise the second term will be ZERO!!
                return ((np.exp(-sigma**2+3*sigma) - 1) / (np.exp(sigma*6) - 1) - 
                    float(reg_pnts[2] - reg_pnts[0]) / (reg_pnts[4] - reg_pnts[0]))
                #return (np.exp(sigma**2) - 1) / (np.exp(6*sigma**2) - 1) - (reg_pnts[2] - reg_pnts[0]) / (reg_pnts[4] - reg_pnts[0])
            return sigma_estimator

        init_guess = []
        vel_profile=self.get_vel_profile(pos_traj)/self.dt
        for reg_pnts in reg_pnts_array:
            #make an estimator for sigma
            sig_est_func = make_sigma_equation(reg_pnts)
            #solve it
            init_sigma = (reg_pnts[4] - reg_pnts[0])/2*self.dt
            #<hyin/Dec-24-2014> solving the equation is still not clear and the results seem not right
            #more investigation is needed to know the derivation of the equation, e.g., how the units are consistent...
            #currently, use an even more simple heuristics...
            sig_sln = sciopt.broyden1(sig_est_func, init_sigma, f_tol=1e-14)
            #try direct nonlinear optimization, note the returned function is modified with square
            #see sigma_estimator above
            #sig_sln = sciopt.fminbound(sig_est_func, 0, init_sigma*3) #search between (0, 3*init_sigma)
            #print sig_sln, sig_est_func(sig_sln)
            sig = sig_sln   #only need scalar value
            #sig = init_sigma
            if sig <= 0:
                #will this happen?
                #this will happen when actual mode locates on the right side of 'Gaussian mean'
                #lognormal distribution is asymmetrical in regards of mode, but does it always distribute more mass on left side? 
                #(okay, i know it's long tail, more mass means some more slopeness when going up)
                sig = np.abs(sig)
            a_array = np.array([3*sig, 1.5*sig**2+sig*np.sqrt(0.25*sig**2+1),
                sig**2, 1.5*sig**2-sig*np.sqrt(0.25*sig**2+1), -3*sig])
            #estimate mu
            mu_1 = np.log((reg_pnts[2]-reg_pnts[0])*self.dt/(np.exp(-a_array[2])-np.exp(-a_array[0])))
            mu_2 = np.log((reg_pnts[4]-reg_pnts[0])*self.dt/(np.exp(-a_array[4])-np.exp(-a_array[0])))
            mu = (mu_1 + mu_2)/2
            #estimate D
            D_array = np.array([np.sqrt(np.pi*2)*vel_profile[i]*np.exp(mu)*sig
                *np.exp((a_array[i]**2/(2*sig**2)-a_array[i])) for i in range(len(a_array))])
            D = np.average(D_array)
            #estimate t0
            t0_array = np.array([reg_pnts[i]*self.dt - np.exp(mu)*np.exp(-a_array[i]) for i in range(len(a_array))])
            t0 = np.average(t0_array)

            theta_s, theta_e = self.siglog_normal_init_ang_guess((D, t0, mu, sig),
                pos_traj, reg_pnts)
            
            #add
            init_guess.append((D, t0, mu, sig, theta_s, theta_e))
        return init_guess

    def siglog_normal_init_ang_guess(self, vel_parms, pos_traj, reg_pnts=None):
        """
        get initial angular position guess from given vel parameters
        """
        #get phi at each registration points...
        D, t0, mu, sig = vel_parms
        a_array = np.array([3*sig, 1.5*sig**2+sig*np.sqrt(0.25*sig**2+1),
                sig**2, 1.5*sig**2-sig*np.sqrt(0.25*sig**2+1), -3*sig])
        #estimate static parameters, theta_s, theta_e
        #first get l
        l_array = np.array([0, 
            D/2*(1+erf(-a_array[1]/(sig*np.sqrt(2)))),
            D/2*(1+erf(-a_array[2]/(sig*np.sqrt(2)))),
            D/2*(1+erf(-a_array[3]/(sig*np.sqrt(2)))),
            D])
        #estimate reg_pnts if not given
        #here the estimation is not necessarily robust as it depends on t0
        #probably a better idea is to get reg_pnts from geometrical properties
        if reg_pnts is None:
            reg_pnts = ((t0 + np.exp(mu)*np.exp(-a_array)) / self.dt).astype(int)
        reg_pnts[reg_pnts>=len(pos_traj)-2] = len(pos_traj)-2
        phi_array=[]
        for i in range(len(reg_pnts)):
            vel_dir = np.array([0, 0])
            if reg_pnts[i] == 0:
                #first point, using next point to estimate
                for j in range(len(reg_pnts)):
                    vel_dir = pos_traj[reg_pnts[i]+j+1, :] - pos_traj[reg_pnts[i], :]
                    vel_norm = np.linalg.norm(vel_dir)
                    if vel_norm > 1e-10:
                        vel_dir = vel_dir/vel_norm
                        break
            elif reg_pnts[i] == len(pos_traj):
                #last point, using previous point to estimate
                for j in range(len(reg_pnts)):
                    vel_dir = pos_traj[reg_pnts[i], :] - pos_traj[reg_pnts[i]-j-1, :]
                    vel_norm = np.linalg.norm(vel_dir)
                    if vel_norm > 1e-10:
                        vel_dir = vel_dir/vel_norm
                        break
            else:
                #using adjacent points to estimate
                vel_dir = (pos_traj[reg_pnts[i]+1, :] - pos_traj[reg_pnts[i]-1, :])/2
                vel_norm = np.linalg.norm(vel_dir)
                vel_dir = vel_dir/vel_norm
            phi_array.append(np.arctan2(vel_dir[1], vel_dir[0]))
        phi_array = np.array(phi_array)
        delta_phi = (phi_array[3] - phi_array[1])/(l_array[3] - l_array[1])
        theta_s = phi_array[2] - delta_phi*(l_array[2]-l_array[0])
        theta_e = phi_array[2] - delta_phi*(l_array[4]-l_array[2])
        return theta_s, theta_e

    def siglog_normal_train_vel_parms(self, init_guess, vel_profile):
        """
        routine to extract velocity profile related parameters given the initial guess
        """
        init_guess_array = np.array(init_guess).flatten()
        def vel_profile_obj_func(parms):
            """
            a scalar objective function to optimize
            """
            t_array = np.arange(len(vel_profile))*self.dt
            #here parm is a 2d array but in optimization it seems to be flattened, so let's convert it
            parms_array = np.reshape(parms, (-1, 6))
            #error between fit vel profile and numerical ones
            #traj_eval, vel_eval = self.eval(np.arange(len(vel_profile))*self.dt, parms_array)
            #obj = np.linalg.norm(np.sqrt(np.sum(vel_eval**2, axis=1).transpose())[0] - vel_profile)**2
            """
            TODO: need further investigation, this seems weird as the wrong evaluation (though it is an approximation)
            is more robust to optimize compare with optimize them all together
            some conjectures:
            1. in their paper - they choose to make t0 and D dependent on mu, sig, theta_s & theta_e
                                and the four variables need to be optimized;
            2. here - our wrong doing seems to be a routine similar to EM, that optimize factorized variables:
                                first D, t0, mu, sig by evaluating an approximation, then theta_s & theta_e separately
            optimizing all variables simultaneously (uncomment above lines) together seems to be detrimental
            let's see if this way is also robust enough and to what extent does it rely on the constraints...
            """
            wrong_vel_prf = []
            for parm in parms_array:
                wrong_vel_prf.append(self.siglog_vel_amp(parm, t_array))
            wrong_vel_prf_sum = np.sum(np.abs(wrong_vel_prf), axis=0)
            obj = np.linalg.norm(wrong_vel_prf_sum - vel_profile)**2

            return obj
        #prepare bounds, now for mu and sig, this seems to be critical, investigate how they regularize this...
        bounds_array = []
        theta_scale = 0
        for parm in init_guess:
            bounds_array = bounds_array + [(None, None), (None, None), #D, t0
            #here note the bound of sig, it must be larger than zero...
            (parm[2]-parm[3], parm[2]+parm[3]), (0.3*parm[3], 3*parm[3]),       #mu, sig
            (parm[4]-np.pi*theta_scale, parm[4]+np.pi*theta_scale), (parm[5]-np.pi*theta_scale, parm[5]+np.pi*theta_scale)]                                 #theta_s, theta_e
        opt_res = sciopt.minimize(vel_profile_obj_func, init_guess_array, bounds=bounds_array)
        opt_parm = np.reshape(opt_res.x, (-1, 6))
        #print opt_parm
        return tuple(opt_parm)

    def siglog_normal_train_ang_parms(self, init_guess, pos_traj):
        """
        routine to extract angular position related parameters given the initial guess
        """
        #note here only make angular position parms be optimized
        #init_guess_array = np.array(init_guess)[:, 4:].flatten()
        vel_parms = np.array(init_guess)[:, 0:4]
        init_guess_array = []
        #prepare initial guess
        for vel_parm in vel_parms:
            theta_s, theta_e = self.siglog_normal_init_ang_guess(vel_parm, pos_traj)
            init_guess_array.append(theta_s)
            init_guess_array.append(theta_e)

        def pos_profile_obj_func(parm):
            #here parm is an flattened array, restore it
            ang_parms = np.reshape(parm, (-1, 2))
            self.mdl_parms_ = np.concatenate([vel_parms, ang_parms], axis=1)
            traj_eval, vel_eval = self.eval(np.arange(len(pos_traj))*self.dt)
            obj = np.sum(np.sum((traj_eval - pos_traj)**2, axis=1))
            return obj
        #again bounds
        bounds_array = []
        for parm in init_guess:
            bounds_array = bounds_array + [(-np.pi, np.pi), (-np.pi, np.pi)]
        opt_res = sciopt.minimize(pos_profile_obj_func, init_guess_array, bounds=bounds_array)
        opt_parm = np.concatenate([vel_parms, np.reshape(opt_res.x, (-1, 2))], axis=1)
        print opt_res.success, opt_res.nit
        print 'Error of reconstruction: {0}'.format(opt_res.fun)
        return opt_parm

    def siglog_normal_Phi(self, parm, t_idx):
        """
        Phi function for sig-log-normal model
        parm is a tuple of (D, t0, mu, sigma, theta_s, theta_e), though D is not used here
        """
        D, t0, mu, sigma, theta_s, theta_e = parm
        #argument for the erf function
        t_minus_t0 = t_idx - t0
        #truncation to keep time larger than 0
        thres = 1e-6
        t_minus_t0[t_minus_t0 < thres] = thres
        #take log, mu and sigma to get the input argument
        z = (np.log(t_minus_t0) - mu) / sigma
        #evaluate erf function
        res = theta_s + (theta_e - theta_s) / 2 * (1 + erf(z))
        return res

    def siglog_vel_amp(self, parm, t_idx):
        """
        siglognormal velocity amplitude evaluation
        """
        D, t0, mu, sigma, theta_s, theta_e = parm
        #argument for the sig log normal function
        t_minus_t0 = t_idx - t0
        #truncation to keep time larger than 0
        thres = 1e-6
        #t_minus_t0[t_minus_t0 < thres] = thres
        res = np.zeros(len(t_idx))
        res[t_minus_t0 < thres] = 0.0
        res[t_minus_t0 >= thres] = D / (np.sqrt(2 * np.pi)) / t_minus_t0[t_minus_t0 >= thres] * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2))
        return res

def siglognormal_mdl_test():
    #test the sig-log-normal model
    mdl = TrajKinMdl()
    mdl.mdl_parms_ = [(0.5, 0.0, 0.2, 0.2, 11./12*np.pi, 7./4*np.pi),
                         (0.4, 0.2, 0.36, 0.2, 7./4*np.pi, 7./4*np.pi), 
                         (0.5, 0.5, 0.75, 0.2, 7./4*np.pi, 11./12*np.pi)]
    traj, vel = mdl.eval(np.linspace(0, 10, 1000))

    fig = plt.figure()
    ax_traj = fig.add_subplot(211)
    ax_traj.plot(traj[:, 0], traj[:, 1])
    ax_vel = fig.add_subplot(212)
    ax_vel.plot(np.linspace(0, 10, 1000), np.sum(vel**2, axis=-1)**(1./2))
    plt.show()
    return

def siglognormal_mdl_train_test(pos_traj, mdl_parms=None):
    #prepare initial guess of parameters
    if mdl_parms is None:
        plt.ion()
        mdl = TrajKinMdl()
        mdl.x0 = pos_traj[0, 0]
        mdl.y0 = pos_traj[0, 1]
        vel_profile = mdl.get_vel_profile(pos_traj)/0.01
        reg_pnts_array = vel_profile_registration(vel_profile)
        t_array = np.arange(len(vel_profile))*mdl.dt
        fig = plt.figure()
        ax_ori = fig.add_subplot(321)
        ax_ori.plot(pos_traj[:, 0], -pos_traj[:, 1])
        ax_ori.axis('equal')
        
        ax = fig.add_subplot(322)
        ax.plot(t_array, vel_profile)

        #for each registration points...
        colors = ['r', 'y', 'k', 'g', 'w']
        for reg_pnt_idx in range(5):
            ax.plot(reg_pnts_array[:, reg_pnt_idx].transpose()*mdl.dt, 
                vel_profile[reg_pnts_array[:, reg_pnt_idx].transpose()],
                colors[reg_pnt_idx]+'o')

        init_guess = mdl.siglog_normal_init_guess(reg_pnts_array, pos_traj)
        #show initial guess of velocity profile
        mdl.mdl_parms_ = init_guess
        traj_guess, vel_vec_guess = mdl.eval(t_array)
        #print init_guess
        ax.plot(t_array, np.sum(vel_vec_guess**2, axis=1)**(1./2))

        #show optimized result of velocity profile
        vel_prf_opt = []
        vel_prf_parms_opt = mdl.siglog_normal_train_vel_parms(init_guess, vel_profile)
        mdl.mdl_parms_ = vel_prf_parms_opt
        traj_opt, vel_vec_opt = mdl.eval(t_array)
        ax.plot(t_array, np.sum(vel_vec_opt**2, axis=1)**(1./2))

        #show a wrong way to evaluate velocity profile
        # wrong_vel_prf = []
        # for parm in init_guess:
        #   wrong_vel_prf.append(mdl.siglog_vel_amp(parm, t_array))
        # wrong_vel_prf_sum = np.sum(np.abs(wrong_vel_prf), axis=0)
        # ax.plot(t_array, wrong_vel_prf_sum)

        #letter profile
        traj, vel_prf = mdl.eval(t_array, init_guess)

        ax_ori.plot(traj[:, 0], -traj[:, 1])
        #try opt parm
        opt_parm = mdl.siglog_normal_train_ang_parms(vel_prf_parms_opt, pos_traj)
        #print opt_parm
        traj_opt, vel_vec_opt = mdl.eval(t_array, opt_parm)
        ax_ori.plot(traj_opt[:, 0], -traj_opt[:, 1])

        #theta
        ax_theta = fig.add_subplot(323)
        vel_vec = np.diff(pos_traj, axis=0)/mdl.dt
        theta = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])
        ax_theta.plot(np.arange(len(theta))*mdl.dt, theta)

        #init one
        theta_guess = np.arctan2(vel_vec_guess[:, 1], vel_vec_guess[:, 0])
        ax_theta.plot(np.arange(len(theta_guess))*mdl.dt, theta_guess)

        #opt one
        theta_opt = np.arctan2(vel_vec_opt[:, 1], vel_vec_opt[:, 0])
        ax_theta.plot(np.arange(len(theta_opt))*mdl.dt, theta_opt)

        #vel x & y
        ax_vel_x = fig.add_subplot(325)
        ax_vel_y = fig.add_subplot(326)
        #original
        ax_vel_x.plot(t_array, vel_vec[:, 0])
        ax_vel_y.plot(t_array, vel_vec[:, 1])

        #init guess
        ax_vel_x.plot(t_array, vel_vec_guess[:, 0])
        ax_vel_y.plot(t_array, vel_vec_guess[:, 1])

        #opt ones
        ax_vel_x.plot(t_array, vel_vec_opt[:, 0])
        ax_vel_y.plot(t_array, vel_vec_opt[:, 1])


        plt.ioff()
        plt.show()

        return opt_parm
    else:
        #if parms is given then use it directly
        mdl = TrajKinMdl()
        mdl.x0 = pos_traj[0, 0]
        mdl.y0 = pos_traj[0, 1]
        mdl.dt = 0.01
        t_array = np.arange(len(pos_traj)-1)*mdl.dt

        vel_vec = np.diff(pos_traj, axis=0)/mdl.dt

        fig = plt.figure()
        plt.ion()
        ax_prf = fig.add_subplot(221)
        ax_prf.plot(pos_traj[:, 0], -pos_traj[:, 1])

        #perturb parameters
        #seed noise
        r = np.random.RandomState()
        perturbed_parms = copy.copy(mdl_parms)
        perturb_idx = 3 
        for parm in perturbed_parms:
            #perturb indexed parms
            noise_sig = 0.25 * parm[perturb_idx]
            noise = noise_sig * r.randn()
            print noise
            parm[perturb_idx] = parm[perturb_idx] + noise
        traj_opt, vel_vec_opt = mdl.eval(t_array, perturbed_parms)
        ax_prf.plot(traj_opt[:, 0], -traj_opt[:, 1])

        #velocity
        ax_vel_x = fig.add_subplot(223)
        ax_vel_x.plot(t_array, vel_vec[:, 0])

        ax_vel_x.plot(t_array, vel_vec_opt[:, 0])

        ax_vel_y = fig.add_subplot(224)
        ax_vel_y.plot(t_array, vel_vec[:, 1])

        ax_vel_y.plot(t_array, vel_vec_opt[:, 1])

        plt.ioff()
        plt.show()
    return 

