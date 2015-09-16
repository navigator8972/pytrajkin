"""
module for affine integral invariant analysis for given 2D planar
curve
 see S. Feng, et al, Classification of curves in 2D and 3D via affine
 integral signatures
"""

import numpy as np
import matplotlib.pyplot as plt

def affine_int_invar_I0(X, Y, t):
    # AFFINE_INT_INVAR_I_0 
    # Calculate affine integral invariant I0 for given X and Y
    #  see S. Feng, et al, Classification of curves in 2D and 3D via affine
    #  integral signatures
    #    Detailed explanation goes here
    # X: X(t)
    # Y: Y(t)
    # t: curve parm, it is an index
    if t >= len(X) or t >= len(Y):
        #invalid index
        print 'AFFINE_INT_INVAR_I_0: invalid index.'
        res = None
    else:
        res = np.sqrt((X[t] - X[0])**2 + (Y[t] - Y[0])**2)

    return res

def affine_int_invar_I1(X, Y, t):
    # %AFFINE_INT_INVAR_I_1 
    # %Calculate affine integral invariant I1 for given X and Y
    # % see S. Feng, et al, Classification of curves in 2D and 3D via affine
    # % integral signatures
    # %   Detailed explanation goes here
    # %X: X(t)
    # %Y: Y(t)
    # %t: curve parm, it is an index
    if t >= len(X) or t >= len(Y) or len(X) != len(Y):
        #invalid index
        print 'AFFINE_INT_INVAR_I_1: invalid index.'
        res = None
    else:
        X_invar = X - X[0]
        Y_invar = Y - Y[0]
        Y_invar_der = np.diff(Y_invar)
        Y_invar_der = np.concatenate([[0], Y_invar_der])
        #integral
        int_sum = np.sum(X_invar * Y_invar_der)
        # for i in range(t+1):
        #     int_sum += X_invar[i]*Y_invar_der[i]
        res = int_sum - X_invar[t] * Y_invar[t] / float(2)

    return res

def affine_int_invar_I2(X, Y, t):
    # %AFFINE_INT_INVAR_I_2 
    # %Calculate affine integral invariant I2 for given X and Y
    # % see S. Feng, et al, Classification of curves in 2D and 3D via affine
    # % integral signatures
    # %   Detailed explanation goes here
    # %X: X(t)
    # %Y: Y(t)
    # %t: curve parm
    if t >= len(X) or t >= len(Y) or len(X) != len(Y):
        #invalid index
        print 'AFFINE_INT_INVAR_I_2: invalid index.'
        res = None
    else:
        X_invar = X - X[0]
        Y_invar = Y - Y[0]
        Y_invar_der = np.diff(Y_invar)
        Y_invar_der = np.concatenate([[0], Y_invar_der])
        int_sum_1 = np.sum(X_invar*Y_invar*Y_invar_der)
        int_sum_2 = np.sum(X_invar**2 * Y_invar_der)
        res = X_invar[t] * int_sum_1 - Y_invar[t] / float(2) * int_sum_2 - X_invar[t]**2*Y_invar[t]**2
    
    return res

def affine_int_invar_signature(X, Y):
    """
    extract affine integral invariant features from given 2D trajectory
    X:  X coordinates
    Y:  Y coordinates
    """
    if len(X) != len(Y):
        print 'AFFINE_INT_INVAR_SIGNATURE: inconsistent length of X and Y.'

    I0 = [affine_int_invar_I0(X, Y, i) for i in range(len(X))]
    I1 = [affine_int_invar_I1(X, Y, i) for i in range(len(X))]
    I2 = [affine_int_invar_I2(X, Y, i) for i in range(len(X))]

    return np.vstack((I0, I1, I2))

def affine_int_invar_signature_scaled(X, Y):
    """
    extract scaled affine integral invariant features from given 2D trajectory
    """
    feat_unscaled = affine_int_invar_signature(X, Y)
    if feat_unscaled is not None:
        res = feat_unscaled.copy()
        res[0, :] = feat_unscaled[0, :] / np.amax(np.abs(feat_unscaled[0, :]))
        res[1, :] = feat_unscaled[1, :] / np.amax(np.abs(feat_unscaled[1, :]))
        res[2, :] = feat_unscaled[2, :] / np.amax(feat_unscaled[1, :]**2)
    else:
        res = None

    return res

def affine_int_invar_signature_test(data):
    """
    the given data is a 2D array
    try to corrupt it with some affine transformation and noise, see 
    if the features give similar signatures
    """
    #get features for the original data
    original_feat = affine_int_invar_signature_scaled(data[:, 0], data[:, 1])
    if original_feat is None:
        return
    #plot the original one...

    axes = affine_int_invar_signature_plot_helper(data, original_feat)

    colors = ['g', 'r', 'y', 'k', 'm']
    for i in range(len(colors)):
        #have some random distorted samples...
        rot_theta = -np.pi/6 + np.pi/3 * np.random.rand()
        # rot_theta = 0.0
        scale_x = 0.5 + 1.5 * np.random.rand()
        scale_y = 0.5 + 1.5 * np.random.rand()
        skew = 0.0 + 0.1*np.random.rand()
        # scale_y  = scale_x
        #transform the data
        trans_data = affine_trans_rot_scale(data, rot_theta, scale_x, scale_y, skew)
        #get signature...
        trans_signature = affine_int_invar_signature_scaled(trans_data[:, 0], trans_data[:, 1])
        #plot this
        affine_int_invar_signature_plot_helper(trans_data, trans_signature, axes=axes, color=colors[i])
    plt.show()
    return

def affine_int_invar_signature_plot_helper(data, feature, axes=None, color='b'):
    if axes is None:
        #create axes by ourselves
        fig = plt.figure()
        # fig.set_size_inches(18.5, 10.5)

        ax_crv = fig.add_subplot(221)
        ax_I0 = fig.add_subplot(222)
        ax_I1 = fig.add_subplot(223)
        ax_I2 = fig.add_subplot(224)

        ax_crv.set_xlabel('X Coordinate', fontsize=14)
        ax_crv.set_ylabel('Y Coordinate', fontsize=14)

        ax_crv.grid(True)

        ax_I0.set_xlabel('t', fontsize=14)
        ax_I0.set_ylabel('I0', fontsize=14)

        ax_I0.grid(True)

        ax_I1.set_xlabel('t', fontsize=14)
        ax_I1.set_ylabel('I1', fontsize=14)

        ax_I1.grid(True)

        ax_I2.set_xlabel('t', fontsize=14)
        ax_I2.set_ylabel('I2', fontsize=14)

        ax_I2.grid(True)

        ax_crv.hold(True)
        ax_I0.hold(True)
        ax_I1.hold(True)
        ax_I2.hold(True)

        plt.tight_layout()
    else:
        ax_crv = axes[0]
        ax_I0 = axes[1]
        ax_I1 = axes[2]
        ax_I2 = axes[3]

    original_crv, = ax_crv.plot(data[:, 0], data[:, 1], color=color, linewidth=3.0)
    I0_crv, = ax_I0.plot(range(len(feature[0, :])), feature[0, :], color=color, linewidth=2.0)

    I1_crv, = ax_I1.plot(range(len(feature[1, :])), feature[1, :], color=color, linewidth=2.0)
    
    I2_crv, = ax_I2.plot(range(len(feature[2, :])), feature[2, :], color=color, linewidth=2.0)


    return [ax_crv, ax_I0, ax_I1, ax_I2]

def affine_trans_rot_scale(traj, theta=0.0, scale_x=1.0, scale_y=1.0, skew=0.0):
    """
    helper function to transform a trajectory with given rotational angle, skewing and scaling parameters
    """
    skew_x_mat = np.array([[1, skew, 0], [0, 1, 0], [0, 0, 1]])
    skew_y_mat = np.array([[1, 0, 0], [skew, 1, 0], [0, 0, 1]])
    trans_mat = np.array([  [np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta), np.cos(theta),  0],
                            [0,             0,              1]])
    # print traj, np.ones((len(traj), 1))
    coord = np.concatenate([traj, np.ones((len(traj), 1))], axis=1)
    trans_coord = trans_mat.dot(coord.T).T
    resx = trans_coord[:, 0] * scale_x
    resy = trans_coord[:, 1] * scale_y
    return np.vstack((resx, resy)).T