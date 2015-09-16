from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 

import sys, os, random
import copy
import numpy as np 
import cPickle as cp

# import pytrajkin as pytk 
import pytrajkin_rxzero as pytkrxz
import pytrajkin_randemb as pytkre
import pytrajkin_painter as pytkpt
import pytrajkin_affineintegral as pytkai
import utils

class PyTrajKin_GUI(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyTrajKin_GUI - PyQt4')
        #size
        self.resize(1080, 720)
        self.move(400, 200)

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
        self.create_action()
        self.main_frame.show()
        self.disable_items()

    def create_menu(self):
        return

    def create_main_frame(self):
        self.main_frame = QWidget()

        hbox = QHBoxLayout()
        self.main_hbox = hbox
        vbox_ctrl_pnl = QVBoxLayout()
        vbox_ctrl_pnl.setAlignment(Qt.AlignTop)
        vbox_fig = QVBoxLayout()

        #for control panel
        #load button
        self.load_btn = QPushButton('Load')
        vbox_ctrl_pnl.addWidget(self.load_btn)
        #combo for letters
        self.char_lbl = QLabel('Characters')
        self.char_combbox = QComboBox()
        vbox_ctrl_pnl.addWidget(self.char_lbl)
        vbox_ctrl_pnl.addWidget(self.char_combbox)

        self.save_prf_fig_btn = QPushButton('Save Profile')
        self.save_vel_fig_btn = QPushButton('Save Velocity')
        vbox_ctrl_pnl.addWidget(self.save_prf_fig_btn)
        vbox_ctrl_pnl.addWidget(self.save_vel_fig_btn)

        #layout for parm sliders
        # <hyin/Feb-09-2015> move to stroke tab
        self.parm_sliders_layout = QVBoxLayout()
        self.parms_sliders = []
        vbox_ctrl_pnl.addLayout(self.parm_sliders_layout)
        self.ctrl_pnl_layout = vbox_ctrl_pnl

        #<hyin/Feb-09-2015> add tabs...
        self.tab_main = QTabWidget()
        self.tab_char = QWidget()
        self.tab_paint = pytkpt.PyTrajKin_Painter(analyzer=self.paint_traj_analyzer, 
            analyzer_drawfunc=self.analyzer_drawfunc)

        self.tab_main.addTab(self.tab_char, 'Character')
        self.tab_main.addTab(self.tab_paint, 'Painter')

        #for drawing part
        fig_hbox = QHBoxLayout()
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        #canvas
        gs = plt.GridSpec(4, 2)
        self.ax_char_prf = self.fig.add_subplot(gs[:, 0])
        self.ax_xvel = self.fig.add_subplot(gs[0, 1])
        self.ax_yvel = self.fig.add_subplot(gs[1, 1])
        self.ax_vel_prf = self.fig.add_subplot(gs[2, 1])
        self.ax_ang_prf = self.fig.add_subplot(gs[3, 1])

        self.ax_char_prf.hold(False)
        self.ax_xvel.hold(False)
        self.ax_yvel.hold(False)
        self.ax_vel_prf.hold(False)
        self.ax_ang_prf.hold(False)

        self.ax_char_prf.set_aspect('equal')

        self.fig.tight_layout()
        fig_hbox.addWidget(self.canvas)

        #control panel for char...
        char_ctrl_pnl = QHBoxLayout()

        char_ctrl_pnl_vlayout = QVBoxLayout()
        # self.idx_lbl = QLabel('Character Index')
        # self.idx_combbox = QComboBox()

        self.strk_lbl = QLabel('Stroke Index')
        self.strk_combbox = QComboBox()

        self.rand_btn = QPushButton('Synthesize')

        #tab for strokes...
        #at least there's one...
        self.tab_char_strk = QTabWidget()
        
        self.parm_sliders_layout = []
        self.parms_sliders = []
        self.strk_tab_wgt_lst_array = []
                
        # char_ctrl_pnl_vlayout.addWidget(self.idx_lbl)
        # char_ctrl_pnl_vlayout.addWidget(self.idx_combbox)
        char_ctrl_pnl_vlayout.addWidget(self.strk_lbl)
        char_ctrl_pnl_vlayout.addWidget(self.strk_combbox)
        char_ctrl_pnl_vlayout.addWidget(self.rand_btn, 20)

        char_ctrl_pnl.addLayout(char_ctrl_pnl_vlayout)
        char_ctrl_pnl.addWidget(self.tab_char_strk, 7)

        vbox_fig.addLayout(fig_hbox, 5)
        vbox_fig.addLayout(char_ctrl_pnl, 2)

        self.tab_char.setLayout(vbox_fig)

        #add layouts
        hbox.addLayout(vbox_ctrl_pnl, 1)
        #hbox.addLayout(vbox_fig, 5)
        hbox.addWidget(self.tab_main, 5)

        self.main_frame.setLayout(hbox)
        self.setCentralWidget(self.main_frame)
        return

    def create_status_bar(self):
        return

    def create_action(self):
        self.load_btn.clicked.connect(self.on_load_data)
        self.save_prf_fig_btn.clicked.connect(self.on_save_prf_fig)
        self.save_vel_fig_btn.clicked.connect(self.on_save_vel_fig)
        self.rand_btn.clicked.connect(self.on_synthesize)

        self.char_combbox.currentIndexChanged.connect(self.on_update_char_comb)
        self.strk_combbox.currentIndexChanged.connect(self.on_update_strk_comb)
        
        return

    def disable_items(self):
        return

    def on_save_prf_fig(self):
        curr_char = str(self.char_combbox.currentText())
        ax_subplot = self.ax_char_prf
        fname = curr_char + '_synthesis_sample_profile.pdf'
        self.save_fig_helper(ax_subplot, fname)
        return

    def on_save_vel_fig(self):
        curr_char = str(self.char_combbox.currentText())
        ax_subplot = self.ax_vel_prf
        fname = curr_char + '_synthesis_sample_velocity.pdf'
        self.save_fig_helper(ax_subplot, fname)
        return

    def save_fig_helper(self, ax_subplot, fname):
        #get extent...
        extent = ax_subplot.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(fname, bbox_inches=extent.expanded(1.3, 1.6))
        return

    def on_load_data(self):

        #<hyin/Feb-11-2015> structure to store trained feature parms...
        self.data_mdls = None
        #<hyin/Feb-12-2015> structure to store statistics
        # self.data_mdls_stat = None

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        default_dir = os.path.join(curr_dir, 'char_mdls')
        print 'loading data from ', default_dir

        #for each char model file, load files
        data_files = [ f for f in os.listdir(default_dir) 
        if os.path.isfile(os.path.join(default_dir, f)) and os.path.join(default_dir, f).endswith(".p") ]

        data_files_sorted  = sorted(data_files)

        for f in data_files_sorted:
            if self.data_mdls is None:
                self.data_mdls = dict()

            #check if name is duplicated
            if f[0] in self.data_mdls:
                print 'Model for character {0} is already loaded. Please check model files.'.format(f[0])
                continue

            tmp_mdl = pytkre.TrajKinMdl(use_kin=True)
            if tmp_mdl.load(os.path.join(default_dir, f)):
                print 'Successfully loaded data for character {0}'.format(f[0])
                self.data_mdls[f[0]] = tmp_mdl
            else:
                print 'Failed to load data for character {0}'.format(f[0])

        
        #refresh char comb
        self.char_combbox.blockSignals(True)
        self.char_combbox.clear()

        #only add valid letters...
        for key in self.data_mdls:
            if len(key) > 1:
                continue
            if (ord(key)<=ord('z') and ord(key)>=ord('a')) or (ord(key)<=ord('Z') and ord(key)>=ord('A')):
                self.char_combbox.addItem(key)

        self.char_mdl = []
        self.dt = 0.01
        self.char_combbox.blockSignals(False)
        self.on_update_char_comb(None)

        return

    def on_update_char_comb(self, idx):
        #refresh strk comb
        curr_char = str(self.char_combbox.currentText())

        self.char_mdl = []
        #check if trained feature parameters are there
        if self.data_mdls is not None:
            if curr_char in self.data_mdls:
                pass
            else:
                print 'No model for character {0}'.format(curr_char)
                return

        #prepare a sample
        self.on_synthesize()

        return

    def on_update_strk_comb(self, idx):
        self.refresh_parm_sliders_layout()
        return


    def refresh_parm_sliders_layout(self):
        #get current stroke
        if not self.char_mdl:
            return
        curr_idx = int(self.strk_combbox.currentText())
        #note clear does not delete the widgets...
        self.tab_char_strk.clear()
        for i in range(len(self.strk_tab_wgt_lst_array[curr_idx])):
            self.tab_char_strk.addTab(self.strk_tab_wgt_lst_array[curr_idx][i], 'Comp {0}'.format(i))
        return

    def clear_parm_sliders_layout(self):
        #clean layout, see relevant stackoverflow threads
        for stroke_layouts in self.parm_sliders_layout:
            for layout in stroke_layouts:
                for i in reversed(range(layout.count())): 
                    widgetToRemove = layout.itemAt( i ).widget()
                    # get it out of the layout list
                    layout.removeWidget( widgetToRemove )
                    # remove it form the gui
                    widgetToRemove.setParent( None )

        self.parm_sliders_layout = []
        for i in reversed(range(self.tab_char_strk.count())):
            widgetToRemove = self.tab_char_strk.widget(i)
            widgetToRemove.setParent( None )
            widgetToRemove.deleteLater()
        self.tab_char_strk.clear()
        self.parms_sliders = []
        self.strk_tab_wgt_lst_array = []
        return

    def populate_parm_sliders(self):
        if not self.char_mdl:
            return
        select_idx = self.strk_combbox.currentIndex()
        for curr_idx in range(len(self.char_mdl)):
            #check model parms, now only deal with the first stroke
            #for each component, interested parameters are : D, mu, sig, delta_theta

            i = 0
            tmp_strk_layout_lst = []
            tmp_strk_parm_slider_lst = []
            tmp_strk_tab_wgt_lst = []
            if 'opt_parms' not in self.char_mdl[curr_idx]:
                print 'Stroke {0} is not with valid kinematic parms, ignore...'.format(curr_idx)
            else:
                tmp_effective_parms = np.array(self.char_mdl[curr_idx]['opt_parms'])[:, [0, 2,3,4, 5]]
                for stroke_parm in tmp_effective_parms:
                    tmp_slider_lbl = QLabel('D, mu, sig, theta_s, theta_e')
                    tmp_parm_sliders_layout = QHBoxLayout()
                    tmp_parm_sliders_lst = []
                    for tmp_parm in stroke_parm:
                        tmp_parm_slider = QSlider()
                        tmp_parm_slider.setOrientation(Qt.Vertical)
                        tmp_parm_slider.setMaximum(100)
                        tmp_parm_slider.setValue(50)
                        #connect message to plot event
                        tmp_parm_slider.valueChanged.connect(self.plot_data)
                        # self.parms_sliders.append(tmp_parm_slider)
                        # self.parm_sliders_layout.addWidget(tmp_parm_slider)
                        tmp_parm_sliders_lst.append(tmp_parm_slider)
                        tmp_parm_sliders_layout.addWidget(tmp_parm_slider)
                    tmp_strk_slider_wgt = QWidget()
                    tmp_strk_slider_wgt.setLayout(tmp_parm_sliders_layout)
                    if select_idx == curr_idx:
                        self.tab_char_strk.addTab(tmp_strk_slider_wgt, 'Comp {0}'.format(i))
                    tmp_strk_layout_lst.append(tmp_parm_sliders_layout)
                    tmp_strk_parm_slider_lst.append(tmp_parm_sliders_lst)
                    tmp_strk_tab_wgt_lst.append(tmp_strk_slider_wgt)
                    i+=1

            self.parm_sliders_layout.append(tmp_strk_layout_lst)
            self.parms_sliders.append(tmp_strk_parm_slider_lst)
            self.strk_tab_wgt_lst_array.append(tmp_strk_tab_wgt_lst)

        return


    def traj_eval_helper(self, strk_idx, t_array, parms, x0, y0):
        """
        evaluate a trajectory with current parameters and perturbations...
        """
        #opt
        opt_parms = copy.copy(parms)
        #get noise
        num_parm_per_comp = 5
        noise_ratio_array = []

        for slider_lst in self.parms_sliders[strk_idx]:
            for slider in slider_lst:
                noise_ratio_array.append(float((slider.value()-50))/100)

        noise_ratio_array = np.reshape(noise_ratio_array, (-1, num_parm_per_comp))

        for row in range(opt_parms.shape[0]):
            # opt_parms[row][0] += noise_ratio_array[row][0] * np.abs(opt_parms[row][0]) * 0.8
            # opt_parms[row][2] += noise_ratio_array[row][1] * np.abs(opt_parms[row][2]) * 0.5
            # opt_parms[row][3] += noise_ratio_array[row][2] * np.abs(opt_parms[row][3]) * 0.5
            opt_parms[row][0] += noise_ratio_array[row][0] * 5
            opt_parms[row][2] += noise_ratio_array[row][1] * 1.0
            opt_parms[row][3] += noise_ratio_array[row][2] * 1.0

            #theta_s & theta_e: noise is applied to delta_theta
            # opt_theta_s = opt_parms[row][4]
            # opt_theta_e = opt_parms[row][5]
            # opt_parms[row][4] = (opt_theta_s + opt_theta_e)/2 - (opt_theta_e-opt_theta_s) * (1 + noise_ratio_array[row][3]*2) / 2
            # opt_parms[row][5] = (opt_theta_s + opt_theta_e)/2 + (opt_theta_e-opt_theta_s) * (1 + noise_ratio_array[row][3]*2) / 2
            opt_parms[row][4] += noise_ratio_array[row][3] * 2*np.pi
            opt_parms[row][5] += noise_ratio_array[row][4] * 2*np.pi

        traj_opt, vel_vec_opt =  pytkrxz.rxzero_traj_eval(opt_parms, t_array, x0, y0)
        return traj_opt, vel_vec_opt, opt_parms

    def plot_data(self):
        #plot data
        #evaluate base
        curr_data = [ mdl['stroke'] for mdl in self.char_mdl ]

        bFirstStroke = True
        last_stroke_end_t = 0.0    
        if 'vel_profile' not in self.char_mdl[0]:
            print 'no velocity profile stored'
            return
        #currently, only consider one stroke
        for strk_idx, stroke in enumerate(curr_data):
            #vel profile
            vel_profile = self.char_mdl[strk_idx]['vel_profile']

            #t_array
            t_array = np.linspace(0, 1.0, len(stroke)) + last_stroke_end_t

            last_stroke_end_t = t_array[-1]
            #vel vec & theta
            vel_vec = np.diff(stroke, axis=0) / (t_array[1] - t_array[0])
            #theta = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])

            theta = utils.get_continuous_ang(stroke)

            #plot
            #only data
            #char profile
            self.ax_char_prf.plot(stroke[:, 0], -stroke[:, 1], 'b', linewidth=4.0)
            self.ax_char_prf.set_title('Character Profile', fontsize=8)
            self.ax_char_prf.set_xlim([-1.5, 1.5])
            self.ax_char_prf.set_ylim([-1.5, 1.5])

            self.ax_char_prf.set_xticks([])
            self.ax_char_prf.set_yticks([])

            #vel_x & vel_y
            self.ax_xvel.plot(t_array[0:-1], vel_vec[:, 0], 'b', linewidth=4.0)
            self.ax_xvel.set_title('X Velocity', fontsize=8)
            self.ax_xvel.set_xlabel('Time (s)', fontsize=8)
            self.ax_xvel.set_ylabel('Velocity (Unit/s)', fontsize=8)
            self.ax_yvel.plot(t_array[0:-1], vel_vec[:, 1], 'b', linewidth=4.0)
            self.ax_yvel.set_title('Y Velocity', fontsize=8)
            self.ax_yvel.set_xlabel('Time (s)', fontsize=8)
            self.ax_yvel.set_ylabel('Velocity (Unit/s)', fontsize=8)
            #vel profile
            self.ax_vel_prf.plot(t_array, vel_profile, 'b', linewidth=4.0)
            self.ax_vel_prf.set_title('Velocity Maganitude', fontsize=8)
            self.ax_vel_prf.set_xlabel('Time (s)', fontsize=8)
            self.ax_vel_prf.set_ylabel('Maganitude (Unit/s)', fontsize=8)
            #ang profile
            self.ax_ang_prf.plot(t_array[0:-1], theta, 'b', linewidth=4.0)
            self.ax_ang_prf.set_title('Angular Position', fontsize=8)
            self.ax_ang_prf.set_xlabel('Time (s)', fontsize=8)
            self.ax_ang_prf.set_ylabel('Angular Position (rad)', fontsize=8)
            if bFirstStroke:
                self.ax_char_prf.hold(True)
                self.ax_xvel.hold(True)
                self.ax_yvel.hold(True)
                self.ax_vel_prf.hold(True)
                self.ax_ang_prf.hold(True)

                bFirstStroke = False

        colors = ['r', 'y', 'k', 'g', 'w']
        last_stroke_end_t = 0.0   
        for curr_idx in range(len(self.char_mdl)):
            #hold current drawings to add new curves
            #now only the first stroke
            #registration points...
            vel_profile = self.char_mdl[curr_idx]['vel_profile']
            t_array = np.linspace(0, 1.0, len(curr_data[curr_idx]))
            if 'start_pnt' in self.char_mdl[curr_idx] and 'opt_parms' in self.char_mdl[curr_idx]:
                x0 = self.char_mdl[curr_idx]['start_pnt'][0]
                y0 = self.char_mdl[curr_idx]['start_pnt'][1]
                opt_parms = np.array(self.char_mdl[curr_idx]['opt_parms'])
 
                traj_opt, vel_vec_opt, opt_parms = self.traj_eval_helper(curr_idx, t_array, opt_parms, x0, y0)
                theta_opt = utils.get_continuous_ang(traj_opt)
                self.ax_char_prf.plot(traj_opt[:, 0], -traj_opt[:, 1], 'r', linewidth=4.0)
                self.ax_vel_prf.plot(t_array[:]+last_stroke_end_t, np.sum(vel_vec_opt**2, axis=1)**(1./2), 'r', linewidth=4.0)

                self.ax_xvel.plot(t_array[:]+last_stroke_end_t, vel_vec_opt[:, 0], 'r', linewidth=4.0)
                self.ax_yvel.plot(t_array[:]+last_stroke_end_t, vel_vec_opt[:, 1], 'r', linewidth=4.0)
                self.ax_ang_prf.plot(t_array[1:]+last_stroke_end_t, theta_opt, 'r', linewidth=4.0)

                #for each component
                for parm in opt_parms:
                    comp_traj, comp_vel_vec = pytkrxz.rxzero_traj_eval([parm], t_array, x0, y0)
                    self.ax_vel_prf.plot(t_array[:]+last_stroke_end_t, np.sum(comp_vel_vec**2, axis=1)**(1./2), 'g', linewidth=2.5)
                    self.ax_xvel.plot(t_array[:]+last_stroke_end_t, comp_vel_vec[:, 0], 'g', linewidth=2.5)
                    self.ax_yvel.plot(t_array[:]+last_stroke_end_t, comp_vel_vec[:, 1], 'g', linewidth=2.5)
                last_stroke_end_t += t_array[-1]
            else:
                last_stroke_end_t += t_array[-1]

        self.ax_char_prf.hold(False)
        self.ax_xvel.hold(False)
        self.ax_yvel.hold(False)
        self.ax_vel_prf.hold(False)
        self.ax_ang_prf.hold(False)
        self.canvas.draw()
        return

    def on_synthesize(self):
        curr_char = str(self.char_combbox.currentText())
        curr_mdl = self.data_mdls[curr_char]
        #sample from rf model
        strk_num = curr_mdl.sample_strk_num()
        sample_data = []
        if strk_num is not None:
            print 'Generate a sample consisted of {0} strokes'.format(strk_num+1)
            #select for the model list
            mdl_lst = curr_mdl.model_[strk_num+1]
            char_data = []
            sample_parms = []
            sample_noise = []
            for mdl in mdl_lst:
                #for each stroke...
                tmp_sample = curr_mdl.sample_from_rf_mdl(mdl)
                sample_data, tree_idx, leaf_idx, noise = tmp_sample[0]
                if (tree_idx, leaf_idx) in mdl['kinparms_dict']:
                    parms = mdl['kinparms_dict'][tree_idx, leaf_idx]
                else:
                    parms = -1
                char_data.append(sample_data)
                sample_parms.append(parms)
                sample_noise.append(noise)

            #prepare char model
            self.char_mdl = []
            for strk_idx in range(len(char_data)):
                tmp_char_mdl = dict()
                tmp_char_data = np.reshape(char_data[strk_idx], (2, -1))

                tmp_char_mdl['char_sample'] = tmp_char_data
                #evaluate model
                if sample_parms[strk_idx] == -1:
                    #no valid parameters
                    tmp_char_mdl['stroke'] = tmp_char_data
                else:
                    tmp_char_mdl['start_pnt'] = sample_parms[strk_idx][0]
                    tmp_char_mdl['opt_parms'] = sample_parms[strk_idx][1]
                    t_array = np.linspace(0, 1.0, len(char_data[strk_idx])/2)

                    eval_traj, eval_vel = pytkrxz.rxzero_traj_eval(tmp_char_mdl['opt_parms'], t_array, sample_parms[strk_idx][0][0], sample_parms[strk_idx][0][1])
                    tmp_char_mdl['stroke'] = eval_traj
                    tmp_char_mdl['vel_profile'] = np.sum(eval_vel**2, axis=1)**(1./2)

                self.char_mdl.append(tmp_char_mdl)

            self.strk_combbox.blockSignals(True)
            self.strk_combbox.clear()
            self.strk_combbox.addItems(map(str, range(len(char_data))))
            self.strk_combbox.blockSignals(False)
            self.clear_parm_sliders_layout()
            #self.on_update_strk_comb(None)
            
            self.clear_parm_sliders_layout()
            self.populate_parm_sliders()
            
            self.plot_data()
                
        return char_data, sample_parms, sample_noise

    def analyzer_drawfunc(self, ax, data):
        lines = []
        if 'processed_data' in data:
            processed_data = data['processed_data']
            processed_data_lines = utils.plot_trajs_helper(ax, processed_data)
            lines = lines + processed_data_lines
        if 'recons_strks' in data:
            recons_data = data['recons_strks']
            recons_data_lines = utils.plot_trajs_helper(ax, recons_data, linespec='-r')
            lines = lines + recons_data_lines
        if 'adjusted_comp_idx' in data and 'adjusted_comp' in data and 'strk_parms_lst' in data:
            #highlight the adjusted component
            for strk_idx, strk_data in enumerate(processed_data):
                for adjusted_comp_parms in data['adjusted_comp'][strk_idx][data['adjusted_comp_idx'][strk_idx]]:
                    #from the adjusted component parameters to infer registration points...
                    reg_pnts = utils.registration_points_from_parms(adjusted_comp_parms)
                    # print reg_pnts
                    modified_section = strk_data[int(np.max([reg_pnts[0], 0.0]) * len(strk_data)):int(np.min([reg_pnts[3], 1.0]) * len(strk_data)), :]
                    #draw this sectoin
                    # print modified_section
                    highlt_section, = ax.plot(modified_section[:, 0], modified_section[:, 1], '-g', linewidth=4.0)
                    lines = lines + [highlt_section]

        if 'opt_tmpl' in data:
            tmpl_data = data['opt_tmpl']
            tmpl_data_lines = utils.plot_trajs_helper(ax, tmpl_data, linespec='-k')
            lines = lines + tmpl_data_lines
        return lines

    def trajkin_parm_exploration_to_fit_template(self, sample, template):
        #detect improvement suggestion in the span of kinematics feature space
        #first extract kinematics for the sample
        recons_sample = []
        strk_parms_lst = []
        fit_parms_lst = []
        adjusted_comp = []
        adjusted_comp_idx = []
        print sample
        for strk_idx, strk in enumerate(sample):
            print '================================='
            print 'For stroke ', strk_idx
            print '================================='
            t_array = np.linspace(0, 1.0, len(strk))
            strk_parms = pytkrxz.rxzero_train(strk, global_opt_itrs=1)
            strk_parms_lst.append(strk_parms)
            eval_traj, eval_vel = pytkrxz.rxzero_traj_eval(strk_parms, t_array, strk[0, 0], strk[0, 1])

            #<hyin/Jun-8th-2015> try another direction, fit the full parameters to the template
            # free_comp_idx = range(len(strk_parms))
            # fit_parms = pytkrxz.fit_parm_component_with_global_optimization(template[strk_idx], strk_parms, free_comp_idx=free_comp_idx, maxIters=1)

            # comp_modulation_lst = []
            # recons_parms_lst = []
            # for comp_idx, parm_comp in enumerate(strk_parms):
            #     print 'Examining component ', comp_idx
            #     recons_parms = [parm for parm in fit_parms[0]]
            #     # print recons_parms, comp_idx, parm_comp
            #     recons_parms[comp_idx] = parm_comp
            #     recons_parms_lst.append(recons_parms)
            #     comp_modulation_lst.append(np.sum(np.abs(parm_comp-fit_parms[0][comp_idx])))
            #     # recons_err_comp_lst.append(recons_err)

            # # find the smallest one
            # significant_comp_idx = np.argmax(comp_modulation_lst)
            # print 'The most significant component: ', significant_comp_idx
            # print comp_modulation_lst[significant_comp_idx]
            # print recons_parms_lst[significant_comp_idx]
            # recons_eval_traj, recons_eval_vel = pytkrxz.rxzero_traj_eval(recons_parms_lst[significant_comp_idx], t_array, 
            #     template[strk_idx][0, 0], template[strk_idx][0, 1])
            #     # strk[0, 0], strk[0, 1])

            # recons_sample.append(recons_eval_traj)
            # adjusted_comp.append([recons_parms_lst[significant_comp_idx]])
            # adjusted_comp_idx.append(significant_comp_idx)

            # recons_sample.append(eval_traj)
            #call the global optimization in rxzero to see how can we fit the template by modulating the extracted parameters
        #     fit_parms = pytkrxz.rxzero_global_optimization(template[strk_idx], strk_parms, dt=0.01, maxIters=1)
        #     fit_parms_lst.append(fit_parms)
        #     fit_eval_traj, fit_eval_vel = pytkrxz.rxzero_traj_eval(fit_parms, t_array, strk[0, 0], strk[0, 1])
        #     # recons_sample.append(fit_eval_traj)
            #for this stroke, see which component can lead us to a better reconstruction towards the template
            recons_err_comp_lst = []
            comp_modulation_lst = []

            if len(strk_parms) == 1:
                #only one component...
                print 'Examining the only components'
                opt_parms, recons_err = pytkrxz.fit_parm_component_with_global_optimization(template[strk_idx], strk_parms, free_comp_idx=[[0]], maxIters=1)
                # opt_parms, recons_err = pytkrxz.fit_parm_scale_ang_component_with_global_optimization(template[strk_idx], strk_parms, free_comp_idx=[[0]], maxIters=1)
                comp_modulation_lst.append(opt_parms)
                recons_err_comp_lst.append(recons_err)

                significant_comp_idx = np.argmin(recons_err_comp_lst)
                print 'The only significant components: ', significant_comp_idx, significant_comp_idx+1
                print comp_modulation_lst[significant_comp_idx]
                recons_eval_traj, recons_eval_vel = pytkrxz.rxzero_traj_eval(comp_modulation_lst[significant_comp_idx], t_array, 
                    template[strk_idx][0, 0], template[strk_idx][0, 1])
                    # strk[0, 0], strk[0, 1])

                recons_sample.append(recons_eval_traj)
                adjusted_comp.append(comp_modulation_lst[significant_comp_idx])
                adjusted_comp_idx.append([significant_comp_idx])
            else:
                for comp_idx, parm_comp in enumerate(strk_parms[:-1]):
                    print 'Examining components ', comp_idx, comp_idx+1
                    free_comp_idx = [[comp_idx], [comp_idx+1]]
                    opt_parms, recons_err = pytkrxz.fit_parm_component_with_global_optimization(template[strk_idx], strk_parms, free_comp_idx=free_comp_idx, maxIters=1)
                    # opt_parms, recons_err = pytkrxz.fit_parm_scale_ang_component_with_global_optimization(template[strk_idx], strk_parms, free_comp_idx=free_comp_idx, maxIters=1)
                    
                    comp_modulation_lst.append(opt_parms)
                    recons_err_comp_lst.append(recons_err)

                # find the smallest one
                significant_comp_idx = np.argmin(recons_err_comp_lst)
                print 'The most significant components: ', significant_comp_idx, significant_comp_idx+1
                print comp_modulation_lst[significant_comp_idx]
                recons_eval_traj, recons_eval_vel = pytkrxz.rxzero_traj_eval(comp_modulation_lst[significant_comp_idx], t_array, 
                    template[strk_idx][0, 0], template[strk_idx][0, 1])
                    # strk[0, 0], strk[0, 1])

                recons_sample.append(recons_eval_traj)
                adjusted_comp.append(comp_modulation_lst[significant_comp_idx])
                adjusted_comp_idx.append([significant_comp_idx, significant_comp_idx+1])

        # #blend these parms to see if this would give us meaningful instructions...
        # comb_parms_lst = []
        # replace_strk_idx = [0]
        # replace_comp_idx = [0]
        # comb_recons_sample = []
        # for strk_idx, strk_parms in enumerate(strk_parms_lst):
        #     t_array = np.linspace(0, 1.0, len(sample[strk_idx]))
        #     comb_strk_parms = []
        #     for comp_idx, strk_parm_comp in enumerate(strk_parms):
        #         if strk_idx in replace_strk_idx and comp_idx in replace_comp_idx:
        #             print 'replace...'
        #             comb_strk_parms.append(fit_parms_lst[strk_idx][comp_idx])
        #         else:
        #             comb_strk_parms.append(strk_parm_comp)
        #         #evaluate comb trajectory 
        #     comb_eval_traj, comb_eval_vel = pytkrxz.rxzero_traj_eval(comb_strk_parms, t_array, strk[0, 0], strk[0, 1])
        #     comb_recons_sample.append(comb_eval_traj)
        #     comb_parms_lst.append(comb_strk_parms)

        return strk_parms_lst, recons_sample, adjusted_comp, adjusted_comp_idx

    def paint_traj_analyzer(self, data):
        if not data:
            #dont proceed if the data is empty
            return
        res_data = dict()
        #first, we need to preprocess the data to ensure it is aligned with the models
        # for strk in data:
        #     print strk
        #     print len(strk)
        processed_data = utils.smooth_and_interp_trajs(data)
        res_data['processed_data'] = processed_data
        # print processed_data
        #find the closest letter from the current models...
        #try affine integral invariant features...
        res_data['processed_data_affine_invar_feat'] = [pytkai.affine_int_invar_signature_scaled(strk[:, 0], strk[:, 1]) for strk in processed_data]
        #in terms of euclidean distance in the cartesian space...
        #similar to do the pointwise min of the cost ensemble
        #but note that, here we need to figure out a template, so classification-like decision
        #is made
        curr_char = str(self.char_combbox.currentText())
        if curr_char is not '':
            curr_mdl = self.data_mdls[curr_char]
            #for each of the strokes, find the best matched stroke...
            num_strk = len(processed_data)
            tmpl_strks = []
            if num_strk in curr_mdl.model_:
                tmpl_mdl = curr_mdl.model_[num_strk]

                #use this
                for strk_idx, strk in enumerate(processed_data):
                    tmpl_strk_mdl = tmpl_mdl[strk_idx]

                    opt_dist = np.inf
                    opt_tree_idx = np.nan
                    opt_leaf_idx = np.nan
                    opt_tmpl = None

                    # print tmpl_strk_mdl['samples_dict']
                    #enumerate all sub-models for this stroke
                    for k, d in tmpl_strk_mdl['samples_dict'].iteritems():
                        tree_idx, leaf_idx = k
                        if not d:
                            #d must not be empty
                            continue
                        mean_sample = np.reshape(np.mean(d, axis=0), (2, -1)).transpose()
                        mean_sample[:, 1] = -mean_sample[:, 1]  #need to flip the vertical direction for uji_data
                        #aligning the starting point?
                        #dist = np.sum(np.sum((strk - mean_sample - strk[0, :] + mean_sample[0, :]) ** 2, axis=1))**.5
                        #dist = np.sum(np.sum((strk - mean_sample) ** 2, axis=1))**.5
                        #L1-norm?
                        # dist = np.sum(np.sum((np.abs(strk - mean_sample)), axis=1))
                        #hausdorff distance?
                        dist = utils.hausdorff_distance(strk, mean_sample)
                        # affine invariant feature for the mean trajectory
                        affine_invar_feat = pytkai.affine_int_invar_signature_scaled(mean_sample[:, 0], mean_sample[:, 1])
                        # compare
                        dist += np.sum(np.sum(np.abs(res_data['processed_data_affine_invar_feat'][strk_idx] - affine_invar_feat)))

                        if dist < opt_dist:
                            opt_dist = dist
                            opt_tree_idx = tree_idx
                            opt_leaf_idx = leaf_idx
                            opt_tmpl = mean_sample

                    tmpl_strks.append(opt_tmpl)
                res_data['opt_tmpl'] = tmpl_strks
                strk_parms_lst, recons_strks, adjusted_comp, adjusted_comp_idx = self.trajkin_parm_exploration_to_fit_template(processed_data, tmpl_strks)
                res_data['strk_parms_lst'] = strk_parms_lst
                res_data['recons_strks'] = recons_strks
                res_data['adjusted_comp'] = adjusted_comp
                res_data['adjusted_comp_idx'] = adjusted_comp_idx
            else:
                #TODO: find the model with closest number of strokes
                pass


        else:
            print 'No character template exists. Forgot to load models?'

        return res_data

def main():
    app = QApplication(sys.argv)
    dp = PyTrajKin_GUI()
    dp.show()
    app.exec_()
    return

if __name__ == '__main__':
    main()