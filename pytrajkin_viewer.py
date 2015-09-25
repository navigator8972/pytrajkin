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

from scipy.fftpack import fft, ifft, fftfreq

import pytrajkin as pytk 
import pytrajkin_painter as pytkpt
import pytrajkin_crvfrq as pytkcf
import utils

class PyTrajKin_GUI(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyTrajKin_GUI - PyQt4')
        #size
        self.resize(1024, 768)
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
        #combo for indexes
        # <hyin/Feb-09-2015> move this to char tab
        # self.idx_lbl = QLabel('Index')
        # self.idx_combbox = QComboBox()
        # vbox_ctrl_pnl.addWidget(self.idx_lbl)
        # vbox_ctrl_pnl.addWidget(self.idx_combbox)
        #button for train
        self.train_btn = QPushButton('Train')
        self.train_char_btn = QPushButton('Train Character')
        #button for curvature frequency analysis
        # self.char_crvfreq_btn = QPushButton('Curvature Frequency')
        self.char_stat_btn = QPushButton('Character Statistics')
        self.train_all_btn = QPushButton('Train All')

        self.save_all_btn = QPushButton('Save All')

        vbox_ctrl_pnl.addWidget(self.train_btn)
        vbox_ctrl_pnl.addWidget(self.train_char_btn)
        # vbox_ctrl_pnl.addWidget(self.char_crvfreq_btn)
        vbox_ctrl_pnl.addWidget(self.char_stat_btn)
        vbox_ctrl_pnl.addWidget(self.train_all_btn)

        vbox_ctrl_pnl.addWidget(self.save_all_btn)
        #layout for parm sliders
        #<hyin/Feb-09-2015> move to stroke tab
        # self.parm_sliders_layout = QVBoxLayout()
        # self.parms_sliders = []
        # vbox_ctrl_pnl.addLayout(self.parm_sliders_layout)

        #<hyin/Feb-09-2015> add tabs...
        self.tab_main = QTabWidget()
        self.tab_char = QWidget()
        self.tab_stat = QWidget()
        #<hyin/Sep-25th-2015> tab for curvature frequency analysis
        self.tab_crvfreq = self.create_curvature_freq_tab()

        self.tab_main.addTab(self.tab_char, 'Character')
        self.tab_main.addTab(self.tab_stat, 'Statistics')
        self.tab_main.addTab(self.tab_crvfreq, 'Curvature Frequency')

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

        #<hyin/Feb-12-2015> drawing part for statistics
        self.stat_fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.stat_canvas = FigureCanvas(self.stat_fig)
        self.ax_stat_strk_hist = self.stat_fig.add_subplot(gs[0, :])
        self.ax_stat_mdl_hist = self.stat_fig.add_subplot(gs[1, :])
        self.ax_mean_char_prf = self.stat_fig.add_subplot(gs[2:, 0])
        self.ax_syn_char_prf = self.stat_fig.add_subplot(gs[2:, 1])

        self.ax_stat_strk_hist.hold(False)

        stat_wgt_layout = QVBoxLayout()
        stat_wgt_layout.addWidget(self.stat_canvas)

        #control panel for char...
        char_ctrl_pnl = QHBoxLayout()

        char_ctrl_pnl_vlayout = QVBoxLayout()
        self.idx_lbl = QLabel('Character Index')
        self.idx_combbox = QComboBox()

        self.strk_lbl = QLabel('Stroke Index')
        self.strk_combbox = QComboBox()

        self.rand_btn = QPushButton('Synthesize')

        #tab for strokes...
        #at least there's one...
        self.tab_char_strk = QTabWidget()
        
        self.parm_sliders_layout = []
        self.parms_sliders = []
        self.strk_tab_wgt_lst_array = []
                
        char_ctrl_pnl_vlayout.addWidget(self.idx_lbl)
        char_ctrl_pnl_vlayout.addWidget(self.idx_combbox)
        char_ctrl_pnl_vlayout.addWidget(self.strk_lbl)
        char_ctrl_pnl_vlayout.addWidget(self.strk_combbox)
        char_ctrl_pnl_vlayout.addWidget(self.rand_btn, 20)

        char_ctrl_pnl.addLayout(char_ctrl_pnl_vlayout)
        char_ctrl_pnl.addWidget(self.tab_char_strk, 7)

        vbox_fig.addLayout(fig_hbox, 5)
        vbox_fig.addLayout(char_ctrl_pnl, 2)

        self.tab_char.setLayout(vbox_fig)
        self.tab_stat.setLayout(stat_wgt_layout)

        #add layouts
        hbox.addLayout(vbox_ctrl_pnl, 1)
        #hbox.addLayout(vbox_fig, 5)
        hbox.addWidget(self.tab_main, 5)

        self.main_frame.setLayout(hbox)
        self.setCentralWidget(self.main_frame)
        return

    def create_curvature_freq_tab(self):
        self.crvfrq_tabwgt = QWidget()
        #painter
        self.crvfrq_painter = pytkpt.PyTrajKin_Painter(analyzer=self.crvfrq_analyzer, 
            analyzer_drawfunc=self.crvfrq_drawer)
        #plot
        self.crvfrq_pltfig = Figure((5.0, 4.0), dpi=100)
        self.crvfrq_pltcanvas = FigureCanvas(self.crvfrq_pltfig)
        self.ax_crvfrqplt = self.crvfrq_pltfig.add_subplot(111)

        #layout
        crvfrq_tab_layout = QVBoxLayout()
        crvfrq_tab_layout.addWidget(self.crvfrq_painter)
        crvfrq_tab_layout.addWidget(self.crvfrq_pltcanvas)

        self.crvfrq_tabwgt.setLayout(crvfrq_tab_layout)
        return self.crvfrq_tabwgt

    def create_status_bar(self):
        return

    def create_action(self):
        self.load_btn.clicked.connect(self.on_load_data)

        self.char_combbox.currentIndexChanged.connect(self.on_update_char_comb)
        self.idx_combbox.currentIndexChanged.connect(self.on_update_idx_comb)
        self.strk_combbox.currentIndexChanged.connect(self.on_update_strk_comb)
        
        self.train_btn.clicked.connect(self.on_train)
        self.train_char_btn.clicked.connect(self.on_train_char)
        self.char_stat_btn.clicked.connect(self.on_char_stat)
        self.train_all_btn.clicked.connect(self.on_train_all)

        self.save_all_btn.clicked.connect(self.on_save_all)
        return

    def disable_items(self):
        return

    def on_load_data(self):

        #<hyin/Feb-11-2015> structure to store trained feature parms...
        self.data_feats = None
        #<hyin/Feb-12-2015> structure to store statistics
        self.data_feats_stat = None

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        default_dir = os.path.join(curr_dir, 'data')
        fileName = QFileDialog.getOpenFileName(self, 'Open', default_dir, selectedFilter='*.p')
        if fileName:
            tmp_dict = cp.load(open(fileName, 'rb'))
            self.data_feats = tmp_dict['data_feats']
            self.data_feats_stat = tmp_dict['data_feats_stat']
            print 'Loaded from to {0}'.format(fileName)

        self.data = dict()
        self.data = cp.load(open('data/uji_data_interped.p', 'rb'))

        
        #refresh char comb
        self.char_combbox.blockSignals(True)
        self.char_combbox.clear()

        #only add valid letters...
        for key in self.data:
            if len(key) > 1:
                continue
            if (ord(key)<=ord('z') and ord(key)>=ord('a')) or (ord(key)<=ord('Z') and ord(key)>=ord('A')):
                self.char_combbox.addItem(key)

        self.char_mdl = []
        self.dt = 0.01
        self.char_combbox.blockSignals(False)
        self.on_update_char_comb(None)

        self.plot_statistics()

        return

    def on_update_char_comb(self, idx):
        #refresh idx comb
        curr_char = str(self.char_combbox.currentText())
        self.idx_combbox.blockSignals(True)
        self.idx_combbox.clear()
        self.idx_combbox.addItems(map(str, range(len(self.data[curr_char]))))
        self.idx_combbox.blockSignals(False)

        self.on_update_idx_comb(None)

        self.plot_statistics()
        return

    def on_update_idx_comb(self, idx):
        #release mdl
        self.char_mdl = []
        curr_char = str(self.char_combbox.currentText())
        curr_idx = int(self.idx_combbox.currentText())
        #check if trained feature parameters are there
        if self.data_feats is not None:
            if curr_char in self.data_feats:
                if len(self.data_feats[curr_char]) > curr_idx:
                    self.char_mdl = self.data_feats[curr_char][curr_idx]

        self.strk_combbox.blockSignals(True)
        self.strk_combbox.clear()
        self.strk_combbox.addItems(map(str, range(len(self.data[curr_char][curr_idx]))))
        self.strk_combbox.blockSignals(False)
        self.clear_parm_sliders_layout()
        #self.on_update_strk_comb(None)
        
        self.clear_parm_sliders_layout()
        self.populate_parm_sliders()
        
        self.plot_data()
        return

    def on_update_strk_comb(self, idx):
        self.refresh_parm_sliders_layout()
        return

    def on_train(self):
        #train model with current data
        curr_data = self.get_current_data()
        self.char_mdl = []
        #print curr_data
        for stroke in curr_data:
            #print 'training...'
            tmp_stroke_dict = dict()
            tmp_mdl = pytk.TrajKinMdl()
            tmp_mdl.x0 = stroke[0, 0]
            tmp_mdl.y0 = stroke[0, 1]
            tmp_vel_prf = tmp_mdl.get_vel_profile(stroke)/self.dt

            tmp_opt_parm, tmp_reg_pnts_array = tmp_mdl.train(stroke)
            #fill dict
            #tmp_stroke_dict['model'] = tmp_mdl
            tmp_stroke_dict['vel_profile'] = tmp_vel_prf
            tmp_stroke_dict['reg_pnts_array'] = tmp_reg_pnts_array
            #tmp_stroke_dict['init_guess'] = tmp_init_guess
            tmp_stroke_dict['opt_parms'] = tmp_opt_parm
            tmp_stroke_dict['start_pnt'] = stroke[0, :]

            self.char_mdl.append(tmp_stroke_dict)
        #print self.char_mdl
        #populate parms sliders & plot
        self.clear_parm_sliders_layout()
        self.populate_parm_sliders()

        self.plot_data()
        return

    def on_train_char(self):
        #print 'Train Char Button Clicked'
        """
        train current Character
        """
        curr_char = str(self.char_combbox.currentText())
        curr_char_data_lst = self.data[curr_char]
        print '========================================'
        print 'Training Character {0}...'.format(curr_char)
        print '========================================'
        char_data_feats_lst = []
        i = 0
        for char_data in curr_char_data_lst:
            print '==========================================='
            print 'Training the {0}-th character...'.format(i)
            print '==========================================='
            char_data_feats = []
            for stroke in char_data:
                tmp_stroke_dict = dict()
                tmp_mdl = pytk.TrajKinMdl()
                tmp_mdl.x0 = stroke[0, 0]
                tmp_mdl.y0 = stroke[0, 1]
                tmp_vel_prf = tmp_mdl.get_vel_profile(stroke)/self.dt

                tmp_opt_parm, tmp_reg_pnts_array = tmp_mdl.train(stroke)
                #fill dict
                #tmp_stroke_dict['model'] = tmp_mdl
                tmp_stroke_dict['vel_profile'] = tmp_vel_prf
                tmp_stroke_dict['reg_pnts_array'] = tmp_reg_pnts_array
                #tmp_stroke_dict['init_guess'] = tmp_init_guess
                tmp_stroke_dict['opt_parms'] = tmp_opt_parm
                tmp_stroke_dict['start_pnt'] = stroke[0, :]
                tmp_stroke_dict['pos_traj'] = copy.copy(stroke)

                #push back trained data
                char_data_feats.append(tmp_stroke_dict)

            char_data_feats_lst.append(char_data_feats)
            i+=1

        print '========================================'
        print 'Finish Training Character {0}.'.format(curr_char)
        print '========================================'

        #write to record
        if self.data_feats is None:
            #create new one
            self.data_feats = dict()
        else:
            pass

        self.data_feats[curr_char] = char_data_feats_lst

        select_idx = self.idx_combbox.currentIndex()
        self.char_mdl = self.data_feats[curr_char][select_idx]

        #populate parms sliders & plot
        self.clear_parm_sliders_layout()
        self.populate_parm_sliders()

        self.populate_statistics()
        
        self.plot_data()
        return  

    def on_char_stat(self):
        self.populate_statistics()
        return

    def on_train_all(self):
        print 'Train All Button Clicked'
        return

    def on_save_all(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        #construct data
        tmp_dict = dict()
        tmp_dict['data_feats'] = self.data_feats
        tmp_dict['data_feats_stat'] = self.data_feats_stat

        default_dir = os.path.join(curr_dir, 'data')
        fileName = QFileDialog.getSaveFileName(self, 'Save', default_dir, selectedFilter='*.p')
        if fileName:
            cp.dump(tmp_dict, open(fileName, 'wb'))
            print 'Saved to {0}'.format(fileName)
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

            tmp_effective_parms = np.array(self.char_mdl[curr_idx]['opt_parms'])[:, [0, 2,3,5]]
            i = 0
            tmp_strk_layout_lst = []
            tmp_strk_parm_slider_lst = []
            tmp_strk_tab_wgt_lst = []
            for stroke_parm in tmp_effective_parms:
                tmp_slider_lbl = QLabel('D, mu, sig, delta_theta')
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

    def populate_statistics(self):
        curr_char = str(self.char_combbox.currentText())

        if self.data_feats is None:
            print 'The character is not trained yet...'
            return
        if curr_char in self.data_feats:
            curr_data_feats = self.data_feats[curr_char]
        else:
            print 'The character is not trained yet...'
            return

        #first make a statistics about the number of strokes
        num_strk_array = []
        num_comps_array = []
        strk_comp_idx_dict = dict()
        for char_inst in curr_data_feats:
            num_strk_array.append(len(char_inst))
            tmp_num_comps = []
            for strk_inst in char_inst:
                tmp_num_comps.append(len(strk_inst['opt_parms']))
            num_comps_array.append(tmp_num_comps)
        comp_hist = np.array([np.sum(strk_array) for strk_array in num_comps_array])

        #store these
        if self.data_feats_stat is None:
            self.data_feats_stat = dict()
        else:
            pass

        self.data_feats_stat[curr_char] = dict()
        self.data_feats_stat[curr_char]['num_strk'] = num_strk_array
        self.data_feats_stat[curr_char]['num_comps'] = num_comps_array
        self.data_feats_stat[curr_char]['num_comp_hist'] = comp_hist

        #categorize number of strokes
        strk_occur, strk_freq = utils.generate_item_freq(num_strk_array)
        #initialize a list for storeing component categorization
        #prepare sub-list for each stroke scenario
        #each sub-list is contains lists with number of strokes
        #e.g., a character can be categorized as written by 2, 3, 5 strokes
        #then str_comp_occur_lst = [[ [], [] #for 2 strokes], [[], [], [] #for 3 strokes], [[], [], [], [], [] #for 5 strokes]]
        #str_comp_occur_lst = [[[] for j in strk_occur[i]] for i in range(len(strk_occur))]
        #this is not intuitive, try dictionary
        str_comp_occur_dict = dict()
        for strk_num in strk_occur:
            if strk_num not in str_comp_occur_dict:
                #each stroke scenario has a list of dictionaries for various number of components
                str_comp_occur_dict[strk_num] = [dict({'comp_num':[], 'prob':[], 'data':dict(), 'model':dict()}) for i in range(strk_num)]

        for char_inst in curr_data_feats:
            num_strk = len(char_inst)
            for i in range(num_strk):
                #ith stroke contains how much components
                tmp_comp_dict = str_comp_occur_dict[num_strk][i]['data']
                #check how many components we have here
                num_comps = len(char_inst[i]['opt_parms'])
                if num_comps not in tmp_comp_dict:
                    tmp_comp_dict[num_comps] = []
                #add parameters belong to this stroke/component categorization
                #format for each stroke (n components):
                #[x0, y0, D1, t01, mu1, sig1, theta_s1, theta_e1, ..., Dn, t0n, mun, sign, theta_sn, theta_en]
                tmp_comp_dict[num_comps].append(np.concatenate([char_inst[i]['start_pnt'], char_inst[i]['opt_parms'].flatten()]))
        #for each stroke/comp group, create item freq and model
        for strk_num, strk_dict_lst in str_comp_occur_dict.iteritems():
            for strk_dict in strk_dict_lst:
                comp_num_lst = []
                comp_num_prob = []
                for comp_num, comp_data in strk_dict['data'].iteritems():
                    comp_num_lst.append(comp_num)
                    comp_num_prob.append(len(comp_data))

                    #<hyin/Feb-13-2015> construct a statistical model here, now try PCA
                    W, V, mean, scale = utils.do_pca(np.array(comp_data))
                    strk_dict['model'][comp_num] = dict()
                    strk_dict['model'][comp_num]['eig_val'] = W
                    strk_dict['model'][comp_num]['eig_vec'] = V
                    strk_dict['model'][comp_num]['mean'] = mean
                    strk_dict['model'][comp_num]['scale'] = scale

                strk_dict['comp_num'] = comp_num_lst
                #strk_dict['prob'] = utils.generate_item_freq(comp_num_prob)
                strk_dict['prob'] = comp_num_prob
        str_comp_data_stat = dict({'strk_num':strk_occur, 'prob':strk_freq, 'data':str_comp_occur_dict})

        self.data_feats_stat[curr_char]['stat'] = str_comp_data_stat

        self.plot_statistics()
        return

    def plot_statistics(self):
        #plot statistics
        curr_char = str(self.char_combbox.currentText())
        #check if data is available

        if self.data_feats_stat is None:
            print 'No data feature statistics'
            return
        elif curr_char not in self.data_feats_stat:
            print 'No data feature statistics for character {0}'.format(curr_char)
            return
        else:
            pass

        n, bins, patches = self.ax_stat_strk_hist.hist(np.array(self.data_feats_stat[curr_char]['num_comp_hist'])
            , 20, normed=0, facecolor='green', alpha=0.75)
        self.ax_stat_strk_hist.set_title('Histogram of Number of Components', fontsize=8)
        self.ax_stat_strk_hist.set_xlabel('Number of Components', fontsize=8)
        self.ax_stat_strk_hist.set_ylabel('Occurrences', fontsize=8)

        #TODO: <hyin/Feb-12-2015> need further consideration about showing the statistics
        #it seems better to only show the ones with the most probable number of strokes/components
        #but then the feature might still be high-dimensional, 6 components might lead to 38 dim feature
        #((D_i, t0_i, mu_i, sig_i, theta_s_i, theta_e_i))_i + (x_0, y_0)
        #further dimension reduction? PCA? These seems to be highly nonlinear... GMM? RandForest?
        #how about the correlation between components, these will introduce a lot of parameters
        #the most conservatively might be just learn a density and synthesize from that, but the data seems a little
        #sparse
        
        #<hyin/Feb-13-2015> tried, but the full length of extracted feature does not seems to be mono-modal distributed
        #I think this somehow makes sense, as the parameters include theta_s & theta_e
        #probably a wise way is to model these parameters separately
        #D, t0, mu, sig, is related to velocity profile, without considering angular position
        #theta_s, theta_e, are the angular position of anchor points
        #x0, y0, start point of pen-down stroke
        #try to treat them separately, I guess (x0, y0) and (theta_s, theta_e) should be more multi-modal
        #may truncate the variability of (D, t0, mu, sig), e.g., ignore t0 which seems not that important
        #so the parameters can be further reduced
        #find an example to do synthesize that Gaussian distribution won't work...

        #probably we need GMM but since the data might not be enough, Random forest embedding might do the trick

        #<hyin/Feb-17th-2015> tried Random forest, but results seem not good, guess:
        #1. random trees embedding implementation of scikit-learn seems not understoodable for me, each tree is ExtraTreeRegressor, why?
        #   what the 'mse' criteria for deciding a partition? I'm expecting something like 'entroy' in classification counterpart
        #2. data might still be too sparse, such that each leaf is not very reasonable partition
        #
        #possible actions: 
        #1. shall we first learn a distribution of possible component parameters as primitives and then encode characters in the space
        # of motion primitives? The data problem might be mitigated as it might be applied to all characters
        #2. parition parameters according to their physical meaning? According to another Ref. of Plamondon:
        # (mu, sig) - neuromuscular parms; (t0) - activation time; (D, theta_s, theta_e) - geometry parms
        # however, they claim the most sensitive parameters seems to be t0, and (mu, sig) are most independent ones
        # this is anti our observations, probably due to their synthesize is targeting a unique user, but then what's the meaning to do this
        # we could probably achieve the same stuff by overfitting a tree structure, and then do KDE at each leaf
        # by controlling the bandwith, variability might be local enough to give reasonable synthesis. But this might not solve the problem of evaluting user writing

        #<hyin/Feb-18th-2015> anyway, let's make things easier, conduct partitioning on the original data and do parameter extraction for each
        #leaf, hopefully this will give more robust local statistics

        #the most probable instance
        if 'stat' not in self.data_feats_stat[curr_char]:
            print 'statistics is not fully formalized...'
            return

        mean_data = []
        curr_stat_data_dict = self.data_feats_stat[curr_char]['stat']
        most_prob_strk_idx = np.array(curr_stat_data_dict['prob']).argmax()
        print 'Most characters contain {0} strokes...'.format(curr_stat_data_dict['strk_num'][most_prob_strk_idx])

        most_strk_data_dict_lst = curr_stat_data_dict['data'][curr_stat_data_dict['strk_num'][most_prob_strk_idx]]
        for i in range(len(most_strk_data_dict_lst)):
            strk_dict = most_strk_data_dict_lst[i]
            #find the most probable number of components for this stroke
            most_prob_comp_idx = np.array(strk_dict['prob']).argmax()
            most_prob_comp_num = strk_dict['comp_num'][most_prob_comp_idx]
            print 'For characters contain {0} strokes, the {1}-th stroke most probably contains {2} components'.format(
                curr_stat_data_dict['strk_num'][most_prob_strk_idx], i+1, most_prob_comp_num)

            most_comp_data_model = strk_dict['model'][most_prob_comp_num]
            mean_data.append(most_comp_data_model['mean'])
            #use one of the data
            #mean_data[-1] = strk_dict['data'][most_prob_comp_num][0]
            #use median
            #mean_data[-1] = np.median(strk_dict['data'][most_prob_comp_num], axis=0)
            #cp.dump(strk_dict['data'][most_prob_comp_num], open('test_data.p', 'wb'))
            #use random forest embedding
            rf_mdl = utils.random_forest_embedding(strk_dict['data'][most_prob_comp_num])
            mean_data[-1] = (utils.sample_from_rf_mdl(rf_mdl))[0]

        #show eigen values for the first stroke
        width = 0.35 
        most_prob_comp_idx_strk_0 = np.array(most_strk_data_dict_lst[0]['prob']).argmax()
        most_prob_comp_strk_0 = most_strk_data_dict_lst[0]['comp_num'][most_prob_comp_idx_strk_0]
        #first stroke - model - with most probable number of components - eigen values
        #eig_vals = most_strk_data_dict_lst[0]['model'][most_prob_comp_strk_0]['eig_val']
        #first stroke - data - D - most probable number of components
        eig_vals = np.array(most_strk_data_dict_lst[0]['data'][most_prob_comp_strk_0])[:, 7:-1:6]
        #self.ax_stat_mdl_hist.bar(np.arange(len(eig_vals)), eig_vals, width, color='r')
        for row in eig_vals:
            self.ax_stat_mdl_hist.plot(np.arange(len(row)), row)
            self.ax_stat_mdl_hist.hold(True)
        self.ax_stat_mdl_hist.hold(False)
        #output percentage of eigen values
        # cumsum_eig_vals = np.cumsum(eig_vals)
        # print cumsum_eig_vals/cumsum_eig_vals[-1]
        # print eig_vals

        print mean_data
        #generate mean letter profile
        letter_prf = []
        for parm in mean_data:
            #decode parm for each stroke
            #the first two are x0, y0
            tmp_mdl = pytk.TrajKinMdl()
            tmp_mdl.x0 = parm[0]
            tmp_mdl.y0 = parm[1]
            tmp_mdl.mdl_parms_ = np.reshape(parm[2:], (-1, 6))
            tmp_pos, tmp_vel = tmp_mdl.eval(np.arange(0.0, 20.0, 0.01))
            letter_prf.append(tmp_pos)

        bFirstDraw = True
        for pos_traj in letter_prf:
            self.ax_mean_char_prf.plot(pos_traj[:, 0], -pos_traj[:, 1], 'b')
            if bFirstDraw:
                self.ax_mean_char_prf.hold(True)
                bFirstDraw = False

        self.ax_mean_char_prf.hold(False)

        self.stat_canvas.draw()
        return

    def plot_data(self):
        #plot data
        curr_data = self.get_current_data()
        bFirstStroke = True
        last_stroke_end_t = 0.0        
        #currently, only consider one stroke
        for stroke in curr_data:
            #vel profile
            vel_profile = utils.get_vel_profile(stroke)/self.dt
            #t_array
            t_array = np.arange(len(vel_profile))*self.dt + last_stroke_end_t
            last_stroke_end_t = t_array[-1]
            #vel vec & theta
            vel_vec = np.diff(stroke, axis=0)/self.dt
            #theta = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])

            theta = utils.get_continuous_ang(stroke)

            #plot
            #only data
            #char profile
            self.ax_char_prf.plot(stroke[:, 0], -stroke[:, 1], 'b')
            self.ax_char_prf.set_title('Character Profile', fontsize=8)
            self.ax_char_prf.set_xticks([])
            self.ax_char_prf.set_yticks([])
            #vel_x & vel_y
            self.ax_xvel.plot(t_array, vel_vec[:, 0], 'b')
            self.ax_xvel.set_title('X Velocity', fontsize=8)
            self.ax_xvel.set_xlabel('Time (s)', fontsize=8)
            self.ax_xvel.set_ylabel('Velocity (Unit/s)', fontsize=8)
            self.ax_yvel.plot(t_array, vel_vec[:, 1], 'b')
            self.ax_yvel.set_title('Y Velocity', fontsize=8)
            self.ax_yvel.set_xlabel('Time (s)', fontsize=8)
            self.ax_yvel.set_ylabel('Velocity (Unit/s)', fontsize=8)
            #vel profile
            self.ax_vel_prf.plot(t_array, vel_profile, 'b')
            self.ax_vel_prf.set_title('Velocity Maganitude', fontsize=8)
            self.ax_vel_prf.set_xlabel('Time (s)', fontsize=8)
            self.ax_vel_prf.set_ylabel('Maganitude (Unit/s)', fontsize=8)
            #ang profile
            self.ax_ang_prf.plot(t_array, theta, 'b')
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
            mdl = pytk.TrajKinMdl()
            mdl.x0 = self.char_mdl[curr_idx]['start_pnt'][0]
            mdl.y0 = self.char_mdl[curr_idx]['start_pnt'][1]
            mdl.mdl_parms_ = self.char_mdl[curr_idx]['opt_parms']
            vel_profile = self.char_mdl[curr_idx]['vel_profile']
            reg_pnts_array = self.char_mdl[curr_idx]['reg_pnts_array']
            t_array = np.arange(len(vel_profile))*self.dt

            #opt
            #get noise
            num_parm_per_comp = 4
            noise_ratio_array = []

            for slider_lst in self.parms_sliders[curr_idx]:
                for slider in slider_lst:
                    noise_ratio_array.append(float((slider.value()-50))/100)

            noise_ratio_array = np.reshape(noise_ratio_array, (-1, num_parm_per_comp))
            opt_parms = np.array(self.char_mdl[curr_idx]['opt_parms'])
            for row in range(opt_parms.shape[0]):
                opt_parms[row][0] += noise_ratio_array[row][0] * np.abs(opt_parms[row][0]) * 0.8
                opt_parms[row][2] += noise_ratio_array[row][1] * np.abs(opt_parms[row][2]) * 0.5
                opt_parms[row][3] += noise_ratio_array[row][2] * np.abs(opt_parms[row][3]) * 0.5
                #theta_s & theta_e: noise is applied to delta_theta
                opt_theta_s = opt_parms[row][4]
                opt_theta_e = opt_parms[row][5]
                opt_parms[row][4] = (opt_theta_s + opt_theta_e)/2 - (opt_theta_e-opt_theta_s) * (1 + noise_ratio_array[row][3]*2) / 2
                opt_parms[row][5] = (opt_theta_s + opt_theta_e)/2 + (opt_theta_e-opt_theta_s) * (1 + noise_ratio_array[row][3]*2) / 2
            traj_opt, vel_vec_opt = mdl.eval(t_array, opt_parms)
            theta_opt = utils.get_continuous_ang(traj_opt)
            self.ax_char_prf.plot(traj_opt[:, 0], -traj_opt[:, 1], 'r')
            self.ax_vel_prf.plot(t_array[:]+last_stroke_end_t, np.sum(vel_vec_opt**2, axis=1)**(1./2), 'r')

            self.ax_xvel.plot(t_array[:]+last_stroke_end_t, vel_vec_opt[:, 0], 'r')
            self.ax_yvel.plot(t_array[:]+last_stroke_end_t, vel_vec_opt[:, 1], 'r')
            self.ax_ang_prf.plot(t_array[1:]+last_stroke_end_t, theta_opt, 'r')

            #for each component
            for parm in opt_parms:
                comp_traj, comp_vel_vec = mdl.eval(t_array, [parm])
                self.ax_vel_prf.plot(t_array[:]+last_stroke_end_t, np.sum(comp_vel_vec**2, axis=1)**(1./2), 'g')
                self.ax_xvel.plot(t_array[:]+last_stroke_end_t, comp_vel_vec[:, 0], 'g')
                self.ax_yvel.plot(t_array[:]+last_stroke_end_t, comp_vel_vec[:, 1], 'g')
            last_stroke_end_t += t_array[-1]

        self.ax_char_prf.hold(False)
        self.ax_xvel.hold(False)
        self.ax_yvel.hold(False)
        self.ax_vel_prf.hold(False)
        self.ax_ang_prf.hold(False)
        self.canvas.draw()
        return

    def get_current_data(self):
        curr_char = str(self.char_combbox.currentText())
        curr_idx = self.idx_combbox.currentIndex()
        curr_data = self.data[curr_char][curr_idx]
        return curr_data

    def crvfrq_analyzer(self, data):
        #check if there is drawing on the canvas
        if data is not None and data:
            # print 'get drawn data'
            letter_trajs = data
        else:
            # print 'get letter data'
            #get current char for the letter trajectories
            curr_data = self.get_current_data()
            if curr_data is not None and curr_data:
                letter_trajs = curr_data

        res_data = dict()
        res_data['sections'] = []
        res_data['crvfrq'] = []
        res_data['ang_sections'] = []
        for strk in letter_trajs:
            ang = pytkcf.get_continuous_ang(strk)
            curvature, ang_sections = pytkcf.get_ang_indexed_curvature_of_t_indexed_curve(strk)
            tmp_sctn = []
            tmp_crvt = []
            tmp_ang_sctn = []
            idx = 0
            for crvt, sctn in zip(curvature, ang_sections):
                #for each section and corresponding log-curvature profile
                tmp_sctn+=[strk[idx:(idx+len(sctn))]]
                tmp_crvt+=[pytkcf.get_curvature_fft_transform(np.log(crvt))]
                tmp_ang_sctn+=[np.linspace(sctn[0], sctn[-1], len(tmp_crvt[-1]))]
                idx+=len(sctn)

            res_data['sections'] += [tmp_sctn[:]]
            res_data['crvfrq'] += [tmp_crvt[:]]
            res_data['ang_sections']+=[tmp_ang_sctn[:]]
        return res_data

    def crvfrq_drawer(self, ax, data):
        '''
        the ax is the canvas to paint...
        '''
        #plot letter sections in canvas
        letter_trajs = []
        for strk in data['sections']:
            for sctn in strk:
                tmp_letter_traj, = ax.plot(sctn[:, 0], sctn[:, 1], linewidth=2.0)
                letter_trajs+=[tmp_letter_traj]
                ax.hold(True)
        ax.hold(False)
        #frequency analysis
        for strk, strk_ang in zip(data['crvfrq'], data['ang_sections']):
            for crvt_sctn, ang_sctn in zip(strk, strk_ang):
                freq_bins = fftfreq(n=len(ang_sctn), d=ang_sctn[1]-ang_sctn[0])
                #cut to show only low frequency part
                n_freq = len(ang_sctn)/2+1
                #<hyin/Sep-25th-2015> need more investigation to see the meaning of fftfreq
                #some threads on stackoverflow suggests the frequency should be normalized by the length of array
                #but the result seems weird...                
                self.ax_crvfrqplt.plot(np.abs(freq_bins[0:n_freq])*2*np.pi, np.abs(crvt_sctn[0:n_freq]))
                self.ax_crvfrqplt.set_ylabel('Normed Amplitude')
                self.ax_crvfrqplt.set_xlabel('Logarithm of Frequency')
                #cut the xlim
                self.ax_crvfrqplt.set_xlim([0, 8])
                self.ax_crvfrqplt.hold(True)
        self.ax_crvfrqplt.hold(False)
        self.crvfrq_pltcanvas.draw()

        return letter_trajs

def main():
    app = QApplication(sys.argv)
    dp = PyTrajKin_GUI()
    dp.show()
    app.exec_()
    return

if __name__ == '__main__':
    main()