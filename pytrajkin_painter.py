from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 


class PyTrajKin_Painter(QWidget):

    def __init__(self, parent=None, analyzer=None, analyzer_drawfunc=None):
        QWidget.__init__(self, parent=parent)

        #analyzer is a function handler that accepts an array of trajectories for analyzing
        #and return another set of trajectories, there is also a draw function handler which passes the canvas for drawing
        #it will not be called if it is not defined
        self.analyzer = analyzer
        self.analyzer_drawfunc = analyzer_drawfunc

        #prepare data and figure
        self.traj_pnts = []
        self.curr_traj = None
        self.lines = []
        self.curr_line = None
        
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax_painter = self.fig.add_subplot(111)
        self.ax_painter.hold(True)
        self.ax_painter.set_xlim([-2, 2])
        self.ax_painter.set_ylim([-2, 2])
        self.ax_painter.set_aspect('equal')

        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.addWidget(self.canvas, 5)

        self.ctrl_pnl_layout = QVBoxLayout()
        #a button to clear the figure
        self.clean_btn = QPushButton('Clear')
        self.ctrl_pnl_layout.addWidget(self.clean_btn)

        #a button for analyzing written
        self.analyze_btn = QPushButton('Evaluate')
        self.ctrl_pnl_layout.addWidget(self.analyze_btn)

        self.hbox_layout.addLayout(self.ctrl_pnl_layout, 1)

        self.setLayout(self.hbox_layout)
        self.drawing = False

        self.create_event_handler()
        return

    def create_event_handler(self):
        self.canvas_button_clicked_cid = self.canvas.mpl_connect('button_press_event', self.on_canvas_mouse_clicked)
        self.canvas_button_released_cid = self.canvas.mpl_connect('button_release_event', self.on_canvas_mouse_released)
        self.canvas_motion_notify_cid = self.canvas.mpl_connect('motion_notify_event', self.on_canvas_mouse_move)
        
        self.clean_btn.clicked.connect(self.on_clean_button_clicked)
        self.analyze_btn.clicked.connect(self.on_analyze_button_clicked)
        return

    def on_canvas_mouse_clicked(self, event):
        # print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        # event.button, event.x, event.y, event.xdata, event.ydata)
        self.drawing = True
        # create a new line if we are drawing within the area
        if event.xdata is not None and event.ydata is not None and self.curr_line is None and self.curr_traj is None:
            self.curr_line, = self.ax_painter.plot([event.xdata], [event.ydata], '-b', linewidth=3.0)
            self.curr_traj = [np.array([event.xdata, event.ydata])]
            self.canvas.draw()
        return

    def on_canvas_mouse_released(self, event):
        # print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        # event.button, event.x, event.y, event.xdata, event.ydata)
        self.drawing = False
        # store finished line and trajectory
        # print self.curr_traj
        self.lines.append(self.curr_line)
        self.traj_pnts.append(self.curr_traj)
        self.curr_traj = None
        self.curr_line = None
        return

    def on_clean_button_clicked(self, event):
        print 'clean the canvas...'
        #clear everything
        for line in self.lines:
            self.ax_painter.lines.remove(line)
        self.lines = []

        if self.curr_line is not None:
            self.ax_painter.lines.remove(self.curr_line)
        self.curr_line = None
        self.canvas.draw()

        self.traj_pnts = []
        self.curr_traj = None
        self.drawing = False
        return

    def on_canvas_mouse_move(self, event):
        if self.drawing:
            # print 'In movement: x=',event.x ,', y=', event.y,', xdata=',event.xdata,', ydata=', event.ydata
            if event.xdata is not None and event.ydata is not None and self.curr_line is not None and self.curr_traj is not None:
                #append new data and update drawing
                self.curr_traj.append(np.array([event.xdata, event.ydata]))
                tmp_curr_data = np.array(self.curr_traj)
                self.curr_line.set_xdata(tmp_curr_data[:, 0])
                self.curr_line.set_ydata(tmp_curr_data[:, 1])
                self.canvas.draw()
        return

    def plot_trajs_helper(self, trajs):
        tmp_lines = []
        for traj in trajs:
            tmp_line, = self.ax_painter.plot(traj[:, 0], traj[:, 1], '-.g', linewidth=3.0)
            tmp_lines.append(tmp_line)
            self.canvas.draw()
        #add these tmp_lines to lines record
        self.lines = self.lines + tmp_lines
        return

    def on_analyze_button_clicked(self, event):
        if self.analyzer is not None:
            processed_trajs = self.analyzer(self.get_traj_data())
            if self.analyzer_drawfunc is not None:
                drawlines = self.analyzer_drawfunc(self.ax_painter, processed_trajs)
                self.canvas.draw()
                if len(drawlines) > 0:
                    self.lines = self.lines + drawlines
            #plot processed_trajs
            # self.plot_trajs_helper(processed_trajs)
        else:
            print 'Analyzing function is not defined!'
        return

    def get_traj_data(self):
        return self.traj_pnts

    