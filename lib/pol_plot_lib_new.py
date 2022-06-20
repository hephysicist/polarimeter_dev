import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import scipy.interpolate as spint

from pol_lib import arrange_region
from pol_fit_lib import get_fit_func
from mapping import get_xy



class data_field:
    def __init__(self, coors, data, data_err = None, label=None, data_type='dat'):
        self.x = coors[0]
        self.y = coors[1]
        self.data = data
        self.data_err = data_err
        self.label = label
        self.data_type = data_type
        
        
    def draw_profilex(self, ax):
        x = (self.x[1:]+self.x[:-1])/2
        y = self.data
        ax.bar(x, y,
                  yerr=self.data_err,
                  color='white',#color='lightseagreen',
                  edgecolor='white',
                  linewidth=1,
                  width=2,
                  zorder=1)
        if self.data_type == 'dat':
            ax.plot(x, y, marker="o", markersize=5, linestyle="", alpha=0.8, color="blue",zorder=3, label = self.label)
        else:
            ax.plot(x, y, color = 'red', zorder=4, label = self.label)
        ax.grid(zorder=0)
        ax.set_xlabel(r'x [mm]')
        ax.legend()
    
    def draw_profiley(self, ax):
        y = (self.y[1:]+self.y[:-1])/2
        x = self.data #We need to change x <-> y because of using barh
        ax.barh(y, 
                   x,
                   xerr=self.data_err,
                   color='white',
                   edgecolor='white',
                   linewidth=1,
                   height=1,
                   zorder=1)
        if self.data_type == 'dat':
            ax.plot(x, y, marker="o", markersize=5, linestyle="", alpha=0.8, color="blue",zorder=3, label = self.label)
        else:
            ax.plot(x,y, color = 'red', zorder=4, label = self.label)
        ax.grid(zorder=0)
        ax.set_ylabel(r'y [mm]')
        ax.legend()
        
    def draw_2d_plot(self, ax):
        im_dat = ax.imshow( self.data, 
                            cmap=plt.cm.viridis,
                            interpolation='none',
                            extent=[self.x[0],self.x[-1],self.y[0],self.y[-1]],
                            origin='lower')
        #cbar_dat = fig.colorbar(im_dat, ax=ax)
        #ax.grid()
        ax.set_aspect(1)
        ax.set_title(self.label)
        ax.set_xlabel(r'x [mm]')
        ax.set_ylabel(r'y [mm]')

    def draw(self, ax):
        print(self.x, self.y)
        if self.y is None:
            self.draw_profilex(ax)
        elif self.x is None: 
            self.draw_profiley(ax)
        else:
            self.draw_2d_plot(ax)
         
         
matplotlib.use('TkAgg')
plt.rcParams.update({'errorbar.capsize': 2})
RGI = spint.RegularGridInterpolator

def init_monitor_figure():
    plt.ion()
    fig, ax = plt.subplots(
        #nrows=3, ncols=3, sharex=False, sharey=False,
        nrows=3, ncols=1, sharex=False, sharey=False,  
        #gridspec_kw={'width_ratios':[1,0.07,0.07], 'height_ratios':[1,1,1]}
        gridspec_kw={'width_ratios':[1], 'height_ratios':[1,1,1]})
    fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.set_size_inches(8,8)
    return fig, ax
   
def init_figure(label): 
        fig = plt.figure(figsize=(15, 8))
        fig.set_tight_layout(True)
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.suptitle(label)
        gs0 = gridspec.GridSpec(1, 2, figure=fig)
        
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0], height_ratios = [2,1])
        ax01 = fig.add_subplot(gs00[0,0])
        ax02 = fig.add_subplot(gs00[0,1])
        ax03 = fig.add_subplot(gs00[1, :])
        
        gs10 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[1], height_ratios = [2,1])
        ax11 = fig.add_subplot(gs10[0,0])
        ax12 = fig.add_subplot(gs10[0,1])
        ax13 = fig.add_subplot(gs10[1, :])
        ax = [ax01,ax02, ax03, ax11, ax12, ax13]
        return fig, ax
        
def print_fit_results(ax, fitres):
    ax.set_title('Fit results')
    ax.set(xlim=(0., 1.25), ylim=(-0.25, 1.25), xticks=[], yticks=[])
    ax.axis('off')
    par_list = ['P','Q','mx', 'my','sx', 'sy', 'NL', 'NR']
    important_pars = ['P', 'NL', 'NR']
    line_size = 0.13
    lc = 0 
    x = 0.
    y = 1.
    for par in par_list:
        text = r'${:s} = {:1.2f} \pm {:1.2f}$'.format(par, fitres.values[par], fitres.errors[par])
        if par in important_pars:
            color = 'red'
        else:
            color = 'black'
        ax.text(x, y+lc, text, size=14, ha='left', va='center', color=color)
        lc -= line_size
    return ax
