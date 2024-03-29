import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import scipy.interpolate as spint
import scipy.stats
from math import sqrt, ceil
import math

from pol_lib import arrange_region
from mapping import get_xy

matplotlib.use('TkAgg')
plt.rcParams.update({'errorbar.capsize': 2})
RGI = spint.RegularGridInterpolator

class data_field:
    def __init__(self, coors, data, data_err = None, label=None, data_type='dat'):
        self.x = coors[0]
        self.y = coors[1]
        self.data = data
        self.data_err = data_err
        self.label = label
        self.data_type = data_type
        self.interpolation = 'none'
        self.palette = plt.cm.viridis
        
    #def make_projection(self, axis, his, his_error):
    #    return np.sum( his, axis = axis),  np.sqrt( np.sum(his_error**2, axis=axis) )
        
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
        #ax.legend()

#    def draw_profile(self, ax):
#        x = (self.y[1:]+self.y[:-1])/2
#        y = self.data 
#        ax.bar(x, 
#                   y,
#                   yerr=self.data_err,
#                   color='white',
#                   edgecolor='white',
#                   linewidth=1,
#               #height=1,
#                   zorder=1)
#        if self.data_type == 'dat':
#            ax.plot(x, y, marker="o", markersize=5, linestyle="", alpha=0.8, color="blue",zorder=3, label = self.label)
#        else:
#            ax.plot(x,y, color = 'red', zorder=4, label = self.label)
#        ax.grid(zorder=0)
#        ax.set_xlabel(r'y [mm]')
#        #ax.legend()
    
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
        #ax.legend()
        
    def draw_2d_plot(self, ax):
        im_dat = ax.imshow( self.data, 
                            cmap=self.palette,
                            interpolation=self.interpolation,
                            extent=[self.x[0],self.x[-1],self.y[0],self.y[-1]],
                            origin='lower')
        #cbar_dat = fig.colorbar(im_dat, ax=ax)
        #ax.grid()
        ax.set_aspect(1)
        ax.set_title(self.label)
        ax.set_xlabel(r'x [mm]')
        ax.set_ylabel(r'y [mm]')

    def draw(self, ax):
        if self.y is None:
            self.draw_profilex(ax)
        elif self.x is None: 
            self.draw_profiley(ax)
        else:
            self.draw_2d_plot(ax)

def init_monitor_figure():
    plt.ion()
    fig, ax = plt.subplots(
       
        nrows=3, ncols=1, sharex=False, sharey=False,  
        gridspec_kw={'width_ratios':[1], 'height_ratios':[1,1,1]})
    fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.set_size_inches(8,8)
    return fig, ax

def plot_hist(x, y, h, ax, cax, fig, diff=False):
    ax.set_aspect(1)
    h_f = h.flatten()
    if np.shape(h_f)[0]:
        if diff:
            tmin = np.quantile(h_f, 0.05)
            tmax = np.quantile(h_f, 0.95)
            tmed = 0
        else:
            tmin = np.min(h_f)
            tmax = np.max(h_f)
            tmed = np.quantile(h_f, 0.5)
        im = ax.pcolormesh(x,y,h,
                           vmin=tmin,
                           vmax=tmax,
                           cmap=plt.cm.gray)
        #cbar = fig.colorbar(im, cax=cax,format=ticker.FuncFormatter(fmt))
        #cax.set_aspect(15)
        #cbar.set_ticks([tmin,tmed, tmax])
    else:
        im = ax.pcolormesh(x,y,h, cmap=plt.cm.gray)
        #cax.set_aspect(15)
        #cbar.set_ticks([0])
        
def plot_hitmap(fig, ax, h_dict, fname, block=False, norm=False):
    hc_l = h_dict['hc_l']
    hc_r = h_dict['hc_r']
    hs_l = h_dict['hs_l']
    hs_r = h_dict['hs_r']

    xxs, yys = np.meshgrid(h_dict['xs'], h_dict['ys'])
    xxc, yyc = np.meshgrid(h_dict['xc'], h_dict['yc'])
    (ax1, ax2, ax3) = ax
    ratio = 15
    for i in range(1,3):
        exec('ax{:d}.clear()'.format(i))
    ax1.set_title(fname[:19])
    ax1.set_aspect(1)
    plot_hist(xxc, yyc, hc_l, ax1, None, fig)

    ax2.set_title('Right')
    ax2.set_aspect(1)
    plot_hist(xxc, yyc, hc_r, ax2, None, fig)

    ax3.set_title('L-R difference')
    ax3.set_aspect(1)
    if norm:
        hs_diff = hs_l/np.sum(np.sum(hs_l))-hs_r/np.sum(np.sum(hs_r))
        hc_diff = hc_l/np.sum(np.sum(hc_l))-hc_r/np.sum(np.sum(hc_r))
    else:
        hs_diff = hs_l-hs_r
        hc_diff = hc_l-hc_r
    plot_hist(xxc, yyc, hc_diff, ax3, None, fig, diff=True)

def draw_ch_numbers(ax, config):
    for idx in range(0,1280):
                x,y = get_xy(idx, config["zone_id"])
                if (x >=0):
                    tx="{}".format((idx+640)%1280) # create a label
                    ax.text(2*x+1-32,y-10,
                            tx,
                            ha="center",
                            color="magenta",
                            va="bottom",
                            fontsize=6)
   
def init_figure(label): 
        fig = plt.figure(figsize=(16, 8))
        fig.canvas.set_window_title(label)
        fig.set_tight_layout(True)
        fig.tight_layout(rect=[0, 0, 1, 1])
        #gs0 = gridspec.GridSpec(1, 2, figure=fig)
        gs0 = gridspec.GridSpec(1, 3, figure=fig, width_ratios = [1,2,2])

        ax00 = fig.add_subplot(gs0[0]) #0

        
        #gs00 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs0[0], height_ratios = [1,1,1])
        #ax01 = fig.add_subplot(gs00[:-1,:-1])
        #ax02 = fig.add_subplot(gs00[0,-1])
        #ax03 = fig.add_subplot(gs00[1,-1])
        #ax04 = fig.add_subplot(gs00[-1, :])
        
        gs00 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs0[1], height_ratios = [2,1,1])
        ax01 = fig.add_subplot(gs00[0,0]) #1
        ax02 = fig.add_subplot(gs00[0,1]) #2
        ax03 = fig.add_subplot(gs00[1, :]) #3
        ax04 = fig.add_subplot(gs00[2, :]) #4

        gs10 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs0[2], height_ratios = [2,1,1])
        ax11 = fig.add_subplot(gs10[0,0]) #5
        ax12 = fig.add_subplot(gs10[0,1]) #6
        ax13 = fig.add_subplot(gs10[1, :]) #7
        ax14 = fig.add_subplot(gs10[2, :]) #8

        ax = [ax00, ax01,ax02, ax03, ax04, ax11, ax12, ax13, ax14]

        #gs20 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[3], height_ratios = [1,1,1,1])
        #ax21 = fig.add_subplot(gs20[0,:]) #9
        #ax22 = fig.add_subplot(gs20[1,:]) #10
        #ax23 = fig.add_subplot(gs20[2,:]) #11
        #ax24 = fig.add_subplot(gs20[3,:]) #12

        #ax = ax + [ax21,ax22,ax23,ax24]

        return fig, ax
        
def init_figure_gen(label, data_fields): 
    fig = plt.figure(figsize=(16, 18))
    fig.canvas.set_window_title(label)
    fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    
    other = [s for s in data_fields.keys() if not (('fit' in s) or ('data' in s))]
    n_other = len(other)
    n_2d_plots = 0
    for key in data_fields.keys():
        if len(np.shape(data_fields[key].data)) == 2: n_2d_plots +=1
    n_plots = ceil((len(data_fields) + n_2d_plots + n_other)/2) 
    n_x = ceil(sqrt(n_plots))
    n_y = ceil(n_plots/n_x)
    
    gs_cols = gridspec.GridSpec(1,n_x, figure=fig)
    ax = []
    for idx in range(n_x):
        gs_row = gridspec.GridSpecFromSubplotSpec(n_y, 1, subplot_spec=gs_cols[idx], height_ratios = np.ones(n_y))
        for idy in range(n_y):
            ax.append(fig.add_subplot(gs_row[idy,:]))
        if len(ax) >= n_plots:
            break
    return fig, ax
        
def print_fit_results(ax, fitter, begintime, endtime):
    minuit = fitter.minuit
    ax.set_title('Fit results (fit method {})'.format(fitter.fit_method))
    ax.set(xlim=(0., 1.25), ylim=(-0.25, 1.25), xticks=[], yticks=[])
    ax.axis('off')
    important_pars = ['P', 'NL', 'NR']
    line_size = 0.07
    lc = 0 
    x = -0.2
    y = 1.1
    #ax.text(x, y+lc, time[:19], size=14, ha='left', va='center', color='black')
    ax.text(x, y+lc, "{:<6} {:21}".format("begin:", begintime), size=14, ha='left', va='center', color='black')
    lc -= line_size
    ax.text(x, y+lc, "{:<6} {:21}".format("end:", endtime), size=14, ha='left', va='center', color='black')
    lc -= line_size
    chi2ndf = r'$\chi^2/ndf = {0:.{1}f}/{2} = {3:.{4}f}$'.format(fitter.chi2, 0 if fitter.chi2 > 10 else 2, fitter.ndf, fitter.chi2/fitter.ndf, 1 if fitter.chi2/fitter.ndf>10 else 2 )
    ax.text(x, y+lc, chi2ndf, size=14, ha='left', va='center', color='black')
    lc -= line_size
    prob = (1.0-scipy.stats.chi2.cdf(fitter.chi2, fitter.ndf))
    prob_txt = r'$prob(\chi^2) = {:.4f}%$'.format(prob) if prob>0.01 else  r'$prob(\chi^2) = {:.2E}%$'.format(prob)
    ax.text(x, y+lc, prob_txt, size=14, ha='left', va='center', color='black')
    lc -= line_size
    for parname in fitter.parlist:
        try:
            if parname=='L':
                    text = r'${:s} = {:1.2f} \pm {:1.2f}$ m'.format(parname, minuit.values[parname]*1e-3, minuit.errors[parname]*1e-3)
                    ax.text(x, y+lc, text, size=14, ha='left', va='center', color='black')
                    lc -= line_size
            else:
                if not minuit.fixed[parname]:
                    if  parname == 'beta':
                        text = r'$\beta = {:.2f} \pm {:.2f}^\circ$'.format(minuit.values[parname]*180./math.pi, minuit.errors[parname]*180/math.pi)
                    else:
                        text = r'${:s} = {:1.3f} \pm {:1.3f}$'.format(parname, minuit.values[parname], minuit.errors[parname])
                    if parname in important_pars:
                        color = 'red'
                    else:
                        color = 'black'
                    ax.text(x, y+lc, text, size=14, ha='left', va='center', color=color)
                    lc -= line_size
        except KeyError: pass
    return ax
