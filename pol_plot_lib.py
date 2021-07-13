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

matplotlib.use('TkAgg')

plt.rcParams.update({'errorbar.capsize': 2})
RGI = spint.RegularGridInterpolator

def init_monitor_figure():
    plt.ion()
    fig, ax = plt.subplots(
        nrows=3, ncols=3, sharex=False, sharey=False, 
        gridspec_kw={'width_ratios':[1,0.07,0.07], 'height_ratios':[1,1,1]}
        )
    fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.set_size_inches(8,6)
    return fig, ax

def init_fit_figure(label, title):
    fig = plt.figure(figsize=(7, 5))
    fig.canvas.set_window_title(title)
    fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.suptitle(label)
    # Define the positions of the subplots.
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[ 1., 1.],
                           height_ratios=[1., 1.])
    ax_py = plt.subplot(gs[0])
    ax_dat = plt.subplot(gs[1])
    ax_fit = plt.subplot(gs[3])
    ax_px = plt.subplot(gs[2])
    ax = [ax_py,ax_dat, ax_fit, ax_px]

    return fig, ax

def init_data_figure(label):
    fig = plt.figure(figsize=(10, 5))
    fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.suptitle(label)
    # Define the positions of the subplots.
    gs = gridspec.GridSpec(1, 2, width_ratios=[ 1., 1.])
    ax_l = fig.add_subplot(gs[0], projection='3d')
    ax_r = fig.add_subplot(gs[1], projection='3d')
    #ax_diff = fig.add_subplot(gs[2], projection='3d')
    ax_l.set_title('Left')
    ax_r.set_title('Right')
    ax = [ax_l,ax_r]
    return fig, ax

def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)

def plot_hist(x, y, h, ax, cax, fig, diff=False):
    ax.set_aspect(1)
    h_f = h.flatten()
    if np.shape(h_f)[0]:
        if diff:
            tmin = np.quantile(h_f, 0.05)
            tmax = np.quantile(h_f, 0.95)
            tmed = 0
        else:
            tmin = np.quantile(h_f, 0.01)
            tmax = np.quantile(h_f, 0.99)
            tmed = np.quantile(h_f, 0.5)
        im = ax.pcolormesh(x,y,h,
                           vmin=tmin,#np.min(h_f),
                           vmax=tmax,#np.max(h_f),
                           cmap=plt.cm.gist_gray)
        cbar = fig.colorbar(im, cax=cax,format=ticker.FuncFormatter(fmt))
        cax.set_aspect(15)
        cbar.set_ticks([tmin,tmed, tmax])
    else:
        im = ax.pcolormesh(x,y,h, cmap=plt.cm.gist_gray)
        cax.set_aspect(15)
        cbar.set_ticks([0])


def plot_hitmap(fig, ax, h_dict, block=False, norm=False):
    hc_l = h_dict['hc_l']
    hc_r = h_dict['hc_r']
    hs_l = h_dict['hs_l']
    hs_r = h_dict['hs_r']

    xxs, yys = np.meshgrid(h_dict['xs'], h_dict['ys'])
    xxc, yyc = np.meshgrid(h_dict['xc'], h_dict['yc'])
    ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = ax
    ratio = 15
    for i in range(1,10):
        exec('ax{:d}.clear()'.format(i))
    ax1.set_title('Left')

    #plot_hist(xxs, yys, hs_l, ax1, ax2, fig)
    #xx_new, yy_new, hc_l_new = get_interp_hist(h_dict['xc'], h_dict['yc'], hc_l, step_ratio=5)
    plot_hist(xxc, yyc, hc_l, ax1, ax3, fig)

    ax4.set_title('Right')
    #plot_hist(xxs, yys, hs_r, ax4, ax5, fig)
    plot_hist(xxc, yyc, hc_r, ax4, ax6, fig)

    ax7.set_title('L-R difference')
    ax7.set_aspect(1)
    if norm:
        hs_diff = hs_l/np.sum(np.sum(hs_l))-hs_r/np.sum(np.sum(hs_r))
        hc_diff = hc_l/np.sum(np.sum(hc_l))-hc_r/np.sum(np.sum(hc_r))
    else:
        hs_diff = hs_l-hs_r
        hc_diff = hc_l-hc_r
    #plot_hist(xxs, yys, hs_diff, ax7, ax8, fig)
    plot_hist(xxc, yyc, hc_diff, ax7, ax9, fig, diff=True)
    

def plot_data3d(h_dict, m_fitres, xrange, fig,  ax, h_type='l'):
    fit_pars = np.array(m_fitres.values)
    h_l = h_dict['hc_l']
    h_r = h_dict['hc_r']

    x = h_dict['xc']
    y = h_dict['yc']
    NL = m_fitres.values['NL']
    NR = m_fitres.values['NR']

    h_l, h_r, x = arrange_region(h_l, h_r ,x ,lim = xrange)
    xx, yy = np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2)
    ax_dat = None
    ax_l, ax_r = ax
    if h_type == 'l':
        h_data = h_l
        ax_dat = ax_l
    elif h_type == 'r':
        h_data = h_r
        ax_dat = ax_r
    elif h_type == 'diff_l':
        fit_l = np.where(h_l > 0, get_fit_func(x, y, fit_pars), 0)
        h_data = h_l/NL - fit_l
        ax_dat = ax_l
    elif h_type == 'diff_r':
        fit_r = np.where(h_r > 0, get_fit_func(x, y, fit_pars, inverse_pol=True), 0)
        h_data = h_r/NR - fit_r
        ax_dat = ax_r
    else:
        pass

    ax_dat.set_xlabel(r'x [mm]')
    ax_dat.set_ylabel(r'y [mm]')
    xpos, ypos = np.meshgrid(x[:-1]+x[1:], y[:-1]+y[1:])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like(xpos)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    '''Draw histogram'''
    dz = h_data.flatten()
    cmap = cm.get_cmap('jet')	# Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    ax_dat.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')

def plot_fit(h_dict, m_fitres, xrange, fig, ax, diff=True, pol='l'):
    fit_pars = np.array(m_fitres.values)
    NL = m_fitres.values['NL']
    NR = m_fitres.values['NR']

    h_l = h_dict['hc_l']/NL
    h_r = h_dict['hc_r']/NR
    x = h_dict['xc']
    y = h_dict['yc']
    #print(h_l)
    xx, yy = np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2)
    h_l, h_r, x = arrange_region(h_l, h_r ,x ,lim = xrange)

    for axs in ax:
        axs.cla()

    ax_py, ax_dat, ax_fit, ax_px = ax

    n_evt_l = np.sum(np.sum(h_l))
    n_evt_r = np.sum(np.sum(h_r))

    h_lx, h_ly = h_l.sum(axis=0), h_l.sum(axis=1)
    h_rx, h_ry = h_r.sum(axis=0), h_r.sum(axis=1)

    fit_l = np.where(h_l>0, get_fit_func(x, y, fit_pars),0)
    fit_r = np.where(h_r>0, get_fit_func(x, y, fit_pars, inverse_pol=True), 0)

    fit_lx, fit_ly = fit_l.sum(axis=0), fit_l.sum(axis=1)
    fit_rx, fit_ry = fit_r.sum(axis=0), fit_r.sum(axis=1)
    
    if diff:
        h_data = h_l - h_r
        h_fit = fit_l-fit_r 
        h_data_x = h_lx-h_rx
        h_fit_x = fit_lx-fit_rx
        h_data_y = h_ly-h_ry
        h_fit_y = fit_ly-fit_ry

    elif pol == 'r' :
        h_data = h_r
        h_fit = fit_r - h_r 
        h_data_x = h_rx
        h_fit_x = fit_rx
        h_data_y = h_ry
        h_fit_y = fit_ry
    elif pol == 'l' :
        h_data = h_l
        h_fit = fit_l - h_l
        h_data_x = h_lx
        h_fit_x = fit_lx
        h_data_y = h_ly
        h_fit_y = fit_ly
    else:
        pass

    '''Draw 2d data histogram (top right)'''

    im_dat = ax_dat.imshow(h_data, 
                           cmap=plt.cm.viridis,
                           interpolation='none',
                           extent=[x[0],x[-1],y[0],y[-1]],
                           origin='lower')#
    #cbar_dat = fig.colorbar(im_dat, ax=ax_dat)
    ax_dat.grid()
    ax_dat.set_aspect(1)
    ax_dat.set_title('Data')
    ax_dat.set_xlabel(r'x [mm]')

    '''Draw  Y-profile histogram (top left)'''

    h_yerr =  np.sqrt(h_ly/NL+h_ry/NR)
    ax_py.barh((y[1:]+y[:-1])/2, 
               h_data_y,
               xerr=h_yerr,
               color='white',#color='lightseagreen',
               edgecolor='white',
               linewidth=1,
               height=1,
               zorder=1)
    ax_py.plot(h_data_y,(y[1:]+y[:-1])/2,  marker="o", markersize=5, linestyle="", alpha=0.8, color="blue",zorder=3, label = 'Data')

    ax_py.plot(h_fit_y, (y[1:]+y[:-1])/2, color = 'red', zorder=4, label = 'Fit')
    ax_py.set_ylabel(r'y [mm]')
    ax_py.set_ylim(ax_dat.get_ylim())
    ax_py.invert_xaxis()
    ax_py.grid(zorder=0)
    ax_py.set_title('Y-profile')
    ax_py.legend()
    '''Draw  X-profile histogram (bottom left)'''
    h_xerr =  np.sqrt(h_lx/NL+h_rx/NR)
    ax_px.bar((x[1:]+x[:-1])/2, h_data_x,
              yerr=h_xerr,
              color='white',#color='lightseagreen',
              edgecolor='white',
              linewidth=1,
              width=2,
              zorder=1)
    ax_px.plot((x[1:]+x[:-1])/2, h_data_x, marker="o", markersize=5, linestyle="", alpha=0.8, color="blue",zorder=3, label = 'Data')

    ax_px.plot((x[1:]+x[:-1])/2, h_fit_x, color = 'red', zorder=4, label = 'Fit')
    ax_px.set_xlim(ax_dat.get_xlim())
    ax_px.grid(zorder=0)
    ax_px.set_title('X-profile')
    ax_px.set_xlabel(r'x [mm]')
    ax_px.legend()
    '''Draw  2d Fit histogram (bottom right)'''
    im_fit = ax_fit.imshow(h_fit, 
                           cmap=plt.cm.viridis,
                           interpolation='none',
                           extent=[x[0],x[-1],y[0],y[-1]],
                           origin='lower')#
    #cbar_fit = fig.colorbar(im_fit, ax=ax_fit)
    ax_fit.grid()
    ax_fit.set_aspect(1)
    ax_fit.set_title('Fit')
    
#Well, I don't use it. Please check that it works ok.
def get_interp_hist(x,y,h,step_ratio=2):
    x_new = np.linspace(x[0], x[-1], num = np.shape(x)[0]*step_ratio)
    y_new = np.linspace(y[0], y[-1], num = np.shape(y)[0]*step_ratio)
    x_mid = (x[1:] + x[:-1])/2
    y_mid = (y[1:] + y[:-1])/2
    xx, yy = np.meshgrid(x_mid, y_mid)
    intrep_f = RGI((y_mid, x_mid), values=h)

    x_mid = (x_new[1:] + x_new[:-1])/2
    y_mid = (y_new[1:] + y_new[:-1])/2
    xx_mid, yy_mid = np.meshgrid(x_mid, y_mid)
    pts = np.array([])
    h_new = intrep_f(y_mid, x_mid)
    print(np.shape(h_new),np.shape(xx_new))
    return xx_new, yy_new, h_new.T
