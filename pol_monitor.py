import numpy as np
import glob
import os
import argparse
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm

from pol_lib import load_hist, add_statistics, mask_hist
from pol_plot_lib import init_monitor_figure, init_data_figure

def transform(h_dict):
    h_l = np.where(h_dict['hc_l'] > 1. ,h_dict['hc_l'], 1.)
    h_r = np.where(h_dict['hc_r'] > 1. ,h_dict['hc_r'], 1.)
    norm = (h_l + h_r)/2.
    buf_dict = {'hc_l': np.divide(h_l, norm)-1.,
                'hc_r': np.divide(h_r, norm)-1.,
                'hs_l': h_dict['hs_l'],
                'hs_r': h_dict['hs_r'],
                'xs': h_dict['xs'],
                'ys': h_dict['ys'],
                'xc': h_dict['xc'],
                'yc': h_dict['yc']}
    return buf_dict

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
                           vmin=tmin,
                           vmax=tmax,
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
    
def plot_data3d(h_dict, fig,  ax, h_type='l'):
    h_l = h_dict['hc_l']
    h_r = h_dict['hc_r']

    x = h_dict['xc']
    y = h_dict['yc']
    NL = 1.
    NR = 1.

    #h_l, h_r, x = arrange_region(h_l, h_r ,x ,lim = [-xrange, xrange])
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
    
    
def savetxt(h_dict):
    hc_l = h_dict['hc_l']
    hc_r = h_dict['hc_r']
    a_file = open("hitmap.txt", "w")
    a_file.write('#x\ty\tval\n')
    for y in range(0, np.shape(hc_l)[0]):
        for x in range(0, np.shape(hc_l)[1]):
            a_file.write('{:3.1f}\t{:3.1f}\t{:6d}\n'.format(-31+x*2, -9.5+y, int(hc_l[y,x])))
    #np.savetxt(a_file, hc_l, delimiter=',', newline='\n', fmt='%6d')
    a_file.close()

def monitor(config, regex_line, n_files):
    hist_fpath = config['hist_fpath']
    
    fname_list = np.sort(np.array(glob.glob1(hist_fpath , regex_line))) #get file list
    fig, ax = init_monitor_figure() 
    fig1, ax1 = init_data_figure('transform')
    file_count = 0
    try:
        for fname in fname_list:
            if file_count == 0:
                h_dict = load_hist(hist_fpath, fname)
                buf_dict = h_dict
            else:
                buf_dict = load_hist(hist_fpath, fname)
                h_dict = add_statistics(h_dict, buf_dict)
            file_count += 1
            if file_count == n_files:
                h_dict = mask_hist(config, h_dict)
                #h_dict = transform(h_dict)
                #savetxt(h_dict)
                plot_hitmap(fig, ax, h_dict, block=False, norm=False)
                plot_data3d(h_dict, fig1,  ax1, h_type='l')
                plot_data3d(h_dict, fig1,  ax1, h_type='r')
                plt.draw()
                plt.pause(1)
                input("Press Enter to continue...")
                del buf_dict
                del h_dict
                file_count = 0
    except KeyboardInterrupt:
        print('\n***Exiting monitor program***')
        pass
        
def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_monitor.py")
    parser.add_argument('-N', default=1, help='Number of files to plot on the same figure')
    parser.add_argument('regex_line', nargs='?', help='Name of the file to start online preprocessing in regex format')
    args = parser.parse_args()
   
    with open(os.getcwd()+'/pol_config.yml', 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print('Error opening pol_config.yaml file:')
            print(exc)
        else:
            monitor(config,  args.regex_line, int(args.N)) 
            
if __name__ == '__main__':
    main()
