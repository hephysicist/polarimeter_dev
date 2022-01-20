#!/usr/bin/env python3 
from iminuit import Minuit

import numpy as np
import glob
import time
from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import yaml

from lib.pol_lib import *
from lib.pol_fit_lib import *
from lib.pol_plot_lib import *
from lib.moments import get_moments

def transform(h_dict):
    h_l = h_dict['hc_l']
    h_r = h_dict['hc_r']
    norm = 1
    norm = (h_l + h_r)/2.
    norm = np.where(norm > 1., norm, 1.) 
    buf_dict = {'hc_l': h_lnorm,
                'hc_r': h_r - norm,
                'hs_l': h_dict['hs_l'],
                'hs_r': h_dict['hs_r'],
                'xs': h_dict['xs'],
                'ys': h_dict['ys'],
                'xc': h_dict['xc'],
                'yc': h_dict['yc']}
    return buf_dict

def make_fit(config, h_dict, vepp4E):
    h_l = h_dict['hc_l']
    h_r = h_dict['hc_r']
    x = h_dict['xc']
    y = h_dict['yc']
    xrange = config['xrange']

    h_l, h_r, x = arrange_region(h_l, h_r ,x ,lim = xrange)

    n_evt_l = np.sum(np.sum(h_l))
    n_evt_r = np.sum(np.sum(h_r))

    hprof_xl = np.sum(h_l, axis=0)
    hprof_yl = np.sum(h_l, axis=1)
    hprof_xr = np.sum(h_r, axis=0)
    hprof_yr = np.sum(h_r, axis=1)

    mx_l = get_mean(x,hprof_xl)
    mx_r = get_mean(x,hprof_xr)
    my_l = get_mean(y,hprof_yl)
    my_r = get_mean(y,hprof_yr)


    x_mid = (x[1:] + x[:-1])/2
    y_mid = (y[1:] + y[:-1])/2

    ndf = np.shape(x_mid)[0]*np.shape(y_mid)[0]

    X = [x_mid,y_mid]
    chi2_2d = Chi2(get_fit_func_, X, h_l, h_r)
    initial_values = config['initial_values']
    initial_values['E'] = vepp4E
    #print('Energy is set to: {:4.2f}'.format(initial_values['E']))
    fix_par = config['fix_par']
    par_err = config['par_err']
    par_lim = config['par_lim']
    
    m2d = Minuit(chi2_2d, **initial_values)
    for p_key in initial_values.keys():
        m2d.fixed[p_key]  = fix_par[p_key]
        m2d.errors[p_key] = par_err[p_key]
        m2d.limits[p_key] = par_lim[p_key]
    m2d.print_level = 0
    m2d.errordef=1
    begin_time = time.time()
    m2d.migrad()
    m2d.hesse()
    print(m2d)
    if not m2d.valid:
       for name in  ['sx', 'sy', 'alpha_x1', 'alpha_x2', 'alpha_y1', 'alpha_y2', 'nx1','nx2', 'ny1','ny2', 'phi', 'p1', 'p2', 'p3']:
            m2d.fixed[name]=True
       m2d.migrad()
       m2d.hesse()
       #print(m2d)
    end_time = time.time()
    #print("Fit time: ", end_time-begin_time, " s")
    return m2d, ndf

def init_figures():
    fig_l, ax_l = init_fit_figure(label = 'L', title='Left')
    fig_r, ax_r = init_fit_figure(label = 'R', title='Right')
    fig_d, ax_d = init_fit_figure(label = 'Diff', title='Diff')
    fig_3d, ax_3d = init_data_figure(label = '3d')
    return [fig_l, fig_r, fig_d, fig_3d], [ax_l, ax_r, ax_d, ax_3d]

def show_res(config, h_dict, fitres, Fig, Ax):
    xrange = config['xrange']
    plot_fit(h_dict, fitres, xrange, Fig[0], Ax[0], diff=False, pol='l')
    plot_fit(h_dict, fitres, xrange, Fig[1], Ax[1], diff=False, pol='r')
    plot_fit(h_dict, fitres, xrange, Fig[2], Ax[2], diff=True)
    plot_data3d(h_dict, fitres, xrange, Fig[3],  Ax[3], h_type='fl')
    plot_data3d(h_dict, fitres, xrange, Fig[3],  Ax[3], h_type='fr')
    plt.show(block=False)
    plt.pause(1)

def accum_data_and_make_fit(config, regex_line, offline = False):
    hist_fpath = config['hist_fpath']
    n_files = config['n_files']
    file_arr = np.array(glob.glob1(hist_fpath, regex_line))
    if np.shape(file_arr)[0]:
        file_arr = np.sort(file_arr)
        if not offline:
            fname = file_arr[-2] #get 2nd last file
        else:
            fname = file_arr[0] 
    fname_prev = ''
    file_count = 0
    files4point_count = 0
    attempt_count = 0
    fig_arr, ax_arr = init_figures()
    fit_counter = 0
    try:
        while (file_count < np.shape(file_arr)[0] and offline) or (not offline):
            if fname_prev != fname:
                if files4point_count == 0: 
                    h_dict = load_hist(hist_fpath,fname)
                    vepp4E = h_dict['vepp4E']
                    buf_list = get_par_from_file('/mnt/vepp4/kadrs/nmr.dat', par_id_arr = [1])
                    vepp4E_nmr = float(buf_list[0])
                    first_fname = fname
                    buf_dict = h_dict
                    print('Enmr: ', vepp4E_nmr)
                    print('Dep freq: ', buf_dict['dfreq'])
                    print('Evt l: ',sum(sum(h_dict['hc_l'])), 'Evt r: ', sum(sum(h_dict['hc_r'])))
                    fname_prev = fname
                    
                else:
                    buf_dict = load_hist(hist_fpath,fname)
                    h_dict = accum_data(h_dict, buf_dict)
                    print('Evt l: ',sum(sum(h_dict['hc_l'])), 'Evt r: ', sum(sum(h_dict['hc_r'])))
                    print('Dep freq: ', buf_dict['dfreq'])
                files4point_count += 1
                file_count += 1
                attempt_count = 0
                fname_prev = fname
                print('Progress: ', int(files4point_count), '/', int(n_files))
            else:
                time.sleep(1)
                attempt_count +=1
                print('Waiting for the new file: {:3d} seconds passed'.format(attempt_count), end= '\r')
            if files4point_count == n_files:
                h_dict = mask_hist(config, h_dict)
                if config['need_blur']:
                    h_dict = eval(config['blur_type']+'(h_dict)')
                fitres, ndf = make_fit(config, h_dict, vepp4E)
                raw_stats = get_raw_stats(h_dict)
                print_stats(raw_stats)
                print_pol_stats_nik(fitres)
                moments=get_moments(h_dict)
                calc_asymmetry(h_dict)
                show_res(config, h_dict, fitres, fig_arr, ax_arr)
                chi2_normed = fitres.fval / (ndf - fitres.npar)
                print('chi2/ndf', chi2_normed)
                par_list = [h_dict['dfreq'], vepp4E_nmr]
                write2file_nik( config['fitres_file'],
                            first_fname,
                            fitres,
                            par_list,
                            raw_stats,
                            fit_counter,
                            moments,
                            chi2_normed)
                del buf_dict
                del h_dict
                fit_counter += 1
                files4point_count = 0
            file_arr = np.array(glob.glob1(hist_fpath, regex_line))
            if not offline:
                fname = file_arr[-2] #get 2nd last file
            else:
                fname = file_arr[file_count] 
           

    except KeyboardInterrupt:
        print('\n***Exiting fit program***')
        pass

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_fit.py")
    parser.add_argument('--offline', help='Use this key to fit iteratively all, starting from regrex_line', default=False, action="store_true")
    parser.add_argument('regex_line', nargs='?', help='Name of the file to start offline  fit in regex format')
    parser.add_argument('-i', help='Interactive mode', action='store_true')
    parser.add_argument('--noblur', help='Disable blur', action='store_true')
    parser.add_argument('--blur', help='Blur algorithm', type=str, default='default')
    parser.add_argument('--plot3d', help='Plot 3D left right', action='store_true')
    
    args = parser.parse_args()
    global INTERACTIVE_MODE
    INTERACTIVE_MODE = args.i
    print("Interactive mode ", INTERACTIVE_MODE)
    with open(os.getcwd()+'/pol_fit_config.yml', 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
            config['need_blur'] = config['need_blur'] and not args.noblur
            config['blur']= args.blur
            print("blur = ", args.blur)
            config['plot3d']=args.plot3d
        except yaml.YAMLError as exc:
            print('Error opening pol_config.yaml file:')
            print(exc)
        else:
            if args.regex_line:
                regex_line = str(args.regex_line)
            else:
                regex_line = str(config['regex_line'])
            accum_data_and_make_fit(config, regex_line, args.offline) 
if __name__ == '__main__':
    main()
