from iminuit import Minuit

import numpy as np
import glob
import time
from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import yaml

from pol_lib import *
from pol_fit_lib import *
from pol_plot_lib import *
from moments import get_moments

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

# def make_fit(h_dict):
#     h_l = h_dict['hc_l']
#     h_r = h_dict['hc_r']
#     x = h_dict['xc']
#     y = h_dict['yc']

#     h_l, h_r, x = arrange_region(h_l, h_r ,x ,lim = [-XRANGE,XRANGE])

#     n_evt_l = np.sum(np.sum(h_l))
#     n_evt_r = np.sum(np.sum(h_r))

#     hprof_xl = np.sum(h_l, axis=0)
#     hprof_yl = np.sum(h_l, axis=1)
#     hprof_xr = np.sum(h_r, axis=0)
#     hprof_yr = np.sum(h_r, axis=1)

#     mx_l = get_mean(x,hprof_xl)
#     mx_r = get_mean(x,hprof_xr)
#     my_l = get_mean(y,hprof_yl)
#     my_r = get_mean(y,hprof_yr)


#     x_mid = (x[1:] + x[:-1])/2
#     y_mid = (y[1:] + y[:-1])/2

#     ndf = np.shape(x_mid)[0]*np.shape(y_mid)[0]

#     X = [x_mid,y_mid]
#     chi2_2d = Chi2_2d(get_fit_func_,
#                       X,
#                       h_l,
#                       h_r)

#     m2d = Minuit(chi2_2d,
#                  mx=(mx_l+mx_r)*0.5,
#                  sx=5,
#                  my=(my_l+my_r)*0.5,
#                  sy=3,
#                  mx2=0.,
#                  sx2=5.,
#                  my2=0.,
#                  sy2=3.,
#                  N_grel = 0.,
#                  Ksi=INITIAL_Q,
#                  phi_lin=-2.413,
#                  P=0,
#                  V=1.,
#                  E=INITIAL_ENERGY,
#                  L=33000.,
#                  NL=1,
#                  NR=1,
#                  alpha=0.)

#     m2d.fixed['mx']=False
#     m2d.fixed['my']=False
#     m2d.fixed['sy']=False

#     use_double_gaus = DOUBLE_GAUS
#     m2d.fixed['mx2']=not use_double_gaus
#     m2d.fixed['my2']=not use_double_gaus
#     m2d.fixed['sx2']= use_double_gaus
#     m2d.fixed['sy2']= use_double_gaus
#     m2d.fixed['N_grel']= use_double_gaus

#     m2d.fixed['Ksi']=FIXQ
#     m2d.fixed['phi_lin']=False
#     m2d.fixed['P']=False
#     m2d.fixed['V']=True
#     m2d.values['V']=1.

#     m2d.fixed['E']=True  
#     m2d.fixed['L']=True
#     m2d.fixed['NL']=False
#     m2d.fixed['NR']=False
#     m2d.fixed['alpha']=True


#     m2d.errors['mx']=5
#     m2d.errors['my']=5
#     m2d.errors['sx']=5
#     m2d.errors['sy']=5

#     m2d.errors['sy2']=1
#     m2d.errors['mx2']=1
#     m2d.errors['my2']=1
#     m2d.errors['sx2']=1
#     m2d.errors['N_grel']=0.1

#     m2d.errors['Ksi']=0.1
#     m2d.errors['phi_lin']=1
#     m2d.errors['P']=0.1
#     m2d.errors['V']=0.1

#     m2d.errors['NL']=100
#     m2d.errors['NR']=100

#     m2d.errors['alpha']=0.1

#     m2d.limits['mx']=(-20,20)
#     m2d.limits['my']=(-10,10)
#     m2d.limits['sx']=(1.,8.)
#     m2d.limits['sy']=(1.,5.)


#     m2d.limits['mx2']=(-20,20)
#     m2d.limits['my2']=(-10,10)
#     m2d.limits['sx2']=(0.5,100.)
#     m2d.limits['sy2']=(0.5,100.)
#     #m2d.limits['N_grel']=(0.,0.95)

#     m2d.limits['Ksi']=(-1,1)
#     #m2d.limits['phi_lin']=(-2*3.1415, 2*3.1415)
#     #m2d.limits['phi_lin']=(-np.pi/4, np.pi/4.)
#     #m2d.limits['P']=(-0.92376,0.92376) 
#     m2d.limits['P']=(-10,10.) 
#     m2d.limits['V']=(-1., 1.)
#     m2d.limits['NL']=(0.1, 1e6)
#     m2d.limits['NR']=(0.1, 1e6)
#     m2d.limits['alpha']=(-3.14/4,+3.14/4)

#     m2d.print_level = 0
#     m2d.errordef=1
#     m2d.migrad()
#     m2d.hesse()
#     print(m2d)
#     #m2d.minos()
#     return m2d

def make_fit(config, h_dict):
    
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
       print(m2d)
    end_time = time.time()
    print("Fit time: ", end_time-begin_time, " s")
    return m2d, ndf


def online_fit(config, regex_line):
    hist_fpath = config['hist_fpath']
    xrange = config['xrange']
    n_files = config['n_files']
    need_blur = config['need_blur']
    fname = np.sort(np.array(glob.glob1(hist_fpath , regex_line)))[-2] #get 2nd last file
    fname_prev = fname
    file_count = 0
    attempt_count = 0
    fig1, ax1 = init_fit_figure(label = 'L', title='Left')
    fig2, ax2 = init_fit_figure(label = 'R', title='Right')
    fig3, ax3 = init_fit_figure(label = 'Diff', title='Diff')
    counter = 0
    try:
        while True:
            if fname != fname_prev:
                if file_count == 0:
                    h_dict = load_hist(hist_fpath, fname)
                    buf_dict = h_dict
                    fname_start = fname
                    calc_asymmetry(h_dict)
                else:
                    buf_dict = load_hist(hist_fpath, fname)
                    h_dict = add_statistics(h_dict, buf_dict)
                    calc_asymmetry(h_dict)
                file_count += 1
                attempt_count = 0
                fname_prev = fname
            else:
                time.sleep(1)
                attempt_count +=1
                print('Waiting for the new file: {:3d} seconds passed'.format(attempt_count), end= '\r')
            if file_count == n_files:
                #h_dict = mask_hist(config, h_dict)
                if need_blur:
                    h_dict = make_blur_nik(h_dict)
                fitres, ndf = make_fit(config, h_dict)
                plot_fit(h_dict, fitres, xrange, fig1, ax1, diff=False, pol='l')
                plot_fit(h_dict, fitres, xrange, fig2, ax2, diff=False, pol='r')
                plot_fit(h_dict, fitres, xrange, fig3, ax3, diff=True)
                plt.show(block=False)
                plt.pause(1)
                #stats = get_raw_stats(h_dict)
                #print_pol_stats(fitres)
                #moments=get_moments(h_dict)
                #calc_all_moments(h_dict)
                chi2_normed = fitres.fval / (ndf - fitres.npar)
                if chi2_normed < 1e100:
                    print('Chi2: {}'.format(chi2_normed))
                    #fitres_file = config['fitres_file']
                    #write2file_(fitres_file, fname, fitres, counter,moments)
                else:
                    print('Chi2 is to big: {}'.format(chi2_normed))
               
                del buf_dict
                file_count = 0
                del h_dict
                counter += 1
            fname = np.sort(np.array(glob.glob1(hist_fpath , regex_line)))[-2]

    except KeyboardInterrupt:
        print('\n***Exiting fit program***')
        pass

def offline_fit(config, regex_line):
    hist_fpath = config['hist_fpath']
    xrange = config['xrange']
    n_files = config['n_files']
    need_blur = config['need_blur']
    blur_algo = config['blur']
    fname_list = np.sort(np.array(glob.glob1(hist_fpath , regex_line))) #get 2nd last file
    file_count = 0
    if config['plot3d']:
        fig0, ax0 = init_data_figure(label='Data')
    fig1, ax1 = init_fit_figure(label = 'L', title='Left')
    fig2, ax2 = init_fit_figure(label = 'R', title='Right')
    fig3, ax3 = init_fit_figure(label = 'Diff', title='Diff')
    counter = 0
    ymask = config['ymask']
    for y in ymask[0]:
        for x in range(config['xrange'][0], config['xrange'][1], 2):
            config['mask'].append([y,x])
    try:
        for fname in fname_list:
            if file_count == 0:
                h_dict = load_hist(hist_fpath, fname)
                buf_dict = h_dict
                fname_start = fname
            #    calc_asymmetry(h_dict)
            else:
                buf_dict = load_hist(hist_fpath, fname)
                h_dict = add_statistics(h_dict, buf_dict)
            #    calc_asymmetry(h_dict)
            file_count += 1
            attempt_count = 0
            fname_prev = fname
            if file_count == n_files:
                h_dict = mask_hist(config, h_dict)
                if need_blur:
                    print("Blur algo: ", blur_algo)
                    if blur_algo == 'default':
                        h_dict = make_blur(h_dict)
                        print("Use default blur algo")
                    if blur_algo == 'nik':
                        h_dict = make_blur_nik(h_dict)
                        print("Use nik blur algo")
                fitres, ndf = make_fit(config, h_dict)
                if config['plot3d']:
                        plot_data3d(h_dict,fitres , xrange, fig0, ax0, h_type='diff_l')
                        plot_data3d(h_dict, fitres, xrange, fig0, ax0, h_type='diff_r')
                plot_fit(h_dict, fitres, xrange, fig1, ax1, diff=False, pol='l')
                plot_fit(h_dict, fitres, xrange, fig2, ax2, diff=False, pol='r')
                plot_fit(h_dict, fitres, xrange, fig3, ax3, diff=True)
                plt.show(block=False)
                plt.pause(1)
                stats = get_raw_stats(h_dict)
                print_pol_stats_nik(fitres)
                moments=get_moments(h_dict)
                chi2_normed = fitres.fval / (ndf - fitres.npar)
                if chi2_normed < 1e100:
                     print('Chi2: {}'.format(chi2_normed))
                     fitres_file = config['fitres_file']
                     write2file_nik(fitres_file, fname, fitres, counter,moments, chi2_normed)
                else:
                     print('Chi2 is too big: {}'.format(chi2_normed))
                file_count = 0
                counter += 1
                del buf_dict
                del h_dict
                if INTERACTIVE_MODE:
                    input("Press Enter to continue...")
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
    #parser.add_argument('--blur', nargs='?', help='Blur algorithm', type=str, default='default')
    parser.add_argument('--blur', help='Blur algorithm', type=str, default='default')
    parser.add_argument('--plot3d', help='Plot 3D left right', action='store_true')
    #parser.add_argument('--noblur', help='Disable blur')
    
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
            if args.offline:
                fit_ = offline_fit
            else:
                fit_ = online_fit

            if args.regex_line:
                regex_line = str(args.regex_line)
            else:
                regex_line = str(config['regex_line'])

            fit_(config, regex_line) 
if __name__ == '__main__':
    main()
