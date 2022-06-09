#!/usr/bin/env python3 
import os 
import sys
sys.path.append('.')
sys.path.append('./lib')

from iminuit import Minuit
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import glob
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import yaml
import ciso8601 #Converts string time to timestamp

from pol_lib import *
from pol_fit_lib import *
from pol_plot_lib import *
from moments import get_moments
from lsrp_pol import *
from my_stat import stat_calc_effect

my_timezone = '+07:00'

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
    if 'real E' in h_dict['env_params'].item():
        initial_values['E'] = h_dict['env_params'].item()['real_E']
    print('Energy is set to: {:4.2f}'.format(initial_values['E']))
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
       #for name in  ['sx', 'sy', 'alpha_x1', 'alpha_x2', 'alpha_y1', 'alpha_y2', 'nx1','nx2', 'ny1','ny2', 'phi', 'p1', 'p2', 'p3']:
       for name in  ['alpha_x2', 'alpha_y2', 'nx2', 'ny2', 'phi', 'p1', 'p2', 'p3']:
            m2d.fixed[name]=True
       m2d.migrad()
       m2d.hesse()
       print(m2d)
    end_time = time.time()
    #print("Fit time: ", end_time-begin_time, " s")
    return m2d, ndf

def init_figures():
    fig_l, ax_l = init_fit_figure(label = 'L', title='Left')
    fig_r, ax_r = init_fit_figure(label = 'R', title='Right')
    fig_d, ax_d = init_fit_figure(label = 'Diff', title='Diff')
    #fig_3d, ax_3d = init_data_figure(label = '3d')
    return [fig_l, fig_r, fig_d], [ax_l, ax_r, ax_d]

def show_res(config, h_dict, fitres, Fig, Ax):
    xrange = config['xrange']
    plot_fit(h_dict, fitres, xrange, Fig[0], Ax[0], diff=False, pol='l')
    plot_fit(h_dict, fitres, xrange, Fig[1], Ax[1], diff=False, pol='r')
    plot_fit(h_dict, fitres, xrange, Fig[2], Ax[2], diff=True)
    #plot_data3d(h_dict, fitres, xrange, Fig[3],  Ax[3], h_type='fl')
    #plot_data3d(h_dict, fitres, xrange, Fig[3],  Ax[3], h_type='fr')
    plt.show(block=False)
    plt.pause(1)
    
def get_unix_time_template(fname, timezone='+07:00'):
    unix_time = ciso8601.parse_datetime(fname[:19]+timezone)
    return(int(unix_time.timestamp()))

get_unix_time = np.vectorize(get_unix_time_template)

    
def get_Edep (v4E, d_freq):
    n = int(v4E/440.648)
    return (n+d_freq/818924.)*440.648*(int(d_freq) != 0)

def db_write(   db_obj,
                config,
                first_fname,
                last_fname,
                fitres,
                chi2,
                ndf,
                env_params, fit_counter, skew,version):
    
    db_obj.local_id = fit_counter
    db_obj.begintime = get_unix_time(first_fname)
    db_obj.endtime = get_unix_time(last_fname)+10 #TODO 10 means the time in seconds for the single file. Needs to be written automatically.
    db_obj.measuretime = db_obj.endtime - db_obj.begintime
    
    db_obj.Eset = env_params['vepp4E']
    db_obj.Fdep = env_params['dfreq']
    db_obj.Adep = env_params['att']
    db_obj.Fspeed = env_params['fspeed']
    db_obj.Edep = get_Edep(env_params['vepp4E'], env_params['dfreq'])
    db_obj.H = env_params['vepp4H_nmr']
    db_obj.chi2 = chi2
    db_obj.ndf = ndf
    db_obj.P.value = fitres.values['P']
    db_obj.P.error = fitres.errors['P']
    db_obj.V.value = fitres.values['V']
    db_obj.V.error = fitres.errors['V']
    db_obj.Q.value = fitres.values['Q']
    db_obj.Q.error = fitres.errors['Q']
    db_obj.NL.value = fitres.values['NL']
    db_obj.NL.error = fitres.errors['NL']
    db_obj.NR.value = fitres.values['NR']
    db_obj.NR.error = fitres.errors['NR']
    db_obj.askewx.value = skew[0]
    db_obj.askewy.value = skew[1]
    db_obj.version = version
    db_obj.write(dbname='test', user='nikolaev', host='127.0.0.1')


def read_batch(hist_fpath, file_arr, vepp4E):
    h_dict = load_hist(hist_fpath,file_arr[0])
    first_fname = file_arr[0]
    buf_dict = h_dict
    env_params = h_dict['env_params'].item()
    print('Hnmr: ', env_params['vepp4H_nmr'])
    
    print('v4E : ', env_params['vepp4E'])
    if 'real E' in env_params:
        print('real E : ', env_params['real_E'])
    print('Dep freq: ', env_params['dfreq'])
    print('Evt l: ',sum(sum(h_dict['hc_l'])), 'Evt r: ', sum(sum(h_dict['hc_r'])))
    if env_params['vepp4E'] < 1000:
        env_params['vepp4E'] = vepp4E
        print('Setting v4E parameter to: ', env_params['vepp4E'])
    for file in file_arr[1:]:
            buf_dict = load_hist(hist_fpath,file)
            h_dict = accum_data(h_dict, buf_dict)
            print('Evt l: ',sum(sum(h_dict['hc_l'])), 'Evt r: ', sum(sum(h_dict['hc_r'])))
    return h_dict
    
def make_file_list_online(hist_fpath, unix_start_time, n_files):
    file_arr = np.array(glob.glob1(hist_fpath, '20*.npz'))
    if np.shape(file_arr)[0]:
                unix_time_arr = get_unix_time(file_arr)
                buffer_size = 0
                buffer_size_old = 0
                file_buffer = np.empty(0)
                counter = 0
                while (buffer_size < n_files):
                    file_arr = np.array(glob.glob1(hist_fpath, '20*.npz')) #Choose only data files
                    unix_time_arr = get_unix_time(file_arr)
                    file_buffer = file_arr[unix_time_arr >= unix_start_time]
                    buffer_size = np.shape(file_buffer)[0]
                    if buffer_size_old != buffer_size:
                        print('Progress: ', int(buffer_size), '/', int(n_files), '\t\t\t\t\t', end='\r')
                        buffer_size_old = buffer_size
                        counter = 0
                    time.sleep(1)
                    counter +=1
    return file_buffer

def make_file_list_offline( hist_fpath,
                            unix_start_time,
                            unix_stop_time,
                            n_files=1):
    file_arr = np.array(glob.glob1(hist_fpath, '20*.npz'))
    
    if np.shape(file_arr)[0]:
        unix_time_arr = get_unix_time(file_arr)
        time_cut = np.logical_and(unix_time_arr >= unix_start_time, unix_time_arr <= unix_stop_time)
        file_buffer = file_arr[time_cut]
        if np.shape(file_buffer)[0] >= n_files:
            file_buffer = file_buffer[:n_files]
    return file_buffer
    
def accum_data_and_make_fit(config, start_time, stop_time, vepp4E, offline = False, version=0):
    hist_fpath = config['hist_fpath']
    n_files = int(config['n_files'])
    file_arr = np.array(glob.glob1(hist_fpath, '20*.npz')) #Choose only data files
    
    if not offline:
        unix_start_time = int(time.time())
    else: 
        unix_start_time = get_unix_time(start_time)
    unix_stop_time = get_unix_time(stop_time)
    
    if config['scale_hits']:
        scale_file = np.load(os.getcwd()+'/scale_array.npz', allow_pickle=True)
        scale_arr = scale_file['scale_arr']
    fig_arr, ax_arr = init_figures()
    db_obj = lsrp_pol()
    fit_counter = 0
    try:
        while(1):
                if not offline:
                    file_buffer = make_file_list_online(   hist_fpath,
                                                                    unix_start_time,
                                                                    n_files)
                else:
                    file_buffer = make_file_list_offline(  hist_fpath,
                                                                    unix_start_time,
                                                                    unix_stop_time,
                                                                    n_files)
                   
                    if np.shape(file_buffer)[0] < n_files:
                        print('Batch size [{:d}] is less than required number of files [{:d}]!\nExiting fit programm.'.format( np.shape(file_buffer)[0],n_files))
                        break
                file_buffer = np.sort(file_buffer)
                unix_start_time = get_unix_time(file_buffer[-1])
                h_dict = read_batch(hist_fpath, file_buffer, vepp4E)
                h_dict = mask_hist(config, h_dict)
                if config['need_blur']:
                    h_dict = eval(config['blur_type']+'(h_dict)')
                if config['scale_hits']:
                    h_dict['hc_l'] *= scale_arr
                    h_dict['hc_r'] *= scale_arr
                print('Performing fit...')
                skew_l, skew_r = stat_calc_effect(h_dict)
                skew = [skew_l[0]-skew_r[0], skew_l[1]-skew_r[1]]
                print('skew_ly:{:1.4f}\tskew_ry:{:1.4f} \tsly-sry:{:1.3f}'.format(skew_l[1], skew_r[1], skew_l[1]-skew_r[1]))
                fitres, ndf = make_fit(config, h_dict)
                raw_stats = get_raw_stats(h_dict)
                print_stats(raw_stats)
                print_pol_stats_nik(fitres)
                moments = get_moments(h_dict)
                show_res(config, h_dict, fitres, fig_arr, ax_arr)
                chi2 = fitres.fval
                true_ndf = (ndf - fitres.npar)
                chi2_normed = chi2/true_ndf
                fit_counter +=1
                is_db_write = True
                if not config['continue']:
                    text = input("Write to db? (y/N):")
                    if text!="y":
                        is_db_write=False
                if is_db_write:
                    db_write(db_obj, config, file_buffer[0], file_buffer[-1], fitres, chi2, true_ndf, h_dict['env_params'].item(), fit_counter, skew, version)
    except KeyboardInterrupt:
        print('\n***Exiting fit program***')
        pass

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_fit.py")
    parser.add_argument('--offline', help='Parameter to distinguish offline fits (db vesrion).', default=False, action='store_true')
    parser.add_argument('start_time', nargs='?', help='Time of the file to start offline fit in regex format')
    parser.add_argument('stop_time', nargs='?', help='Time of the file to start offline fit in regex format', default='2100-01-01T00:00:01')
    parser.add_argument('--noblur', help='Disable blur', action='store_true')
    parser.add_argument('--version', help='Parameter to distinguish offline fits (db vesrion).',nargs='?', default=0)
    parser.add_argument('--config', help='Name of the config file to use while performing fit',nargs='?', default='pol_fit_config.yml')
    parser.add_argument('--E', help='vepp4 E', default=0)
    args = parser.parse_args()
    print('\nReading config file: ', os.getcwd()+'/'+args.config +'\n')
    with open(os.getcwd()+'/'+args.config, 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
            if args.E:
                vepp4E = float(args.E)
            else:
                vepp4E = config['initial_values']['E']
        except yaml.YAMLError as exc:
            print('Error opening pol_config.yaml file:')
            print(exc)
        else:
            accum_data_and_make_fit(config, args.start_time, args.stop_time, vepp4E, args.offline, args.version) 
if __name__ == '__main__':
    main()
