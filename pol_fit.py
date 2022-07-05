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
from datetime import datetime,timedelta
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
from FitMethod1 import *
from FitMethod2 import *
from pol_plot_lib import *
import copy

my_timezone = '+07:00'

def make_fit(config, h_dict):
    h_l = h_dict['hc_l']
    h_r = h_dict['hc_r']
    x = h_dict['xc']
    y = h_dict['yc']
    #h_l, h_r, x = arrange_region(h_l,h_r, x, config['xrange'])
    x_mid = (x[1:] + x[:-1])/2
    y_mid = (y[1:] + y[:-1])/2
    ndf = np.shape(x_mid)[0]*np.shape(y_mid)[0]
    X = [x_mid,y_mid]
    cfg = copy.deepcopy(config)
    if  cfg['initial_values']['E'] < 1000 : 
        cfg['initial_values']['E'] = h_dict['env_params'].item()['vepp4E']

    fit_method = cfg['fit_method']

    if  fit_method == 1:
        fm = FitMethod1(X, h_l, h_r)
    elif fit_method == 2:
        fm = FitMethod2(X, h_l, h_r)

    fm.fit(cfg)
    data_fields = fm.get_fit_result(cfg)

    return fm, data_fields

def show_res(fitres, data_fields, ax):
    for the_ax in ax:
        the_ax.cla()
    print_fit_results(ax[0], fitres.minuit)
    data_fields['data_sum'].draw(ax[3])
    data_fields['data_diff'].draw(ax[6])
    
    data_fields['data_sum_py'].draw(ax[1])
    data_fields['fit_sum_py'].draw(ax[1])
    ax[1].grid()
    data_fields['data_sum_px'].draw(ax[2])
    data_fields['fit_sum_px'].draw(ax[2])
    ax[2].grid()
    
    data_fields['data_diff_py'].draw(ax[4])
    data_fields['fit_diff_py'].draw(ax[4])
    ax[4].grid()
    data_fields['data_diff_px'].draw(ax[5])
    data_fields['fit_diff_px'].draw(ax[5])
    ax[5].grid()
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
    print('Reading ', len(file_arr), ' files: ', file_arr[0], ' ... ', file_arr[-1])

    def print_stat(count, filename, D):
        env = D['env_params'].item()
        nl = float(sum(sum(D['hc_l'])))
        nr = float(sum(sum(D['hc_r'])))
        if (nl+nr) > 0:
            delta_n = (nl-nr)/(nl+nr)*2
            delta_n_error = 4./(nl+nr)**2*np.sqrt(nl*nr*(nl+nr))
        else:
            delta_n, delta_n_error = 0.0, 0.0

        print('{:>5} {:>30} {:12.0f} {:12.0f} {:>15} {:>15.2f} {:>15.2f} {:15.3f}'.format( 
            count, 
            filename, 
            nl , nr ,
            '{:.2f} +- {:.2f}'.format( delta_n*100., delta_n_error*100.) ,
            env['vepp4E'],
            env['vepp4H_nmr'],
            env['dfreq']
            )
            )

    line_size = 129
    print(''.rjust(line_size,'━'))
    print('{:>5} {:^30} {:>12} {:>12} {:^15} {:>15} {:>15} {:>15}'.format(
        '#', 'file', 'Nl', 'Nr', '(Nl-Nr)/(Nl+Nr),%',  'Eset, MeV', 'H, Gs', 'Fdep, Hz'
        ) )
    print(''.rjust(line_size,'─'))
    buf_dict_list = []
    count  = 0 
    for file in file_arr:
            buf_dict = load_hist(hist_fpath,file)
            buf_dict_list.append(buf_dict)
            print_stat(count+1, file, buf_dict)
            count+=1

    h_dict = buf_dict_list[0]
    env_params = h_dict['env_params'].item()
    if env_params['vepp4E'] < 1000:
        env_params['vepp4E'] = vepp4E

    for bd in buf_dict_list[1:]:
        h_dict = accum_data(h_dict, bd)
    print(''.rjust(line_size,'─'))

    print_stat('', 'all {} files'.format(len(buf_dict_list)),  h_dict)

    return h_dict
    

def make_file_list( hist_fpath, regex_line,  unix_start_time,    unix_stop_time,    n_files=1):
    buffer_size = 0
    old_buffer_size = 0
    count = 0
    clock = ['|','/','─','\\','|','/','─','\\']
    while (buffer_size < n_files):
        file_arr = np.array(glob.glob1(hist_fpath, regex_line)) #Choose only data files
        unix_time_arr = get_unix_time(file_arr) 
        time_cut = np.logical_and(unix_time_arr >= unix_start_time, unix_time_arr <= unix_stop_time)
        file_buffer = file_arr[time_cut]
        buffer_size = np.shape(file_buffer)[0]
        current_timestamp  =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        last_file_name = file_buffer[-1] if buffer_size>0 else ''
        first_file_name = file_buffer[0] if buffer_size>0 else ''
        ln = '{:<19} Progress: {} {:>3}/{:<3}: {} ... {}'.format( 
            current_timestamp,
            clock[count%len(clock)],
            int(buffer_size), int(n_files),
                first_file_name, last_file_name)
        print(ln, end='\r' ) 
        count+=1
        time.sleep(1)
    print('')
    return file_buffer[:n_files]

def get_unixtime_smart(time_string, fix_future=False):
    def from_today(t1):
        t0 = datetime.now()
        t = datetime(t0.year, t0.month, t0.day, t1.hour, t1.minute, t1.second)
        if t>t0 or fix_future:
            t = t-timedelta(seconds=86400)
        return t

    for time_format in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%H:%M:%S", "%H:%M"]:
        try:
            t = datetime.strptime(time_string, time_format)
            if time_format ==  "%H:%M:%S" or time_format == "%H:%M":
                t = from_today(t)
            break
        except ValueError:
            if time_format == "%H:%M":
                print("pol_fit: ERROR: Unable to parse time: ", time_string)
                exit(1)
    return datetime.timestamp(t)

    
def accum_data_and_make_fit(config, start_time, stop_time):
    vepp4E = config['initial_values']['E']
    hist_fpath = config['hist_fpath']
    regex_line = config['regex_line']
    n_files = int(config['n_files'])
    
    if config['offline']:
        unix_start_time = get_unixtime_smart(start_time, fix_future=True)
    else: 
        file_arr = glob.glob1(hist_fpath, regex_line)
        size = len(file_arr)
        #print("size = ", size)
        if size < n_files:
            if size == 0:
                unix_start_time = datetime.timestamp(datetime.now())
            else:
                unix_start_time = get_unix_time( file_arr[0] )
        else:
            unix_start_time = get_unix_time( file_arr[-n_files] )

    unix_stop_time = get_unixtime_smart(stop_time)
    
    if config['scale_hits']:
        scale_file = np.load(os.getcwd()+'/scale_array.npz', allow_pickle=True)
        scale_arr = scale_file['scale_arr']
    fig, ax = init_figure('Laser Polarimeter 2D Fit')
    db_obj = lsrp_pol()
    fit_counter = 0
    try:
        while(1):
                file_buffer = make_file_list(hist_fpath, regex_line,  unix_start_time, unix_stop_time, n_files)
                file_buffer = np.sort(file_buffer)
                unix_start_time = get_unix_time(file_buffer[-1])+1
                h_dict = read_batch(hist_fpath, file_buffer, vepp4E)
                h_dict = mask_hist(config, h_dict)
                if config['scale_hits']:
                    h_dict['hc_l'] *= scale_arr
                    h_dict['hc_r'] *= scale_arr
                if config['need_blur']:
                    h_dict = eval(config['blur_type']+'(h_dict)')
                print('Performing fit...')
                skew_l, skew_r = stat_calc_effect(h_dict)
                skew = [skew_l[0]-skew_r[0], skew_l[1]-skew_r[1]]
                #print('skew_ly:{:1.4f}\tskew_ry:{:1.4f} \tsly-sry:{:1.3f}'.format(skew_l[1], skew_r[1], skew_l[1]-skew_r[1]))
                fitter, data_fields = make_fit(config, h_dict)
                raw_stats = get_raw_stats(h_dict)
                print_stats(raw_stats)
                print_pol_stats_nik(fitter.minuit)
                moments = get_moments(h_dict)
                show_res(fitter, data_fields, ax)
                chi2 = fitter.minuit.fval
                true_ndf = (fitter.ndf - fitter.minuit.npar)
                chi2_normed = chi2/true_ndf
                fit_counter +=1
                is_db_write = True
                if not config['continue']:
                    text = input("Write to db? (y/N):")
                    if text!="y":
                        is_db_write=False
                if is_db_write:
                    db_write(db_obj, config, file_buffer[0], file_buffer[-1], fitter.minuit, chi2, true_ndf, h_dict['env_params'].item(), fit_counter, skew, config['version'])
    except KeyboardInterrupt:
        print('\n***Exiting fit program***')
        pass

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_fit.py")
    parser.add_argument('--offline', help='Parameter to distinguish offline fits (db vesrion).', default=False, action='store_true')
    parser.add_argument('start_time', nargs='?', help='Time of the file to start offline fit in regex format')
    parser.add_argument('stop_time', nargs='?', help='Time of the file to start offline fit in regex format', default='2100-01-01T00:00:01')
    parser.add_argument('--blur', help='Apply blur (general_blur, nonzero_blur), default: none', default='none')
    parser.add_argument('--version', help='Parameter to distinguish offline fits (db vesrion).',nargs='?', default=0)
    parser.add_argument('--config', help='Name of the config file to use while performing fit',nargs='?', default='pol_fit_config.yml')
    parser.add_argument('--E', help='vepp4 E', default=0)
    parser.add_argument('--L', help='photon flight length', default=0)
    parser.add_argument('--N', help='Number of preprocessed files to fit', default=30)
    parser.add_argument('-i', help='Interactive mode',action='store_true')

    args = parser.parse_args()
    print('Reading config file: ', os.getcwd()+'/'+args.config +'\n')
    with open(os.getcwd()+'/'+args.config, 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
            if args.E:
                config['initial_values']['E'] = float(args.E)
                print ("Set default VEPP4 energy ", args.E, " MeV")
            if args.L:
                config['initial_values']['L'] = float(args.L)
                print ("Set default photon flight length ", args.L, " mm")
            if args.blur != 'none':
                print ("Set blur algorithm: ", args.blur)
                config['need_blur'] = True
                config['blur_type'] = args.blur
            if args.N:
                print("Set number of preprocessed files: ", args.N)
                config['n_files'] = args.N
            if args.i:
                config['continue']=False
                print ("Set interactive mode (continue = False)")

            if args.version:
                config['version'] = args.version

            if args.offline or args.start_time:
                config['offline'] = True
                print ("Starting from  date ", args.start_time, '...')
            else:
                config['offline'] = False
                print ("Starting from  now...")


        except yaml.YAMLError as exc:
            print('Error opening pol_config.yaml file:')
            print(exc)
        else:
            accum_data_and_make_fit(config, args.start_time, args.stop_time) 
if __name__ == '__main__':
    main()
