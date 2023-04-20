#!/usr/bin/env python3 
import os 
import sys
sys.path.append('.')
sys.path.append('./lib')
sys.path.append('./lib/fit_methods')

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
from importlib import import_module

from pol_lib import *
from pol_fit_lib import large_deviation_blur, make_central_coord, print_pol_stats
from pol_plot_lib import *
from moments import get_moments
from database_interface import Db_obj, db_write

import copy

my_timezone = '+07:00'

fit_state = None

def make_fit(config, h_dict):
    data_left = h_dict['hc_l']
    data_right = h_dict['hc_r']
    xy_coord = make_central_coord(h_dict['xc'], h_dict['yc'])
    cfg = copy.deepcopy(config)
    if  cfg['model_params']['E'][0] < 1000 : 
        cfg['model_params']['E'][0] = h_dict['env_params']['vepp4E']

    fit_method = cfg['fit_method'] #Importing module with desired fit method
    full_src_fname = os.getcwd()+'/lib/fit_methods/fit_method'+str(fit_method)
    fit_method_module = import_module('fit_method'+str(fit_method), full_src_fname)
    fm = eval('fit_method_module.FitMethod'+str(fit_method)+'(xy_coord, data_left, data_right)')
    fm.fit(cfg)
    data_fields = fm.get_fit_result(cfg)

    return fm, data_fields


def show_res(fitter, data_fields, ax, begintime,endtime):
    for the_ax in ax:
        the_ax.cla()

    total_figure_list = list(data_fields.keys())

    main_figure_list = []
    def show(name, id):
        try : 
            data_fields[name].draw(ax[id]) 
            main_figure_list.append(name)
        except KeyError: 
            pass

    print_fit_results(ax[0], fitter, begintime,endtime)

    show('data_sum_py'  , 1)
    show('fit_sum_py'   , 1)
    show('data_sum_px'  , 2)
    show('fit_sum_px'   , 2)
    show('data_sum'     , 3)
    show('fit_sum'      , 4)

    show('data_diff_py' , 5)
    show('fit_diff_py'  , 5)
    show('data_diff_px' , 6)
    show('fit_diff_px'  , 6)
    show('data_diff'    , 7)
    show('fit_diff'     , 8)

    #show('beam_shape'   , 9)
    #show('efficiency'   , 10)
    #show('remains'      , 11)

    remaining_figure_list = list(set(total_figure_list).difference(set(main_figure_list)))


    ax[1].grid()
    ax[2].grid()
    ax[5].grid()
    ax[6].grid()


    plt.show(block=False)
    plt.pause(1)
    return remaining_figure_list
    
def show_res_gen(data_fields, ax, remaing_figure_list=None, time=None):
    for the_ax in ax:
        the_ax.cla()
    idx = 0
    data_names = [s for s in remaing_figure_list if 'data' in s]
    fit_names = [s for s in remaing_figure_list if 'fit' in s]
    other = [s for s in remaing_figure_list if not (('fit' in s) or ('data' in s))]
    for the_data_name in data_names:
        if idx < len(ax):
            data_fields[the_data_name].draw(ax[idx])
            the_fit_name = [string for string in fit_names if  the_data_name[5:] == string[4:] ]
            if len(the_fit_name):
                if data_fields[the_fit_name[0]].data.size == data_fields[the_data_name].data.size:
                    if len(np.shape(data_fields[the_fit_name[0]].data)) == 2:
                        idx +=1
                    data_fields[the_fit_name[0]].draw(ax[idx])
                else:
                    print("ERROR: found a fit plot for {:s} that has a different shape! Please correct your data_field!".format(the_data_name))
            idx +=1
        else:
            print('Not enough axes for plots!')
    for the_name in other:
        if idx < len(ax):
            data_fields[the_name].draw(ax[idx])
            idx += 1
        else:
            print('Not enough axes for plots!')
    plt.show(block=False)
    plt.pause(1)
        
    
def get_unix_time_template(fname, timezone='+07:00'):
    unix_time = ciso8601.parse_datetime(fname[:19]+timezone)
    return(int(unix_time.timestamp()))

get_unix_time = np.vectorize(get_unix_time_template)

    
def get_Edep (v4E, d_freq):
    n = int(v4E/440.648)
    return (n+d_freq/818924.)*440.648*(int(d_freq) != 0)
    
def get_env_params(h_dict, default_params_dict):
    env_dict_valid = False
    if 'env_params' in h_dict:
        env_params = h_dict['env_params'].item()
        env_dict_valid = True
    else:
        print('WARNING: your data is outdated and does not contain env_params dictionary.')
        env_params = {}
        var_names = ['vepp4E', 'vepp4H_nmr','dfreq', 'att', 'fspeed']
        var_units = ['MeV', 'MeV', 'Hz', 'dB', 'Hz/sec']
        if not default_params_dict:
            if input('Do you want to set default parameters? y/N\n') == 'y':
                for var_name in var_names[2:]:
                     env_params[var_name] = -1.
                env_params['vepp4E'] = 4760.
                env_params['vepp4H_nmr'] = 4738.
                env_params['real_E'] = guess_real_energy(env_params['vepp4E'], env_params['vepp4H_nmr'])
            else:
                print('Please enter all necessary information manually:\n')
                for var_name, var_unit in zip(var_names, var_units):
                    env_params[var_name] = float(input('Enter {:s} in {:s}\n'.format(var_name, var_unit)))
            print('Check the env_params dictionary:\n')
            for key in env_params.keys():
                print('{:s} = {:4.0f}'.format(key, env_params[key]))
        else:
            env_params = default_params_dict
            print('Check the env_params dictionary:\n')
            for key in env_params.keys():
                print('{:s} = {:4.0f}'.format(key, env_params[key]))
    return env_params, env_dict_valid

def print_batch_item_stat(count, filename, D, env_params):
    line_size = 129
    if count == 1:
        print(''.rjust(line_size,'━'))
        print('{:>5} {:^30} {:>12} {:>12} {:^15} {:>15} {:>15} {:>15}'.format(
        '#', 'file', 'Nl', 'Nr', '(Nl-Nr)/(Nl+Nr),%',  'Eset, MeV', 'H, Gs', 'Fdep, Hz') )
        print(''.rjust(line_size,'─'))
    if count == '':
        print(''.rjust(line_size,'─'))
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
        env_params['vepp4E'],
        env_params['vepp4H_nmr'],
        env_params['dfreq']))

def read_batch(hist_fpath, file_arr, vepp4E, default_params_dict=None):
    print('Reading ', len(file_arr), ' files: ', file_arr[0], ' ... ', file_arr[-1])
    buf_dict_list = []
    count  = 0 
    vepp4E_list = []
    env_dict_valid = True
    env_params = {}
    
    for file in file_arr:
            buf_dict = load_hist(hist_fpath,file)
            if env_dict_valid:
                env_params, env_dict_valid = get_env_params(buf_dict, default_params_dict)
                if not env_dict_valid:
                    default_params_dict = env_params
                E = env_params['vepp4E']
            if E > 1000: vepp4E_list.append(E)
            buf_dict_list.append(buf_dict)
            print_batch_item_stat(count+1, file, buf_dict, env_params)
            count+=1
    h_dict = buf_dict_list[0]
    if not env_dict_valid:
        h_dict['env_params'] = env_params
    else:
        h_dict['env_params'] = h_dict['env_params'].item() #Gryazny hak Ne nado tak.
    E_mean = np.average(vepp4E_list)
    h_dict['env_params']['vepp4E'] = E_mean
    for bd in buf_dict_list[1:]:
        h_dict = accum_data(h_dict, bd)
    print_batch_item_stat('', 'all {} files'.format(len(buf_dict_list)), h_dict, env_params)
    return h_dict, default_params_dict
    

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
            t = datetime.datetime.strptime(time_string, time_format)
            if time_format ==  "%H:%M:%S" or time_format == "%H:%M":
                t = from_today(t)
            break
        except ValueError:
            if time_format == "%H:%M":
                print("pol_fit: ERROR: Unable to parse time: ", time_string)
                exit(1)
    return datetime.datetime.timestamp(t)

def save_png_figure(config, fig , timestamp):
    if config['figsave']:
        begin_timestamp = timestamp[:-9].replace(':','-')
        filename = '{}/{}.png'.format(config['savedir'], begin_timestamp)
        if not os.path.exists(config['savedir']):
            print('pol_fit.py: "{}" directory doesnt exists! Create new one'.format(config['savedir']))
            os.mkdir(config['savedir'])
        if os.path.isdir(config['savedir']):
            fig.savefig(filename) 
        else:
            print('pol_fit.py: ERROR: Saving figure:  "{}" is not directory!'.format(config['savedir']))

    
def unx2str(unixtime):
    return datetime.datetime.fromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S')

def accum_data_and_make_fit(config, start_time, stop_time):
    vepp4E = config['model_params']['E'][0]
    hist_fpath = config['hist_fpath']
    regex_line = config['regex_line']
    n_files = int(config['n_files'])
    
    if config['offline']:
        unix_start_time = get_unixtime_smart(start_time, fix_future=True)
    else: 
        file_arr = glob.glob1(hist_fpath, regex_line)
        size = len(file_arr)
        if size < n_files:
            if size == 0:
                unix_start_time = datetime.timestamp(datetime.now())
            else:
                unix_start_time = get_unix_time( file_arr[0] )
        else:
            unix_start_time = get_unix_time( file_arr[-n_files] )

    unix_stop_time = get_unixtime_smart(stop_time)
    
    if config['scale_hits']:
        scale_file = np.load(os.getcwd()+'/rel_eff_array.npz', allow_pickle=True)
        scale_arr = scale_file['scale_arr']
    db_obj = Db_obj()
    fit_counter = 0
    default_params_dict = {}
    INIT_FIGURES = False
    INIT_ADD_FIGURES = False
    try:
        while(1):
                file_buffer = make_file_list(hist_fpath, regex_line,  unix_start_time, unix_stop_time, n_files)
                file_buffer = np.sort(file_buffer)
                unix_start_time = get_unix_time(file_buffer[-1])+1
                h_dict, default_params_dict = read_batch(hist_fpath, file_buffer, vepp4E, default_params_dict)
                h_dict = mask_hist(config, h_dict)
                if config['scale_hits']:
                    h_dict['hc_l'] *= scale_arr
                    h_dict['hc_r'] *= scale_arr
                if config['need_blur']:
                    h_dict = eval(config['blur_type']+'(h_dict)')

                begintime=get_unix_time(file_buffer[0])
                endtime=get_unix_time(file_buffer[-1])
                file_count_time = 10.0 if len(file_buffer)==1 else (endtime-begintime)/(len(file_buffer)-1)
                endtime = endtime+file_count_time

                print('Performing fit...')
                fitter, data_fields = make_fit(config, h_dict)
                moments = get_moments(h_dict)
                raw_stats = get_raw_stats(h_dict, endtime-begintime, moments)
                print_stats(raw_stats)
                print_pol_stats(fitter)

                if not INIT_FIGURES:
                    INIT_FIGURES = True
                    fig, ax = init_figure('Laser Polarimeter 2D Fit')
                remaining_figure_list = show_res(fitter, data_fields, ax, unx2str(begintime),unx2str(endtime) )


                if config['draw_additional_figures'] and len(remaining_figure_list)>0:
                    if not INIT_ADD_FIGURES:
                        INIT_ADD_FIGURES = True
                        fig1, ax1 = init_figure_gen('Laser Polarimeter additional plots', data_fields)
                    show_res_gen(data_fields, ax1, remaining_figure_list, file_buffer[0])

                save_png_figure(config, fig, file_buffer[0])
                fit_counter +=1
                is_db_write = True
                if not config['continue']:
                    text = input("Write to db? (y/N):")
                    if text!="y":
                        is_db_write=False
                if is_db_write:
                    env_params = h_dict['env_params']
                    e_dep = get_Edep(env_params['vepp4E'], env_params['dfreq'])
                    db_obj.asym_y.value = raw_stats['Ay']
                    db_obj.asym_y.error = raw_stats['dAy']
                    db_obj.asym_x.value = raw_stats['Ax']
                    db_obj.asym_x.error = raw_stats['dAx']
                    db_write(   db_obj,
                                config,
                                get_unix_time(file_buffer[0]),
                                get_unix_time(file_buffer[-1]),
                                fitter,
                                env_params,
                                e_dep,
                                fit_counter,
                                config['version'])
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
    #parser.add_argument('--fit_version', help='Number of the fit version', default=3)
    parser.add_argument('--config', help='Name of the config file to use while performing fit',nargs='?', default='pol_fit.yml')
    parser.add_argument('--E', help='vepp4 E', default=0)
    parser.add_argument('--L', help='photon flight length', default=0)
    parser.add_argument('--N', help='Number of preprocessed files to fit', default=30)
    parser.add_argument('-i', help='Interactive mode',action='store_true')
    parser.add_argument('--nonstop', help='Non Interactive mode',action='store_true')
    parser.add_argument('--figsave', help='save figures into dir')
    parser.add_argument('--draw-additional-figures', help='Draw additional figures', action='store_true')


    args = parser.parse_args()
    full_conf_fname = args.config
    print('Reading config file: ', full_conf_fname+'\n')
    with open(full_conf_fname, 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
            
            if args.E:
                config['model_params']['E'][0] = float(args.E)
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

            if args.nonstop:
                config['continue']=True
                print ("Set non interactive mode (continue = True)")

            if args.version:
                config['version'] = args.version

            if args.offline or args.start_time:
                config['offline'] = True
                print ("Starting from  date ", args.start_time, '...')
            else:
                config['offline'] = False
                print ("Starting from  now...")

            if args.figsave:
                config['savedir'] = args.figsave
                config['figsave'] = True
            else:
                config['figsave'] = False

            if args.draw_additional_figures:
                config['draw_additional_figures'] = True
            else:
                config['draw_additional_figures'] = False


        except yaml.YAMLError as exc:
            print('Error opening ' + full_conf_fname + ' file:')
            print(exc)
        else:
            accum_data_and_make_fit(config, args.start_time, args.stop_time) 
if __name__ == '__main__':
    main()
