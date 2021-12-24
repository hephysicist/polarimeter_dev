#!/usr/bin/env python3 
import glob
import numpy as np 
import time
from numba import jit
import argparse
import matplotlib.pyplot as plt
import yaml
import os

from lib.mapping import get_side_ch_id, get_center_ch_id
from lib.pol_lib import *
from lib.pol_plot_lib import init_monitor_figure, plot_hitmap

def get_evt_arr(in_fpath, out_fpath, f_name,amp_cut, save_raw=False):
    print('Reading file: ', f_name,'...')
    raw_evt_arr = load_events(in_fpath+f_name)
    print('Uploaded '+str(np.shape(raw_evt_arr)[0])+' raw events')
    print('Preprocessing...')
    raw_evt_arr = raw_evt_arr[raw_evt_arr[:,1]>0,:] #get rid of trigger events
    evt_list = []
    for evt in raw_evt_arr:
        if evt[1] > amp_cut:
            evt_list.append([evt[0],  # polarization
                             evt[1],  # signal
                             evt[2],  # channel
                             evt[3]]) #frame
    evt_arr = np.array(evt_list)
    n_evt = np.shape(evt_arr)[0]
    if  n_evt:
        print('Preprocessed '+str(n_evt)+' raw events')
        if save_raw:
            np.savez(out_fpath+f_name[:19], evt_arr = evt_arr)
            print('Saved raw events to file: ' + str(f_name[:19])+'.npz')
    else:
        print('No events found after preselection!\t'+str(f_name[:19])+'.npz')        
    return evt_arr

def map_events(evt_arr, zone_id):
    cut = evt_arr[:,3]==2 #Choose only second frame 
    cut_l = np.logical_and(cut, evt_arr[:,0]==-1)
    cut_r = np.logical_and(cut, evt_arr[:,0]==+1)
    evt_arr_l = evt_arr[cut_l,:]
    evt_arr_r = evt_arr[cut_r,:]
    
    ch_id_hist_l ,_ = np.histogram(evt_arr_l[:,2], range=[0,1280], bins=1280)
    ch_id_hist_r ,_ = np.histogram(evt_arr_r[:,2], range=[0,1280], bins=1280)

    hist_s_l = np.zeros((20,32))
    hist_s_r = np.zeros((20,32))
    hist_c_l = np.zeros((20,32))
    hist_c_r = np.zeros((20,32))
    
    for x in range(0,32):
        for y in range (0,20):
            ch_id_s = get_side_ch_id(x,y, zone_id)
            hist_s_l[y,x] = ch_id_hist_l[ch_id_s]
            hist_s_r[y,x] = ch_id_hist_r[ch_id_s]  

            ch_id_c = get_center_ch_id(x,y, zone_id)
            hist_c_l[y,x] = ch_id_hist_l[ch_id_c]
            hist_c_r[y,x] = ch_id_hist_r[ch_id_c]

    grid = get_coor_grid()
    h_dict = { 'hc_l': hist_c_l,
                'hc_r' : hist_c_r,
                'hs_l' : hist_s_l,
                'hs_r' : hist_s_r,
                'xs' : grid['xs'],
                'ys' : grid['ys'],
                'xc' : grid['xc'],
                'yc' : grid['yc']}
    return h_dict
    
def save_mapped_hist(hist_fpath, h_dict,  vepp4E, f_name):
    h_dict['vepp4E'] = vepp4E
    print('ENERGY IS:', vepp4E)
    np.savez(hist_fpath+f_name[:19]+'_hist',  hc_l = h_dict['hc_l'], hc_r = h_dict['hc_r'],
    hs_l = h_dict['hs_l'], hs_r = h_dict['hs_r'], xs = h_dict['xs'], ys = h_dict['ys'],
    xc = h_dict['xc'], yc = h_dict['yc'], vepp4E = h_dict['vepp4E'])
    print('nl: ',sum(sum(h_dict['hc_l'])), 'nr: ', sum(sum(h_dict['hc_r'])))
    print('Saved mapped hist to the file: '+f_name[:19]+'_hist.npz\n')

def preprocess_single_file(config, f_name, vepp4E, fig, ax ):
        evt_arr = get_evt_arr(config['bin_fpath'],
                                      config['raw_fpath'],
                                      f_name,
                                      config['preprocess']['amp_cut'],
                                      config['preprocess']['save_raw_file'] )
        if np.shape(evt_arr)[0] != 0:
            h_dict = map_events(evt_arr, config['zone_id']) 
            save_mapped_hist(  config['hist_fpath'],
                                h_dict,
                                vepp4E,
                                f_name) 
            print_stats(get_raw_stats(h_dict))
            if config['preprocess']['draw']:
                plot_hitmap(fig, ax, h_dict, block=False, norm=False)
                fig.canvas.draw_idle()
                plt.pause(0.1)

def preprocess(config, regex_line, offline = False):
    bin_fpath = config['bin_fpath']
    raw_fpath = config['raw_fpath']
    save_raw = config['preprocess']['save_raw_file']
    amp_cut = config['preprocess']['amp_cut'] 
    zone_id = config['zone_id'] 
    hist_fpath = config['hist_fpath']
    attempt_count = 0
    fig, ax = init_monitor_figure()
    plt.draw()
    f_name_old = '' 
    file_arr = np.sort(np.array(glob.glob1(bin_fpath , regex_line)))
    if offline:
        f_name = file_arr[0]
        vepp4E = float(input('Enter VEPP4 energy in MeV: '))
    else:
        f_name = file_arr[-2]
    file_count = 0
    try:
        while (file_count < np.shape(file_arr)[0] and offline) or (not offline):
            if(f_name_old != f_name):
                if not offline:
                     vepp4E = read_vepp4_stap()
                preprocess_single_file(config, f_name, vepp4E, fig, ax)
                f_name_old = f_name
                attempt_count = 0
                file_count +=1
            else:
                time.sleep(1)
                attempt_count +=1
                print('Waiting for the new file: {:3d} seconds passed'.format(attempt_count), end= '\r')
           
            file_arr = np.sort(np.array(glob.glob1(bin_fpath , regex_line)))
            if offline:
                f_name = file_arr[file_count]
            else:
                f_name = file_arr[-2]
    except KeyboardInterrupt:
        print('\n***Exiting online preprocessing***')
        pass

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_preprocess.py")
    parser.add_argument('config', nargs='?', help='Name of the config file')
    parser.add_argument('--offline', help='Use this key to preprocess iteratively all, starting from regrex_line',default=False, action="store_true")
    parser.add_argument('--test', help='Use this key to perform standart preprocessing on a test file',default=False, action="store_true")
    parser.add_argument('regex_line', nargs='?', help='Name of the file to start online preprocessing in regex format')
    args = parser.parse_args()
   
    with open(os.getcwd()+'/'+str(args.config), 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print('Error opening config file:')
            print(exc)
        else:
            if args.regex_line:
                regex_line = str(args.regex_line)
            else:
                regex_line = str(config['regex_line'])

            preprocess(config, regex_line, args.offline) 

if __name__ == '__main__':
    main()
