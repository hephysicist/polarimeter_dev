#!/usr/bin/env python3 
import glob
import numpy as np 
import time
from numba import jit
import argparse
import matplotlib.pyplot as plt
import yaml
import os
import statistics as st

from lib.mapping import get_side_ch_id, get_center_ch_id, get_xy
from lib.pol_lib import *
from lib.pol_plot_lib import plot_hist

def read_raw_hits(fname, n_evt=None):
    data = []
    input_file = open(fname, 'rb')
    word_arr = np.fromfile(input_file, dtype=np.uint32)
    evt_id = 0
    if n_evt:
        word_arr = word_arr[:n_evt]
    for word in word_arr:
        trig, val, ch, chip, fr = translate_word(word)
        #print(trig, '\t', val)
        if trig == 0 and fr < 3:
            data.append([evt_id, val, ch, fr]) # evt_id is the same for all events between trigger words.
            #print(evt_id)
        elif trig == 2 or trig == 4:
            evt_id +=1
    return data
    
#This is kind a copy of the original init_monitor_figure from pol plot lib. However, it deals only with charge distribution histograms
def init_monitor_figure():
    plt.ion()
    fig, ax = plt.subplots(
        nrows=3, ncols=1, sharex=False, sharey=False,  
        gridspec_kw={'width_ratios':[1], 'height_ratios':[1,1,1]})
    fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.set_size_inches(8,8)
    return fig, ax

def plot_hitmap(fig, ax, ch_distr_hist):
    print('Plotting')
    coor_grid = get_coor_grid()
    xxc, yyc = np.meshgrid(coor_grid['xc'], coor_grid['yc'])
    (ax1, ax2,ax3) = ax
    ratio = 15
    mean_arr = np.zeros((20,32))
    sigma_arr = np.zeros((20,32))
    n_evt_arr = np.zeros((20,32))
    for iy, ix in np.ndindex(ch_distr_hist.shape):
        pad_list = ch_distr_hist[iy, ix]
        if pad_list:
            mean_arr[iy,ix] = st.fmean(pad_list)
            sigma_arr[iy,ix] = st.pvariance(pad_list)
            n_evt_arr[iy,ix] = len(pad_list)
    for i in range(1,2):
        exec('ax{:d}.clear()'.format(i))
    ax1.set_title('Mean value')
    ax1.set_aspect(1)
    plot_hist(xxc, yyc, mean_arr, ax1, None, fig)

    ax2.set_title('Sigma value')
    ax2.set_aspect(1)
    plot_hist(xxc, yyc, sigma_arr, ax2, None, fig)
    ax3.set_title('N evt per channel')
    ax3.set_aspect(1)
    plot_hist(xxc, yyc, n_evt_arr, ax3, None, fig)

def preprocess_single_file(config, f_name, hist=None):
    print('Reading file: ', f_name,'...')
    hit_list = read_raw_hits(config['bin_fpath']+f_name, n_evt=5000000) #LOOK HERE: you can edit n_evt to make preprocess faster!
    n_hit = len(hit_list)
    if n_hit:
        print('Preprocessed '+str(n_hit)+' hits')
    else:
        print('No events found after preselection!\t'+str(f_name[:19])+'.npz')
    hit_list_filtered = list(filter(lambda hit: hit[-1] == 2, hit_list))
    hit_list_filtered = list(filter(lambda hit: hit[1] > 300, hit_list_filtered)) 
    zone_id = config['zone_id']
    hit_hist = np.empty((20,32),dtype=object)
    for i , j in np.ndindex(hit_hist.shape): hit_hist[i,j] = []
    for hit in hit_list_filtered:
        if hit[1] > 0:
            coors = get_xy(hit[2], zone_id)
            if coors[0] >=0:
                hit_hist[coors[1],coors[0]].append(hit[1])
    out_file = open('/storage/pol_rel_eff/'+f_name[:19]+'ch_distr_hist.npz', 'wb')
    np.savez(out_file, hit_hist, allow_pickle=True)
    out_file.close()
    return hit_hist

def preprocess(config, regex_line, offline = False):
    print('Caution: you are using relative efficiency measurement algorithm!')
    time.sleep(1)
    attempt_count = 0
    fig, ax = init_monitor_figure()
    f_name_old = '' 
    file_arr = np.sort(np.array(glob.glob1(config['bin_fpath'] , regex_line)))
    if offline:
        f_name = file_arr[0]
    else:
        f_name = file_arr[-2]
    file_count = 0
    try:
        while (file_count < np.shape(file_arr)[0] and offline) or (not offline):
            if(f_name_old != f_name):
                hit_ch_arr = preprocess_single_file(config, f_name)
                f_name_old = f_name
                attempt_count = 0
                file_count +=1
                plot_hitmap(fig, ax, hit_ch_arr)
                fig.canvas.draw_idle()
                plt.pause(0.1)
            else:
                time.sleep(1)
                attempt_count +=1
                print('Waiting for the new file: {:3d} seconds passed'.format(attempt_count), end= '\r')
            file_arr = np.sort(np.array(glob.glob1(config['bin_fpath'] , regex_line)))
            if offline and file_count < np.shape(file_arr)[0]:
                f_name = file_arr[file_count]
            else:
                f_name = file_arr[-2]
    except KeyboardInterrupt:
        print('\n***Exiting online preprocessing***')
        pass

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_cluster.py")
    parser.add_argument('--offline', help='Use this key to preprocess iteratively all, starting from regrex_line',default=False, action="store_true")
    parser.add_argument('regex_line', nargs='?', help='Name of the file to start online preprocessing in regex format')
    args = parser.parse_args()
    with open(os.getcwd()+'/'+str('pol_config.yml'), 'r') as conf_file:
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
           

