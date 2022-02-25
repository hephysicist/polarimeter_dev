#!/usr/bin/env python3 
import os 
import sys
sys.path.append('.')
sys.path.append('../lib')

import glob
import numpy as np 
import time
from numba import jit
import argparse
import matplotlib.pyplot as plt
import yaml
import statistics as st
from ROOT import TCanvas, TH1D

from mapping import get_side_ch_id, get_center_ch_id, get_xy
from pol_lib import *
from pol_plot_lib import plot_hist, init_monitor_figure, plot_hitmap

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
#def init_monitor_figure():
#    plt.ion()
#    fig, ax = plt.subplots(
#        nrows=3, ncols=1, sharex=False, sharey=False,  
#        gridspec_kw={'width_ratios':[1], 'height_ratios':[1,1,1]})
#    fig.set_tight_layout(True)
#    fig.tight_layout(rect=[0, 0, 1, 1])
#    fig.set_size_inches(8,8)
#    return fig, ax

#def plot_hitmap(fig, ax, ch_distr_hist):
#    print('Plotting')
#    coor_grid = get_coor_grid()
#    xxc, yyc = np.meshgrid(coor_grid['xc'], coor_grid['yc'])
#    (ax1, ax2,ax3) = ax
#    ratio = 15
#    mean_arr = np.zeros((20,32))
#    sigma_arr = np.zeros((20,32))
#    n_evt_arr = np.zeros((20,32))
#    for iy, ix in np.ndindex(ch_distr_hist.shape):
#        pad_list = ch_distr_hist[iy, ix]
#        if pad_list:
#            mean_arr[iy,ix] = st.fmean(pad_list)
#            sigma_arr[iy,ix] = st.pvariance(pad_list)
#            n_evt_arr[iy,ix] = len(pad_list)
#    for i in range(1,2):
#        exec('ax{:d}.clear()'.format(i))
#    ax1.set_title('Mean value')
#    ax1.set_aspect(1)
#    plot_hist(xxc, yyc, mean_arr, ax1, None, fig)

#    ax2.set_title('Sigma value')
#    ax2.set_aspect(1)
#    plot_hist(xxc, yyc, sigma_arr, ax2, None, fig)
#    ax3.set_title('N evt per channel')
#    ax3.set_aspect(1)
#    plot_hist(xxc, yyc, n_evt_arr, ax3, None, fig)

def get_occupancy_plots(regex_line):
    print('Caution: you are using relative efficiency measurement algorithm!')
    conf_file = open(os.getcwd()+'/../'+str('pol_config.yml'), 'r')
    config = yaml.load(conf_file, Loader=yaml.Loader)
    hist_fpath = '/storage/pol_rel_eff/'
    file_arr = np.sort(np.array(glob.glob1(hist_fpath, regex_line)))
    buf_dict = []
    h_dict = load_hist(hist_fpath,file_arr[0])
    fig, ax = init_monitor_figure()
    c1 = TCanvas( 'c1', 'Monitor', 10000,6000)
    #hist = TH1D( 'monitor', 'hit number distribution', 50, 50000,90000)
    hist = TH1D ( 'h1', 'h1', 300,0,300)
    hist.SetFillColor(4)
    hist.SetFillStyle(3004)
    empty_ch = 0
    filled_ch = 0
    try:
        for single_file in file_arr[1:]: 
            buf_dict = load_hist(hist_fpath, single_file)
            h_dict = accum_data(h_dict, buf_dict)
            print('Evt l: ',sum(sum(h_dict['hc_l'])), 'Evt r: ', sum(sum(h_dict['hc_r'])))
        buf_dict = h_dict['hc_r']+h_dict['hc_l']
        h_dict['hc_r'] = buf_dict
        h_dict['hc_l'] = buf_dict
        n_nonzero = np.count_nonzero(buf_dict > 10000)
        mean_charge = sum(sum(np.where(buf_dict > 10000, buf_dict, 0)))/n_nonzero
        print(n_nonzero, mean_charge)
        scale_arr = mean_charge/ np.where(buf_dict > 0, buf_dict, 1)
        scale_arr = np.where(scale_arr < 2., scale_arr, 0)
        for idy, idx in np.ndindex(buf_dict.shape):
            ch_id = get_center_ch_id(idx, idy, config['zone_id'])
            if(buf_dict[idy,idx] > 10000):
                filled_ch +=1
                hist.SetBinContent(hist.FindBin(ch_id), buf_dict[idy, idx]*scale_arr[idy,idx])
            else:
                empty_ch +=1
            print('Coors: x = {:2d},y = {:2d}, n_hit = {:6.0f}'.format(idx, idy, buf_dict[idy, idx]))
            #hist.Fill(h_dict['hc_r'][idy, idx])
        plot_hitmap(fig, ax, h_dict, block=False, norm=False)
        fig.canvas.draw_idle()
        plt.pause(0.1)
        hist.Draw('')
        c1.Update()
        c1.Modified()
        c1.SaveAs('h1.pdf')
        print('Empty ch: ', empty_ch, 'nonempty ch: ', filled_ch)
        np.savez(os.getcwd()+'/../scale_array', scale_arr = scale_arr)
        text = input()

    except KeyboardInterrupt:
        print('\n***Exiting online preprocessing***')
        pass

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_cluster.py")
    parser.add_argument('regex_line', nargs='?', help='Name of the file to start online preprocessing in regex format')
    args = parser.parse_args()
    regex_line = str(args.regex_line)
    get_occupancy_plots(regex_line) 

if __name__ == '__main__':
    main()
           

