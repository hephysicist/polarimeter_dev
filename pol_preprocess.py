import glob
import numpy as np 
import time
from numba import jit
import argparse
import matplotlib.pyplot as plt
import yaml
import os

from mapping import get_side_ch_id, get_center_ch_id
from pol_lib import *
from pol_fit_lib import make_blur
from pol_plot_lib import init_monitor_figure, plot_hitmap

def preprocess_bin_file(in_fpath, out_fpath, fname,amp_cut, save_raw=False):
    print('Reading file: ', fname,'...')
    raw_evt_arr = load_events(in_fpath+fname)
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
            np.savez(out_fpath+fname[:19], evt_arr = evt_arr)
            print('Saved raw events to file: ' + str(fname[:19])+'.npz')
    else:
        print('No events found after preselection!\t'+str(fname[:19])+'.npz')        
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
    mask = np.array([[9,10], [7,16], [10,21], [18,1], [15,8]])
    hist_c_l = mask_ch_map(hist_c_l, mask)
    hist_c_r = mask_ch_map(hist_c_r, mask)
    return [hist_s_l, hist_s_r, hist_c_l, hist_c_r]
    
def save_mapped_hists(hist_fpath, h_list, fname):
    grid = get_coor_grid()
    np.savez(hist_fpath+fname[:19]+'_hist',
                     hs_l=h_list[0],
                     hs_r=h_list[1],
                     hc_l=h_list[2],
                     hc_r=h_list[3],
                     xs=grid['xs'],
                     ys=grid['ys'],
                     xc=grid['xc'],
                     yc=grid['yc'])
    print('nl: ',sum(sum(h_list[2])), 'nr: ', sum(sum(h_list[3])))
    print('Saved mapped hist to the file: '+fname[:19]+'_hist.npz\n')

def online_preprocess(config, regex_line):
    bin_fpath = config['bin_fpath']
    raw_fpath = config['raw_fpath']
    save_raw = config['preprocess']['save_raw_file']
    amp_cut = config['preprocess']['amp_cut'] 
    zone_id = config['zone_id'] 
    hist_fpath = config['hist_fpath']
    fname_old = '' 
    fname = np.sort(np.array(glob.glob1(bin_fpath , regex_line)))[-2] #get 2nd last file
    attempt_count = 0
    fig, ax = init_monitor_figure()
    plt.draw()
    try:
        while True:
            if(fname_old != fname):
                evt_arr = preprocess_bin_file(bin_fpath,raw_fpath, fname, amp_cut, save_raw) #extract events to numpy array with the following structure [[polarization, value, channel, frame],...]
                if np.shape(evt_arr)[0] != 0:
                    h_list = map_events(evt_arr,zone_id) #map raw events to 2d coordinate histogramms return 4 histogram side region l/r center region l/r
                    save_mapped_hists(hist_fpath, h_list, fname) #save them to np.array[hs_l, hs_r, hc_l hc_r, xs, ys, xc, yc]
                    h_dict = load_hist(hist_fpath, fname[:19]+'_hist.npz')
                    #h_dict = make_blur(h_dict)
                    print_stats(get_raw_stats(h_dict))
                    plot_hitmap(fig, ax, h_dict, block=False, norm=False)
                    fig.canvas.draw_idle()
                    plt.pause(0.1)
                fname_old = fname
                attempt_count = 0
            else:
                time.sleep(1)
                attempt_count +=1
                print('Waiting for the new file: {:3d} seconds passed'.format(attempt_count), end= '\r')
                fname = np.sort(np.array(glob.glob1(bin_fpath , regex_line)))[-2]
    except KeyboardInterrupt:
        print('\n***Exiting online preprocessing***')
        pass

def offline_preprocess(config, regex_line):
    bin_fpath = config['bin_fpath']
    raw_fpath = config['raw_fpath']
    save_raw = config['preprocess']['save_raw_file']
    amp_cut = config['preprocess']['amp_cut'] 
    zone_id = config['zone_id'] 
    hist_fpath = config['hist_fpath']
    
    f_list = np.sort(np.array(glob.glob1(bin_fpath , regex_line))) #get file list
    fig, ax = init_monitor_figure()
    try:
        for fname in f_list:
            evt_arr = preprocess_bin_file(bin_fpath,raw_fpath, fname, amp_cut, save_raw) #extract events to numpy array with the following structure [[polarization, value, channel, frame],...]
            if np.shape(evt_arr)[0]: 
                h_list = map_events(evt_arr,zone_id) #map raw events to 2d coordinate histogramms return 4 histogram side region l/r center region l/r
                save_mapped_hists(hist_fpath, h_list, fname) #save them to np.array[hs_l, hs_r, hc_l hc_r, xs, ys, xc, yc]
                h_dict = load_hist(hist_fpath, fname[:19]+'_hist.npz')
                print_stats(get_raw_stats(h_dict))
                if config['preprocess']['draw']:
                    plot_hitmap(fig, ax, h_dict, block=False, norm=False)
    except KeyboardInterrupt:
        print('\n***Exiting offline preprocessing***')
        pass

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_preprocess.py")
    parser.add_argument('config', nargs='?', help='Name of the config file')
    parser.add_argument('--offline', help='Use this key to preprocess iteratively all, starting from regrex_line',default=False, action="store_true")
    parser.add_argument('regex_line', nargs='?', help='Name of the file to start online preprocessing in regex format')
    args = parser.parse_args()
   
    with open(os.getcwd()+'/'+str(args.config), 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print('Error opening config file:')
            print(exc)
        else:
            if args.offline:
                preprocess_ = offline_preprocess
            else:
                preprocess_ = online_preprocess

            if args.regex_line:
                regex_line = str(args.regex_line)
            else:
                regex_line = str(config['regex_line'])

            preprocess_(config, regex_line) 

if __name__ == '__main__':
    main()
