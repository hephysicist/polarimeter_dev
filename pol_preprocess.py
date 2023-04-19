#!/usr/bin/env python3 
import os 
import sys
sys.path.append('.')
sys.path.append('./lib')

import glob
import numpy as np 
import time
import argparse
import matplotlib.pyplot as plt
import yaml

from mapping import get_side_ch_id, get_center_ch_id
from pol_lib import *
from pol_plot_lib import init_monitor_figure, plot_hitmap, draw_ch_numbers
import pyinotify

def get_evt_arr(in_fpath, out_fpath, f_name,amp_cut, save_raw=False):
    print('Reading file: ', f_name,'...')
    t0 = time.time()
    raw_evt_arr = load_events(in_fpath+f_name)
    print("load_events time: ", time.time() - t0, "s")
    print('Uploaded '+str(np.shape(raw_evt_arr)[0])+' raw events')
    print('Preprocessing...')
#    raw_evt_arr = raw_evt_arr[raw_evt_arr[:,1]>0,:] #get rid of trigger events
    evt_arr = raw_evt_arr[raw_evt_arr[:,1]>amp_cut,:] # amplitude cut 5.1 sec

    ''' 12 sec
    evt_list = []
    for evt in raw_evt_arr:
        if evt[1] > amp_cut:
            evt_list.append([evt[0],  # polarization
                             evt[1],  # signal
                             evt[2],  # channel
                             evt[3]]) #frame
    evt_arr = np.array(evt_list)
    '''
    
#    print("evt_arr.shape =", evt_arr.shape)
    n_evt = np.shape(evt_arr)[0]
    #print(np.min(evt_arr), np.max(evt_arr))
    if  n_evt:
        print('Preprocessed '+str(n_evt)+' raw events')
        if save_raw:
            np.savez(out_fpath+f_name[:19], evt_arr = evt_arr)
            print('Saved raw events to file: ' + str(f_name[:19])+'.npz')
    else:
        print('No events found after preselection!\t'+str(f_name[:19])+'.npz')        
    return evt_arr

def map_events(evt_arr, zone_id):
    max_ch = 0
    max_val = -1
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
            if max_val < (ch_id_hist_l[ch_id_c] + ch_id_hist_r[ch_id_c])/2.:
                max_val = (ch_id_hist_l[ch_id_c] + ch_id_hist_r[ch_id_c])/2.
                max_ch = ch_id_c
    print('Channel with maximal load: ch={:4d}  n_evt={:5d}\n'.format(int((max_ch + 640)%1280),int(max_val)))
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

def save_mapped_hist(hist_fpath, h_dict, env_params, f_name):
    if env_params['dfreq'] < 0:
        print('No scan is running! Depolarizer is OFF.')
    np.savez(hist_fpath+f_name[:19]+'_hist',
            hc_l = h_dict['hc_l'],
            hc_r = h_dict['hc_r'],
            hs_l = h_dict['hs_l'],
            hs_r = h_dict['hs_r'],
            xs = h_dict['xs'],
            ys = h_dict['ys'],
            xc = h_dict['xc'],
            yc = h_dict['yc'],
            env_params=env_params)

    print('Evt l: ',sum(sum(h_dict['hc_l'])), 'Evt r: ', sum(sum(h_dict['hc_r'])))
    print('Saved mapped hist to the file: '+f_name[:19]+'_hist.npz\n')

def preprocess_single_file(config, f_name, env_params, fig, ax ):
        t0 = time.time()
        evt_arr = get_evt_arr(config['bin_fpath'],
                                      config['raw_fpath'],
                                      f_name,
                                      config['preprocess']['amp_cut'],
                                      config['preprocess']['save_raw_file'] )
        print("get_evt_arr time: ", time.time() - t0, "s")
        if np.shape(evt_arr)[0] != 0:
            t0 = time.time()
            h_dict = map_events(evt_arr, config['zone_id'])
            print("map_events time: ", time.time() - t0, "s")
            if config['preprocess']['impute_broken_ch']:
                h_dict = impute_hist(config, h_dict)
            save_mapped_hist(  config['hist_fpath'],
                                h_dict,
                                env_params,
                                f_name) 

            print_stats(get_raw_stats(h_dict))
            if config['preprocess']['draw']:
                plot_hitmap(fig, ax, h_dict, f_name, block=False, norm=False)
                draw_ch_numbers(ax[0], config)
                fig.canvas.draw_idle()
                plt.pause(0.1)

#def preprocess(config, regex_line, offline = False):
#    bin_fpath = config['bin_fpath']
#    raw_fpath = config['raw_fpath']
#    save_raw = config['preprocess']['save_raw_file']
#    amp_cut = config['preprocess']['amp_cut'] 
#    zone_id = config['zone_id'] 
#    hist_fpath = config['hist_fpath']
#    attempt_count = 0
#    fig, ax = init_monitor_figure()
#    plt.draw()
#    f_name_old = '' 
#    file_arr = np.sort(np.array(glob.glob1(bin_fpath , regex_line)))
#    print(file_arr)
#    if offline:
#        f_name = file_arr[0]
#        vepp4E = float(input('Enter VEPP4 energy [MeV]: '))
#        vepp4H_nmr = float(input('Enter VEPP4 H field [Gauss]: '))
#        env_params = { 'vepp4E':vepp4E, 
#                       'vepp4H_nmr':vepp4H_nmr}
#    else:
#        while len(file_arr) < 2:
#            time.sleep(10)
#            file_arr = np.sort(np.array(glob.glob1(bin_fpath , regex_line)))
#        f_name = file_arr[-2]
#        if config['preprocess']['use_depolarizer']:
#            depol_device = init_depol()
#    file_count = 0
#    dfreq = np.zeros(2)
#    try:
#        while (file_count < np.shape(file_arr)[0] and offline) or (not offline):
#            if(f_name_old != f_name):
#                tbegin=time.time()
#                if not offline:
#                     vepp4E = read_vepp4_stap()
#                     buf_list = get_par_from_file('/mnt/vepp4/kadrs/nmr.dat', par_id_arr = [1])
#                     vepp4H_nmr = float(buf_list[0])
#                     real_E = guess_real_energy(vepp4E, vepp4H_nmr)
#                     env_params = { 'vepp4E':vepp4E, 
#                                    'vepp4H_nmr':vepp4H_nmr,
#                                    'real_E': real_E}
#
#                if config['preprocess']['use_depolarizer']:
#                    [dtime, dfreq, att, fspeed] = get_depol_params(depol_device, f_name)
#                    env_params['dfreq'] = dfreq
#                    env_params['att']   = att
#                    env_params['fspeed']   = fspeed
#                else:
#                    env_params['dfreq'] = 0
#                    env_params['att']   = 0
#                    env_params['fspeed']= 0
#                print(env_params)
#                preprocess_single_file(config, f_name, env_params, fig, ax)
#                f_name_old = f_name
#                attempt_count = 0
#                file_count +=1
#                tend=time.time()
#                print("Time for proceeding single file is: ", tend-tbegin, " second")
#            else:
#                time.sleep(1)
#                attempt_count +=1
#                print('Waiting for the new file: {:3d} seconds passed'.format(attempt_count), end= '\r')
#           
#            file_arr = np.sort(np.array(glob.glob1(bin_fpath , regex_line)))
#            if offline:
#                f_name = file_arr[file_count]
#            else:
#                f_name = file_arr[-2]
#    except KeyboardInterrupt:
#        print('\n***Exiting online preprocessing***')
#        pass


class ProcessNewFile(pyinotify.ProcessEvent):
    def __init__(self, config):
        self.config = config
        self.fig, self.ax = init_monitor_figure()
        plt.draw()
        self.proceeded_files = []
        if self.config['preprocess']['use_depolarizer']:
            self.depol_device = init_depol()
    
    def process_IN_CLOSE_WRITE(self, event):
        f_name = os.path.basename(event.pathname)
        print("CLOSE_WRITE event:", f_name)
        vepp4E = read_vepp4_stap()
        buf_list = get_par_from_file('/mnt/vepp4/kadrs/nmr.dat', par_id_arr = [1])
        vepp4H_nmr = float(buf_list[0])
        real_E = guess_real_energy(vepp4E, vepp4H_nmr)
        env_params = { 'vepp4E':vepp4E, 'vepp4H_nmr':vepp4H_nmr, 'real_E': real_E}
        dfreq = np.zeros(2)
        if self.config['preprocess']['use_depolarizer']:
            [dtime, dfreq, att, fspeed] = get_depol_params(self.depol_device, f_name)
            env_params['dfreq'] = dfreq
            env_params['att']   = att
            env_params['fspeed']   = fspeed
        else:
            env_params['dfreq'] = 0
            env_params['att']   = 0
            env_params['fspeed']= 0
        print(env_params)
        preprocess_single_file(self.config, f_name, env_params, self.fig, self.ax)
        self.proceeded_files.append(self.config['bin_fpath']+f_name)
        max_number_of_proceeded_files_keep = 100
        for i in range(0,len(self.proceeded_files)- max_number_of_proceeded_files_keep):
            filename = self.proceeded_files[0]
            try:
                print("Remove proceeded file: ", filename)
                os.remove(filename)
                self.proceeded_files.pop(0)
            except:
                print("Unable to remove file: ", filename)



def online_preprocess(config):
    wm = pyinotify.WatchManager()
    wm.add_watch(config['bin_fpath'], pyinotify.IN_CLOSE_WRITE, rec=True)
    new_file_handler = ProcessNewFile(config)
    notifier = pyinotify.Notifier(wm, new_file_handler)
    print("Waiting new data files in ", config['bin_fpath'], '...')
    notifier.loop()

def main():
    np.set_printoptions(linewidth=360)
    parser = argparse.ArgumentParser("pol_preprocess.py")
    parser.add_argument('--offline', help='Use this key to preprocess iteratively all, starting from regrex_line',default=False, action="store_true")
    parser.add_argument('--test', help='Use this key to perform standart preprocessing on a test file',default=False, action="store_true")
    parser.add_argument('regex_line', nargs='?', help='Name of the file to start online preprocessing in regex format')
    args = parser.parse_args()
   
    with open(os.getcwd()+'/'+str('pol_preprocess.yml'), 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print('Error opening config file:')
            print(exc)
        else:
            if args.regex_line:
                config['regex_line'] = str(args.regex_line)
            else:
                regex_line = str(config['regex_line'])

            online_preprocess(config)

if __name__ == '__main__':
    main()
