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
from ROOT import TCanvas, TH2F

from mapping import get_side_ch_id, get_center_ch_id, get_xy
from pol_lib import *
from pol_plot_lib import init_monitor_figure, plot_hitmap

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

def check_neighbours(arr, xc, yc):
    id_list = []
    for x in np.arange(-1,2, dtype='int64')+int(xc):
         for y in np.arange(-1,2, dtype='int64')+int(yc):
                #print(f'x={x:d}, y={y:d}')
                evt_id = np.where(np.logical_and(arr[:,4]==x, arr[:,5]==y))
                if np.shape(evt_id[0])[0]:
                    id_list.append(evt_id[0][0])
    return(id_list)

def find_clusters(evt_list, charge_cut=0):
    evt_arr = np.array(evt_list)
    evt_arr_sorted = evt_arr[evt_arr[:,1].argsort()[::-1]]
    buf_arr = evt_arr_sorted[evt_arr_sorted[:,1] > charge_cut,:] #apply external cut to eliminate noize 
    cluster_list = []
    while np.shape(buf_arr)[0]:
        #print('New clustering iteration:')
        #print(buf_arr)
        xc = buf_arr[0,4]
        yc = buf_arr[0,5]
        id_arr = check_neighbours(buf_arr, xc, yc)
        #print('id_arr',id_arr)
        buf_cluster_arr = np.take(buf_arr, id_arr, axis=0)
        cluster_list.append(buf_cluster_arr)
        #print('Cluster:', buf_cluster_arr)
        buf_arr = np.delete(buf_arr, id_arr, axis=0)
    return cluster_list

#        cluster_arr = np.array(cluster_list, dtype=object)
#        #print(cluster_arr)
#        out_file = open('/home/zakharov/clustering/cluster_arr/'+fname[:19]+'_clusters.npy', 'wb')
#        out_file2 = open('/home/zakharov/clustering/cluster_arr/'+fname[:19]+'_clusters_size.npy', 'wb')
#        np.save(out_file, cluster_arr, allow_pickle=True)
#        np.save(out_file2, np.array(raw_evt_size_list))

def get_cluster_center(cluster):
    xz = 0.
    yz = 0.
    z = 0.
    for hit in cluster:
        xz += hit[1]*hit[4]
        yz += hit[1]*hit[5]
        z += hit[1]
    xm = float(xz/z)
    ym = float(yz/z)
    #print(cluster)
    #print(xm, ym)
    #text = input()
    return [z, xm, ym]

def preprocess_single_file(config, f_name, hist=None):
    print('Reading file: ', f_name,'...')
    hit_list = read_raw_hits(config['bin_fpath']+f_name)
    n_hit = len(hit_list)
    if n_hit:
        print('Preprocessed '+str(n_hit)+' hits')
    else:
        print('No events found after preselection!\t'+str(f_name[:19])+'.npz')
    hit_list_buf = list(filter(lambda hit: hit[-1] == 2, hit_list))
    hit_list_filtered = list(filter(lambda hit: hit[1] > 300, hit_list_buf)) 
   #print(hit_list_1fr)
    zone_id = config['zone_id']
    
    single_evt_hits_list = []
    evt_list = []
    cluster_list = []
    cluster_coor_list = []
    if hit_list_filtered:
        evt_id = hit_list_filtered[0][0]
        for hit in hit_list_filtered:
            if hit[0] != evt_id:
                if(single_evt_hits_list):
                    evt_list.append(single_evt_hits_list)
                single_evt_hits_list = []
                evt_id = hit[0]
            if hit[1] > 0:
                coors = get_xy(hit[2], zone_id)
                if coors[0] >=0:
                    hit = hit + coors
                    single_evt_hits_list.append(hit)
        for evt in evt_list:
            #print('evt id: ',evt_id,'\n',evt)
            some_clusters = find_clusters(evt) #TODO: Include some info about the average activity at the sensitive area: if high, do not take these events.
            if len(some_clusters):
                cluster_list += some_clusters
        #print('***Clusters***')
        for cluster in cluster_list:
        #    print(f'id {idx+1:d}:\n',cluster)
            if len(cluster)>1:
                cl_coors = get_cluster_center(cluster)
                #print(f'COG coordinates: xm={cl_coors[0]:2.2f}\tym={cl_coors[1]:2.2f}\tz={cl_coors[2]:8.0f}\n')
                if len(cluster)==2:
                    hist.Fill(cl_coors[1], cl_coors[2])
                cluster_coor_list.append([cluster[0][0]]+cl_coors)
            else:
                #hist.Fill(cluster[0][4], cluster[0][5])
                cluster_coor_list.append([cluster[0][0],cluster[0][1], float(cluster[0][4]),float(cluster[0][5])])
        #print(f'Evt_count={len(cluster_coor_list):5d}\n')
        #for cluster in cluster_coor_list:
        #    print(cluster)
        out_file = open('/storage/pol_eff_measurement/'+f_name[:19]+'_clusters.npy', 'wb')
        np.save(out_file, np.array(cluster_coor_list), allow_pickle=True)
        out_file.close()

def preprocess(config, regex_line, offline = False):
    print('Caution: you are using clustering algorithm!')
    attempt_count = 0
    #fig, ax = init_monitor_figure()
    #plt.draw()
    c1 = TCanvas( 'c1', 'Monitor', 2560,800)
    c1.SetFillColor( 42 )
    c1.GetFrame().SetFillColor( 21 )
    c1.GetFrame().SetBorderSize( 6 )
    c1.GetFrame().SetBorderMode( -1 )
    hist = TH2F( 'monitor', 'monitor',640, 0, 32, 400, 0, 20)
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
                preprocess_single_file(config, f_name, hist)
                f_name_old = f_name
                attempt_count = 0
                file_count +=1
                hist.Draw('colz')
                c1.Modified()
                c1.Update()
                text = input()
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
    with open(os.getcwd()+'/../'+str('pol_config.yml'), 'r') as conf_file:
        try:
            config = yaml.load(conf_file, Loader=yaml.Loader)
            config['bin_fpath'] = '/home/kudryavtsev/POLARIMETER/polarimeter_readout/run_view/2022-02-09_GEMPolarimeter/'
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
           

