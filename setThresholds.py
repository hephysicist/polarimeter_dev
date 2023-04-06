#!/usr/bin/python3

import xmlrpc.client
import os
import datetime
import time
import socket
import sys
import json


input_line = input('Use unique thresholds? [Y/N]: ')
if (input_line == 'Y'):
    print('Using unique thresholds.')
    th_fname = input('Enter threshold filename: ')
    use_uniq = True
elif (input_line == 'N'):
    print('Using single threshold value.')
    th_fname = 'thresholds_2022-04-26T15:24:40_nsigma-6p0.txt'
    th_val = int(input('Enter threshold value: '))
    use_uniq = False
else:
    print('Invalid option: use letters Y or N \n Exiting the programm')
    sys.exit()
threshold_value = 160

if len(sys.argv)>1:
    threshold_value=int(sys.argv[1])

SPI_MASK=0xFFFF
#SPI_MASK=0x1F1F
#SPI_RANGE=[0,1,2,3,4, 8,9,10,11,12]
SPI_RANGE=[0,1,2,3,4, 8,9,10,11,12]

with xmlrpc.client.ServerProxy("http://de10_nano:8080/RPC2") as proxy:

    MEM_SETVAL = threshold_value #160 #33
    MEM_SETVAL_UNREACHABLE = 0xFFFF
    MEM_ADDR   = 0x80000
    ADDR_BASE  = (MEM_ADDR>>16)|0x8000|0x2000

    thresholds = []
    full_fname = '/home/lsrp/pol_calibration/thresholds/'+th_fname
    with open(full_fname, 'r') as filehandle:
        thresholds = json.load(filehandle)
    bad_channels = []
    #bad_channels300 = [0, 1, 64, 65, 128, 129, 192, 193, 256, 257, 258, 290, 320, 321, 448, 449, 641, 642, 704, 705, 769, 832, 833, 1025, 1048, 1052]
    #bad_channels = [640,258,32,834, 1025, 1026,1048,1052,672]
    #bad_channels = [833,832,192,769,193,1,641,129,1025, 672, 743, 674]
    #bad_channels = [1052,1051,348,347,1100,260,833,832,192,769,193,1058] #2022-1012
    bad_channels = [1052,1051,1100,347,348,260,832, 833,192,193,769,1,129,899,898,386] #2023-03-09
    for chan, threshold in enumerate([int(i) for i in thresholds]):
        chan = (chan + 640)%1280 # Legacy
        mask = (1 << SPI_RANGE[ int(chan/128) ])
        proxy.write_reg(mask, 0x90, chan % 128)
        board_id = int(chan/128)
        if threshold > 1 and chan not in bad_channels:
            if use_uniq:
                proxy.write_reg(mask, 0x92, threshold)
                print('setting ch {:d} threshold value: {:d}'.format (chan, threshold))
            else:
                proxy.write_reg(mask, 0x92, th_val)
                print('setting ch {:d} threshold value: {:d}'.format (chan, th_val))
            proxy.write_reg(mask, 0x91, ADDR_BASE)
            proxy.write_reg(mask, 0x91, 0)
        else:
            proxy.write_reg(mask, 0x92, MEM_SETVAL_UNREACHABLE)
            proxy.write_reg(mask, 0x91, ADDR_BASE)
            proxy.write_reg(mask, 0x91, 0)
            print('setting ch {:d} threshold value: {:d}'.format (chan, int(MEM_SETVAL_UNREACHABLE)))
