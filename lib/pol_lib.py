import glob
import numpy as np 
import numpy.ma as ma
from numba import jit

import time
from datetime import datetime
from depol.depolarizer import *
from mapping import get_xy
import math

@jit(nopython=True)
def convert_val(x):
    x += ((x&0x2000)>>13)*0xFFFFC000
    if x>>31 :
        return int((1<<32)-x)
    else:
        return -x

@jit(nopython=True)
def translate_word(word):
    chip   = (word >> 27) & 0x1F
    if chip >= 20:
        val  = -1
        trig = (word>>24 & 0x07)
    else:
        val  = convert_val(int(word & 0x3FFF))
        trig = 0
    ch     = (word >> 14) & 0x3F
    fr     = (word >> 20) & 0x7F
    glb_ch = (chip*64+ch + 640) % 1280
    return trig, val, glb_ch, chip, fr

def load_events(fname, n_evt=None):
    word_arr = np.fromfile(fname, dtype=np.uint32)
    if n_evt:
        word_arr = word_arr[:n_evt]
    data = np.empty((word_arr.shape[0], 4), dtype=np.int32)
    return make_events(word_arr, data)

@jit(nopython=True)
def make_events(word_arr, data):
    curr_pol = 0
    n = 0
    for word in word_arr:
        pol, val, ch, chip, fr = translate_word(word)
        if fr < 3 and curr_pol > 0 and val>0:
            data[n] = [curr_pol-3, val, ch, fr] #curr_pol = +2, +4 => curr_pol-3 = -1, +1
            n += 1
        if pol != 0:
            curr_pol = pol
    data = data[:n]
    return data
    
#    word_arr = np.fromfile(fname, dtype=np.uint32)
#    if n_evt:
#        word_arr = word_arr[:n_evt]
#    curr_pol = 0
#
#    ''' # 5.7 sec
#    data = np.empty((word_arr.shape[0], 4), dtype=np.int32)
#    n = 0
#    for word in word_arr:
#        pol, val, ch, chip, fr = translate_word(word)
#        if fr < 3 and curr_pol > 0 and val>0:
#            data[n] = [curr_pol-3, val, ch, fr] #curr_pol = +2, +4 => curr_pol-3 = -1, +1
#            n += 1
#        if pol != 0:
#            curr_pol = pol
#    data = data[:n]
#    return data

    ''' # 6.3 sec
    data = []
    for word in word_arr:
        pol, val, ch, chip, fr = translate_word(word)
        if curr_pol > 0 and fr < 3 :
            data.append([curr_pol-3, val, ch, fr]) #curr_pol = +2, +4 => curr_pol-3 = -1, +1
        if pol != 0:
            curr_pol = pol
    d = np.array(data)
    print("dtype:", d.dtype)
    return d
#    '''

def mask_ch_map(hist, mask):
    buf = np.array(hist)
    if mask != None:
        for [y,x] in mask:
            buf[y,x] = 0
        return buf
    else: 
        return hist

def get_mean(x, h):
    x_mid = (x[1:] + x[:-1])/2
    pdf = h /np.sum(h)
    x_m = np.sum(x_mid*pdf)
    return x_m 

def get_stdev(x, h):
    x_mid = (x[1:] + x[:-1])/2
    pdf = h /np.sum(h)
    x_m = np.sum(x_mid*pdf)
    x2_m = np.sum(np.power(x_mid, 2) * pdf)
    return np.sqrt(x2_m - x_m**2)

def get_coor_grid():
    return {'xs': np.linspace(-64,64,num=33),
            'ys': np.linspace(-20,20, num=21),
            'xc': np.linspace(-32,32,num=33),
            'yc': np.linspace(-10,10, num=21)}

def arrange_region(h_l, h_r , x, lim=[None,None]):
    buf_l = h_l[:, np.where(x==lim[0])[0][0]:np.where(x==lim[1])[0][0]]
    buf_r = h_r[:, np.where(x==lim[0])[0][0]:np.where(x==lim[1])[0][0]]
    buf_x = x[np.where(x==lim[0])[0][0]:np.where(x==lim[1])[0][0]+1]
    return buf_l, buf_r, buf_x
    
def get_hist_asymmetry(h_dict, x_dir=None):
    hist_l = h_dict['hc_l']
    hist_r = h_dict['hc_r']
    h_sum = hist_l+hist_r
    (idy,idx) = np.unravel_index(np.argmax(h_sum, axis=None), h_sum.shape)
    #print('max bin', idy, idx)
    A_list = []
    dA_list =[]
    N_list = []
    if idy:
        for hist in [hist_l, hist_r]:
            if x_dir:
                n_up = np.sum(np.sum(hist[idx+1:,:]))
                n_down = np.sum(np.sum(hist[:idx:,:]))
            else:
                n_up = np.sum(np.sum(hist[idy+1:,:]))
                n_down = np.sum(np.sum(hist[:idy:,:]))
            dn_up = np.sqrt(n_up)
            dn_down = np.sqrt(n_down)
            A_list.append((n_up-n_down)/(n_up + n_down+1))
            dfdn_up = 2*n_up/(n_up+n_down)**2
            dfdn_down = 2*n_down/(n_up+n_down)**2
            dA_list.append(np.sqrt((dfdn_up*dn_up)**2 + (dfdn_down*dn_down)**2))
            N_list.append(np.sum(np.sum(hist)))

        A = (A_list[0]*N_list[0]-A_list[1]*N_list[1])/(N_list[0]+N_list[1])
        #A = (A_list[0]/N_list[0]-A_list[1]/N_list[1])
        dA = np.sqrt(dA_list[0]**2 + dA_list[1]**2)
    else:
        print('Unable to find asymmetry!')
        A=-999
        dA = -999

    return A, dA 

#def get_hist_asymmetry(hist,x=None): #Deprecated version: isn't stable for left and right 
#    (idy,idx) = np.unravel_index(np.argmax(hist, axis=None), hist.shape)
#    if idy:
#        if x:
#            n_up = np.sum(np.sum(hist[idx+1:,:]))
#            n_down = np.sum(np.sum(hist[:idx:,:]))
#        else:
#            n_up = np.sum(np.sum(hist[idy+1:,:]))
#            n_down = np.sum(np.sum(hist[:idy:,:]))
#        dn_up = np.sqrt(n_up)
#        dn_down = np.sqrt(n_down)
#        A = (n_up-n_down)/(n_up + n_down + 1)
#        dfdn_up = 2*n_up/(n_up+n_down)**2
#        dfdn_down = 2*n_down/(n_up+n_down)**2
#        dA = np.sqrt((dfdn_up*dn_up)**2 + (dfdn_down*dn_down)**2)
#    else:
#        print('Unable to find asymmetry!')
#        A=-999
#        dA = -999
#    return A, dA 



def get_raw_stats(h_dict, t, stats={}):
    stats.update({'count_time' : t})
    hist_l = h_dict['hc_l']
    hist_r = h_dict['hc_r']
    xc = h_dict['xc']
    yc = h_dict['yc']
    for hist, pol_type in zip([hist_l, hist_r],['l','r']):
        hprof_x = np.sum(hist, axis=0)
        hprof_y = np.sum(hist, axis=1)
        n_evt = np.sum(hprof_x)
        mx = get_mean(xc, hprof_x)
        my = get_mean(yc, hprof_y)
        sx = get_stdev(xc, hprof_x)
        sy = get_stdev(yc, hprof_y)
        stats.update({'n_evt_'+pol_type : n_evt,
                      'mx_'+pol_type : mx,
                      'my_'+pol_type : my,
                      'sx_'+pol_type : sx,
                      'sy_'+pol_type : sy})
    Ay, dAy = get_hist_asymmetry(h_dict)
    Ax, dAx = get_hist_asymmetry(h_dict, x_dir=True)
    stats.update({'Ay' : Ay, 'dAy' : dAy, 'Ax' : Ax, 'dAx' : dAx})
    return stats

def calc_asymmetry(h_dict):
    hist_l = h_dict['hc_l']
    hist_r = h_dict['hc_r']
    yc = h_dict['yc']
    y_mid = (yc[1:] + yc[:-1])/2
    hprof_yl = np.sum(hist_l, axis=1)
    hprof_yr = np.sum(hist_r, axis=1)
    my_l = get_mean(yc, hprof_yl)
    my_r = get_mean(yc, hprof_yr)
    my = (my_l + my_r)/2.

    sum_up = np.sum(hprof_yl[y_mid>my])+np.sum(hprof_yr[y_mid<=my])
    sum_down = np.sum(hprof_yl[y_mid<=my])+np.sum(hprof_yr[y_mid>my])
    A = (sum_up - sum_down)/(sum_up + sum_down)
    print('*** Asymmetry: {:2.2f} % ***'.format(A*100.))
    return A

def get_hist_asymmetry(h_dict, x_dir=None):
    hist_l = h_dict['hc_l']
    hist_r = h_dict['hc_r']
    h_sum = hist_l+hist_r
    (idy,idx) = np.unravel_index(np.argmax(h_sum, axis=None), h_sum.shape)
    #print('max bin', idy, idx)
    A_list = []
    dA_list =[]
    N_list = []
    if idy:
        for hist in [hist_l, hist_r]:
            if x_dir:
                n_up = np.sum(np.sum(hist[idx+1:,:]))
                n_down = np.sum(np.sum(hist[:idx:,:]))
            else:
                n_up = np.sum(np.sum(hist[idy+1:,:]))
                n_down = np.sum(np.sum(hist[:idy:,:]))
            dn_up = np.sqrt(n_up)
            dn_down = np.sqrt(n_down)
            A_list.append((n_up-n_down)/(n_up + n_down+1))
            dfdn_up = 2*n_up/(n_up+n_down)**2
            dfdn_down = 2*n_down/(n_up+n_down)**2
            dA_list.append(np.sqrt((dfdn_up*dn_up)**2 + (dfdn_down*dn_down)**2))
            N_list.append(np.sum(np.sum(hist)))
        
        A = (A_list[0]*N_list[0]-A_list[1]*N_list[1])/(N_list[0]+N_list[1])
        #A = (A_list[0]/N_list[0]-A_list[1]/N_list[1])
        dA = np.sqrt(dA_list[0]**2 + dA_list[1]**2)
    else:
        print('Unable to find asymmetry!')
        A=-999
        dA = -999
        
    return A, dA 

def print_stats(stats):
    nl = stats['n_evt_l']
    nr = stats['n_evt_r']
    T = stats['count_time']
    print('{:─^95}'.format("") )
    print('{:<20} {:^17.1f} seconds'.format("Count time:", T))
    print('{:<20} {:^17} {:^17} {:^17} {:^17}'.format("Raw stats", "DIFF", "LEFT", "RIGHT", "SUM"))
    print('{:─^95}'.format("") )
    print('{:<20} {:>7} ± {:<7} {:^17} {:^17} {:^17}'.format("Event number", int(nl-nr), int(math.sqrt(nl+nr)), int(nl), int(nr), int(nl+nr) ))
    print('{:<20} {:>7.3} ± {:<7.3}%'.format("", (nl-nr)*2.0/(nl+nr)*100., math.sqrt(nl+nr)*2.0/(nl+nr)*100. ))
    print('{:<20} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f} kHz'.format("Hit rate [kHz]", 
                                                                                          (nl-nr)/T*1e-3, math.sqrt(nl+nr)/T*1e-3,
                                                                                          nl/T*1e-3, math.sqrt(nl)/T*1e-3, 
                                                                                          nr/T*1e-3, math.sqrt(nr)/T*1e-3, 
                                                                                          (nl+nr)/T*1e-3, math.sqrt(nl+nr)/T*1e-3
                                                                                          ))
    print('{:<20} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f}'.
          format("<x> [mm]", 
                 0.5*(stats['mx_l']-stats['mx_r']), math.hypot(stats['sx_r']/math.sqrt(nr), stats['sx_l']/math.sqrt(nl) )/math.sqrt(2.0),
                 stats['mx_l'], stats['sx_l']/math.sqrt(nl),
                 stats['mx_r'], stats['sx_r']/math.sqrt(nr),
                 0.5*(stats['mx_r']+stats['mx_l']), math.hypot(stats['sx_r']/math.sqrt(nr), stats['sx_l']/math.sqrt(nl) )/math.sqrt(2.0)
            )
          )
    print('{:<20} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f} {:>7.3f} ± {:<7.3f}'.
          format("<y> [mm]", 
                 0.5*(stats['my_l']-stats['my_r']), math.hypot(stats['sy_r']/math.sqrt(nr), stats['sy_l']/math.sqrt(nl) )/math.sqrt(2.0),
                 stats['my_l'], stats['sy_l']/math.sqrt(nl),
                 stats['my_r'], stats['sy_r']/math.sqrt(nr),
                 0.5*(stats['my_r']+stats['my_l']), math.hypot(stats['sy_r']/math.sqrt(nr), stats['sy_l']/math.sqrt(nl) )/math.sqrt(2.0)
            )
          )

    print('{:<20} {:>7.3f} ± {:<7.3f}%'. format("Asym x", stats['Ax']*100, stats['dAx']*100))
    print('{:<20} {:>7.3f} ± {:<7.3f}%'. format("Asym y", stats['Ay']*100, stats['dAy']*100))

def load_hist(hist_fpath, fname):
    h_dict = dict(np.load(hist_fpath+fname, allow_pickle=True))
    return h_dict

def load_files(file_path, regex_filename):
    filenames = np.sort(np.array(glob.glob1(file_path , regex_filename)))
    fname = np.load(file_path+filenames[0], allow_pickle=True)
    h_l = fname['hc_l']
    h_r = fname['hc_r']
    x = fname['xc']
    y = fname['yc']
    for filename in filenames[1:]:
        print('Reading file: ', filename)
        fname = np.load(file_path+filename, allow_pickle=True)
        h_l += fname['hc_l']
        h_l += fname['hc_r']
    h_l, h_r, x = arrange_fit_region(h_l, h_r ,x ,lim = [-XRANGE,XRANGE])
    return h_l, h_r, x, y, filenames[0], filenames[-1]


def accum_data(h_dict1, h_dict2):
    buf_dict = h_dict1
    buf_dict['hc_l'] = h_dict1['hc_l'] + h_dict2['hc_l']
    buf_dict['hc_r'] = h_dict1['hc_r'] + h_dict2['hc_r']
    buf_dict['hs_l'] = h_dict1['hs_l'] + h_dict2['hs_l']
    buf_dict['hs_r'] = h_dict1['hs_r'] + h_dict2['hs_r']
    return buf_dict


def write2file(out_file, fname, fitres, stats):
    data = {}
    data.update({'date/time' : fname[:19]})
    data.update(stats)
    fitres = {'chi2'	  : fitres.fval,
            'ndf'		  : 1280 - fitres.npar,
            'E [MeV]'	  : fitres.values['E'],
            'L [mm]'	  : fitres.values['L'],
            'V [rel.u]'   : np.sqrt(1.-fitres.values['Ksi']**2),
            'dV [rel.u]'  : fitres.errors['Ksi']/(np.sqrt(1.-fitres.values['Ksi']**2)),
            'Q [rel.u]'   : fitres.values['Ksi'],
            'dQ [rel.u]'  : fitres.errors['Ksi'],
            'phi_l [rad]' :fitres.values['phi_lin'],
            'd phi_l [rad]':fitres.errors['phi_lin'],
            'P [rel.u]'   : fitres.values['P'],
            'dP [rel.u]'  : fitres.errors['P'],
            'mx [mm]'	  : fitres.values['mx'],
            'dmx [mm]'	  : fitres.errors['mx'],
            'my [mm]'	  : fitres.values['my'],
            'dmy [mm]'	  : fitres.errors['my'],
            'sx [mm]'	  : fitres.values['sx'],
            'dsx [mm]'	  : fitres.errors['sx'],
            'sy [mm]'	  : fitres.values['sy'],
            'dsy [mm]'	  : fitres.errors['sy'],
            'mx2 [mm]'	   : fitres.values['mx2'],
            'dmx2 [mm]'    : fitres.errors['mx2'],
            'my2 [mm]'	   : fitres.values['my2'],
            'dmy2 [mm]'    : fitres.errors['my2'],
            'sx2 [mm]'	   : fitres.values['sx2'],
            'dsx2 [mm]'    : fitres.errors['sx2'],
            'sy2 [mm]'	   : fitres.values['sy2'],
            'dsy2 [mm]'    : fitres.errors['sy2'],
            'N_grel [rel.u]': fitres.errors['N_grel'],
            'NL'		  : fitres.values['NL'],
            'dNL'		  : fitres.errors['NL'],
            'NR'		  : fitres.values['NR'],
            'dNR'		  :fitres.errors['NR']}

    data.update(fitres)
    out_file = '/mnt/vepp4/kadrs/lsrp.txt'
    with open(out_file, 'w') as the_file:
        for key, value in data.items():
            if type(value) is str:
                the_file.write('{0:20s}\t\t#{1:s}\n'.format(value, key))
            else:
                the_file.write('{0:>6.6f}\t\t#{1:s}\n'.format(value, key))
                
def write2file_(the_file, fname, fitres, dfreq, counter, moments):
    with open(the_file, 'a') as the_file:
        if counter%10==0:
            the_file.write('#{:>3s}\t{:>9s}\t'.format('cnt', 'utime'))
            for i in ['P',          'P_err',    'Ksi',      'Ksi_err',
                      'V',          'V_err',    'phi_lin',  'phi_lin_err',
                      'dip_amp',    'dip_ang',  'quad_amp', 'quad_ang',
                      'fft1_amp',   'fft1_ang', 'fft2_amp', 'fft2_ang',
                      'gross_moments',          'gross_moments_err']:
                the_file.write('{:>16s}\t'.format(i))
            the_file.write('#{:20s}\n'.format('date'))
        the_file.write('{0:>3d}\t'.format(counter))
        unix_time = time.mktime(datetime.strptime(fname[:19], '%Y-%m-%dT%H:%M:%S').timetuple())
        the_file.write('{0:>.0f}\t'.format(unix_time))
    # 		freq = float(Depol.get_frequency())
    # 		energy = 0
    # 		if freq!= 0:
    # 			energy = (9+freq/818924.)*440.648
    # 		the_file.write('\t{0:>10.3f}\t{1:>10.3f}\t'.format(freq,energy))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(fitres.values['P'], fitres.errors['P']))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(fitres.values['Ksi'], fitres.errors['Ksi']))
        V = np.sqrt(1.-fitres.values['Ksi']**2)
        V_err = fitres.errors['Ksi']/(np.sqrt(1.-fitres.values['Ksi']**2))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(V, V_err))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(fitres.values['phi_lin'], fitres.errors['phi_lin'] ))
  
        for i in ['dip_amp',  'dip_ang',  'quad_amp', 'quad_ang',
                  'fft1_amp', 'fft1_ang', 'fft2_amp', 'fft2_ang',
                  'gross_moments', 'gross_moments_err']:
            the_file.write('{:>10.6f}\t'.format(moments[i]))

        the_file.write('#{0:20s}\n'.format(fname[:19]))
        
def write2file_nik(the_file, fname, fitres, par_list, raw_stats, counter, moments,normchi2):
    freq = par_list[0]
    vepp4E_nmr = par_list[1]
    with open(the_file, 'a') as the_file:
#        if counter%10==0:
#            the_file.write('#{:>3s}\t{:>9s}\t'.format('cnt', 'utime'))
#            for i in ['P',          'P_err',    'Q',      'Q_err',
#                      'V',          'V_err',    'beta',  'beta_err', 'chi2',
#                      'dip_amp',    'dip_ang',  'quad_amp', 'quad_ang',
#                      'fft1_amp',   'fft1_ang', 'fft2_amp', 'fft2_ang',
#                      'gross_moments',          'gross_moments_err']:
#                the_file.write('{:>16s}\t'.format(i))
#            the_file.write('#{:20s}\n'.format('date'))
        the_file.write('{0:>3d}\t'.format(counter))
        unix_time = time.mktime(datetime.strptime(fname[:19], '%Y-%m-%dT%H:%M:%S').timetuple())
        the_file.write('{0:>.0f}\t'.format(unix_time))
        energy = 0
        if freq!= 0:
            #ATTENTION temporary solution --- Dendrofecal method is applied !!!!!
            vepp4E=4730
            n = int(vepp4E/440.648)
            energy = (n+freq/818924.)*440.648
        the_file.write('\t{0:>10.3f}\t{1:>10.3f}\t{2:>10.3f}\t'.format(freq,energy, vepp4E_nmr))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(fitres.values['P'], fitres.errors['P']))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(fitres.values['Q'], fitres.errors['Q']))
        V = np.sqrt(1.-fitres.values['Q']**2)
        V_err = fitres.errors['Q']/(np.sqrt(1.-fitres.values['Q']**2))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(V, V_err))
        the_file.write('{0:>10.6f}\t{1:>10.6f}\t'.format(fitres.values['beta'], fitres.errors['beta'] ))
        the_file.write('{0:>10.6f}\t'.format(normchi2))
  
        for i in ['dip_amp',  'dip_ang',  'quad_amp', 'quad_ang',
                  'fft1_amp', 'fft1_ang', 'fft2_amp', 'fft2_ang',
                  'gross_moments', 'gross_moments_err']:
            the_file.write('{:>10.6f}\t'.format(moments[i]))
        
        for key in ['n_evt_l', 'n_evt_r', 'mx_l', 'mx_r', 'my_l', 'my_r', 'sx_l', 'sx_r', 'sy_l', 'sy_r']:
            the_file.write('{:>10.6f}\t'.format(raw_stats[key]))

        the_file.write('#{0:20s}\n'.format(fname[:19]))
        
def mask_hist(config, h_dict):
    mask = config['mask']
    hc_l = np.array(h_dict['hc_l'])
    hc_r = np.array(h_dict['hc_r'])
    if mask is not None:
        for [y,x] in mask:
            idx = int((x+31)/2)
            idy = int(y+9.5) 
            hc_l[idy, idx] = 0
            hc_r[idy, idx] = 0
        buf_dict = h_dict
        buf_dict['hc_l'] = hc_l
        buf_dict['hc_r'] = hc_r
        return buf_dict
    else:
        return h_dict
        
@jit(nopython=True)
def impute_ch(x, y, hc_l, hc_r):
    n_evt_center_l = 0
    n_evt_center_r = 0
    if x > -1:
        x_arr = np.arange(x-1, x+2)
        y_arr = np.arange(y-1, y+2)
        x_arr = x_arr[np.logical_and(x_arr >= 0, x_arr <32)]
        y_arr = y_arr[np.logical_and(y_arr >= 0, y_arr <20)]
        #print('center: ',x,y)
        count = 0
        for idx in x_arr:
            for idy in y_arr:
                #print('side ch: ',idx,idy)
                n_evt_center_l += hc_l[idy,idx]
                n_evt_center_r += hc_r[idy,idx]
                count += 1
        if count:
            n_evt_center_l /= count
            n_evt_center_r /= count
    return int(n_evt_center_l), int(n_evt_center_r)
    
def impute_hist(config, h_dict):
    raw_ch_arr = config['broken_ch']
    hc_l = np.array(h_dict['hc_l'])
    hc_r = np.array(h_dict['hc_r'])
    if raw_ch_arr is not None:
        for raw_ch in raw_ch_arr:
            [x,y] = get_xy((raw_ch+640)%1280, config['zone_id'])
            n_l, n_r = impute_ch(x, y, hc_l, hc_r)
            hc_l[y,x] = n_l
            hc_r[y,x] = n_r
        buf_dict = h_dict
        buf_dict['hc_l'] = hc_l
        buf_dict['hc_r'] = hc_r
        return buf_dict


def read_vepp4_stap():
    path2stap = '/mnt/vepp4/kadrs/stap.dat'
    with open(path2stap, 'r', encoding='UTF-8') as stap:
        lines = stap.readlines()
        if len(lines) > 13:
            vepp4E = float(lines[13])/100.
        else:
            print('ERROR: stap file is empty!')
            vepp4E = -1
        return vepp4E
        
def get_par_from_file(path2file, par_id_arr):
    with open(path2file, 'r', encoding='UTF-8') as data:
        lines = data.readlines()
        data = []
        if len(lines) > 0:
            for par_id in par_id_arr:
                data.append(lines[par_id])
        else:
            data.append("-1")
            print('ERROR: stap file is empty!')
        return data
        
def init_depol():
    d = depolarizer('vepp4-spin',9090)
    print("Depolarizer is ", ("ON" if d.is_on() else "OFF" ))
    print("Attenuation ", d.get_attenuation())
    print("Scanning speed {:.3f} Hz".format(d.get_speed()))
    print("Frequency step {:.3f} Hz".format(d.get_step()))
    print("Initial frequency {:.3f} Hz".format(d.get_initial()))
    print("State: ",d.get_state())
    #d.start_fmap()
    return d

def get_depol_params(device, fname):
    t = datetime.datetime.strptime(fname[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
    depol_state = device.get_frequency_by_time(t)
    if len(depol_state) > 0:
        state = depol_state[-1]
        return [ (state.timestamp*1e-9), state.frequency, state.attenuation, state.speed ]

    else:
        return [0, 0, 0, 0]

   # fmap = device.get_fmap()
   # 
   # depol_pair = []
   # if len(fmap) > 0:
   #     idx = -1
   #     for i, depol_pair in enumerate(fmap):
   #         #print(datetime.fromtimestamp(depol_pair[0]*1e-9), depol_pair[0]*1e-9)
   #         if (depol_pair[0]*1e-9 - ts) > 0:
   #             idx = i
   #             break
   #     if idx >= 0:
   #         depol_pair = fmap[idx]
   #     else:
   #         print('No frequency for corresponding time found!')
   # att = float(device.get_attenuation())
   # fspeed = float(device.get_speed())
   # if len(depol_pair):
   #     return [depol_pair[0], depol_pair[1], att, fspeed]
    #else:
   #     return [0, 0, fspeed, att]
        
def guess_real_energy(v4E, vepp4H_nmr):
    v4E_nmr = 1.04277*vepp4H_nmr
    if v4E_nmr < 1000:
        return (v4E - 9.2)/1.0053
    else:
        return v4E_nmr



