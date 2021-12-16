import glob
import numpy as np 
import numpy.ma as ma
from numba import jit

import time
from datetime import datetime

@jit(nopython=True)
def convert_val(x):
    x = x+((x&0x2000)>>13)*0xFFFFC000
    if x>>31 :
        x=int((1<<32)-x)
        return x
    else:
        return -x


@jit(nopython=True)
def translate_word(word):
    buf    = int(word & 0x3FFF)
    val    = convert_val(buf)
    ch     = (word >> 14) & 0x3F
    fr     = (word >> 20) & 0x7F
    chip   = (word >> 27) & 0x1F
    glb_ch = (chip*64+ch + 640) % 1280
    pol = 0
    if chip < 20:
        out = val
    elif chip >= 20:
        val = -1
        pol = (word>>24 & 0x07)
    else:
        val = -1
    return pol, val, glb_ch, chip, fr

def load_events(fname, n_evt=None):
    data = []
    input_file = open(fname, 'rb')
    word_arr = np.fromfile(input_file, dtype=np.uint32)
    curr_pol = 0
    if n_evt:
        word_arr = word_arr[:n_evt]
    for word in word_arr:
        pol, val, ch, chip, fr = translate_word(word)
        if curr_pol > 0 and fr < 3 and ch!=412:
            data.append([curr_pol-3, val, ch, fr]) #curr_pol = +2, +4 => curr_pol-4 = -1, +1
        if pol != 0:
            curr_pol = pol
    return np.array(data)

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

def get_raw_stats(h_dict):

    hist_l = h_dict['hc_l']
    hist_r = h_dict['hc_r']
    xc = h_dict['xc']
    yc = h_dict['yc']
    stats = {}
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

def print_stats(stats):
    print('*** Raw stats ***')
    print('mx_l: {:2.2f}\tmx_r: {:2.2f}\tmy_l: {:2.2f}\tmy_r: {:2.2f}'.format(stats['mx_l'],
                                                                              stats['mx_r'],
                                                                              stats['my_l'], 
                                                                              stats['my_r']))
    print('sx_l: {:2.2f}\tsx_r: {:2.2f}\tsy_l: {:2.2f}\t sy_r: {:2.2f}'.format(stats['sx_l'],
                                                                              stats['sx_r'],
                                                                              stats['sy_l'], 
                                                                              stats['sy_r']))
    n_l = stats['n_evt_l']
    n_r = stats['n_evt_r']
    diff = n_l - n_r
    rel_diff = 2.*diff/(n_l+n_r)*100
    print('n_evt_l: {0} \tn_evt_r: {1}\tdiff: {2}\t({3:3.2f}%)'.format(n_l,n_r,diff, rel_diff))

def print_pol_stats(fitres):
    print('*** Polarization ***')
    print('P={:>6.2f} ± {:1.2f}'.format(fitres.values['P'],fitres.errors['P']))
    print('Q={:>6.2f} ± {:1.2f}'.format(fitres.values['Ksi'],fitres.errors['Ksi']))
    print('V={:>6.2f} ± {:1.2f}'.format(np.sqrt(1-fitres.values['Ksi']**2),fitres.errors['Ksi']/np.sqrt(1-fitres.values['Ksi']**2)))

def print_pol_stats_nik(fitres):
    print('*** Polarization ***')
    print('P={:>6.2f} ± {:1.2f}'.format(fitres.values['P'],fitres.errors['P']))
    print('Q={:>6.2f} ± {:1.2f}'.format(fitres.values['Q'],fitres.errors['Q']))
    print('V={:>6.2f} ± {:1.2f}'.format(np.sqrt(1-fitres.values['Q']**2),fitres.errors['Q']/np.sqrt(1-fitres.values['Q']**2)))

def load_hist(hist_fpath, fname):
    print('Reading histogram file: ', fname)
    h_dict = np.load(hist_fpath+fname, allow_pickle=True)
    return h_dict

def load_files(file_path, regex_filename):
    filenames = np.sort(np.array(glob.glob1(file_path , regex_filename)))
    file = np.load(file_path+filenames[0], allow_pickle=True)
    h_l = file['hc_l']
    h_r = file['hc_r']
    x = file['xc']
    y = file['yc']
    for filename in filenames[1:]:
        print('Reading file: ', filename)
        file = np.load(file_path+filename, allow_pickle=True)
        h_l += file['hc_l']
        h_l += file['hc_r']
    h_l, h_r, x = arrange_fit_region(h_l, h_r ,x ,lim = [-XRANGE,XRANGE])
    return h_l, h_r, x, y, filenames[0], filenames[-1]


def accum_data(h_dict1, h_dict2):
    buf_dict = {'hc_l': h_dict1['hc_l']+h_dict2['hc_l'],
                'hc_r': h_dict1['hc_r']+h_dict2['hc_r'],
                'hs_l': h_dict1['hs_l']+h_dict2['hs_l'],
                'hs_r': h_dict1['hs_r']+h_dict2['hs_r'],
                'xs': h_dict1['xs'],
                'ys': h_dict1['ys'],
                'xc': h_dict1['xc'],
                'yc': h_dict1['yc'],
                'vepp4E': h_dict1['vepp4E']}
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
                
def write2file_(the_file, fname, fitres, counter, moments):
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
        
def write2file_nik(the_file, fname, fitres, counter, moments,normchi2):
    with open(the_file, 'a') as the_file:
        if counter%10==0:
            the_file.write('#{:>3s}\t{:>9s}\t'.format('cnt', 'utime'))
            for i in ['P',          'P_err',    'Q',      'Q_err',
                      'V',          'V_err',    'beta',  'beta_err', 'chi2',
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

        the_file.write('#{0:20s}\n'.format(fname[:19]))
        
def mask_hist(config, h_dict):
    mask = config['mask']
    hc_l = np.array(h_dict['hc_l'])
    hc_r = np.array(h_dict['hc_r'])
    if mask is not None:
        for [y,x] in mask:
            idx = int((x+32)/2)
            idy = y+10 
            hc_l[idy, idx] = 0
            hc_r[idy, idx] = 0
        buf_dict = {'hc_l': hc_l,
                'hc_r': hc_r,
                'hs_l': h_dict['hs_l'],
                'hs_r': h_dict['hs_r'],
                'xs': h_dict['xs'],
                'ys': h_dict['ys'],
                'xc': h_dict['xc'],
                'yc': h_dict['yc'],
                'vepp4E': h_dict1['vepp4E']}
        return buf_dict
    else:
        return h_dict
    
    
def read_vepp4_stap():
    path2stap = '/mnt/vepp4/kadrs/stap.dat'
    with open(path2stap, 'r', encoding='UTF-8') as stap:
        lines = stap.readlines()
        if len(lines) > 0:
            vepp4E = float(lines[13])/100.
            print(vepp4E)
        else:
            print('ERROR: stap file is empty!')
            vepp4E = -1
        return vepp4E


