import numpy as np
from math import pi, sqrt, exp, cos, sin
from iminuit.util import make_func_code
from scipy import signal
import scipy.stats
import cupy as cp
import cusignal
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter, maximum_filter, generic_filter


def wrap_array(x_mid, n_p):
    step = x_mid[1]-x_mid[0]
    x_left = np.linspace(x_mid[0]-n_p*step,x_mid[0], num=n_p, endpoint=False)
    x_right = np.linspace(x_mid[-1]+step, x_mid[-1]+(n_p+1)*step, num=n_p, endpoint=False)
    return np.concatenate((x_left, x_mid, x_right))

def make_central_coord(x,y):
    x_mid = (x[1:] + x[:-1])/2
    y_mid = (y[1:] + y[:-1])/2
    return [x_mid, y_mid]

def general_blur(h_dict):
    hc_l = np.array(h_dict['hc_l'])
    hc_r = np.array(h_dict['hc_r'])
    zer_y, zer_x = np.where(hc_l==0)
    for x, y in zip(zer_x, zer_y):
        idx1 = x-1
        idx2 = x+2
        idy1 = y-1
        idy2 = y+2

        if idx1 < 0:
            idx1 = 0
        elif idy1 < 0:
            idy1 = 0
        elif idx2 > np.shape(hc_l)[1]:
            idx2 = np.shape(hc_l)[1]
        elif idy2 > np.shape(hc_l)[0]:
            idy2 = np.shape(hc_l)[0]
        else:
            pass

        buf_l = hc_l[idy1:idy2,idx1:idx2].flatten()
        buf_r = hc_r[idy1:idy2,idx1:idx2].flatten()
        buf_l = buf_l[buf_l>1]
        buf_r = buf_r[buf_r>1]
        if np.shape(buf_l)[0]:
            hc_l[y,x] = np.mean(buf_l)
            hc_r[y,x] = np.mean(buf_r)
        else:
            hc_l[y,x] = 0
            hc_r[y,x] = 0
    filter_ = gaussian_filter
    hc_l = filter_(hc_l, sigma=0.5, mode='nearest')
    hc_r = filter_(hc_r, sigma=0.5, mode='nearest')
    res = h_dict
    res['hc_l']=hc_l
    res['hc_r']=hc_r
    return res

#Make blur among only nonzero pixels
def blur_nonzero_pixels(h,radius):
	result = np.array(h)
	Nx, Ny = h.shape
	for x in range(0, Nx):
		for y in range(0, Ny):
			if h[x,y] > 0: 
				window = np.array(h[max(0,x-radius):min(x+radius+1, Nx), max(0,y-radius):min(y+radius+1, Ny)])
				result[x,y] = np.sum(np.where(window>0, window, 0.))/np.sum(np.where(window>0, 1, 0.))
			else:
				result[x,y]=0
				
	return result



def nonzero_blur(h_dict):
    hc_l = np.array(h_dict['hc_l'])
    hc_r = np.array(h_dict['hc_r'])
    hc_l = blur_nonzero_pixels(hc_l,1)
    hc_r = blur_nonzero_pixels(hc_r,1)
    res = h_dict
    res['hc_l']=hc_l
    res['hc_r']=hc_r
    return res



def blur_zero_pixels(h,radius):
	result = np.array(h)
	Ny, Nx= h.shape
	for y in range(0, Ny):
		for x in range(0, Nx):
			if np.abs(h[y,x]) < 1:
				window = np.array(h[max(0,y-radius):min(y+radius+1, Ny), max(0,x-radius):min(x+radius+1, Nx)])
				s = np.sum(np.where(window>0, window, 0.))
				n = np.sum(np.where(window>0, 1, 0.))
				result[y,x] =  s/n
	return result



def zero_blur(h_dict):
    print("zero_blur")
    hc_l = np.array(h_dict['hc_l']).copy()
    hc_r = np.array(h_dict['hc_r']).copy()
    hc_l = blur_zero_pixels(hc_l,1)
    hc_r = blur_zero_pixels(hc_r,1)
    res = h_dict
    res['hc_l']=hc_l
    res['hc_r']=hc_r
    return res


def make_blur_window(h, y, x, radius):
	Ny, Nx= h.shape
	return np.array( h[ max(0, y-radius) : min(y+radius+1, Ny), max(0,x-radius) : min(x+radius+1, Nx)] )

def blur_pixels_with_large_deviation(h,radius=1, nsigma=3):
    Ny, Nx= h.shape
    fltr = np.array(h)
    for y in range(0, Ny):
        for x in range(0, Nx):
            window = make_blur_window(h, y, x, radius)
            mean  = np.average( window[window>0] )
            sigma = np.std( window[window>0] )
            if  np.abs(h[y,x]-mean) > nsigma*sigma or np.abs(h[y,x])<1:
                #print (y,x, h[y,x], mean, sigma)
                fltr[y,x] = 1
            else:
                fltr[y,x] = 0
    result = np.zeros(h.shape)
    for y in range(0, Ny):
        for x in range(0, Nx):
            if fltr[y,x]==1:
                window = make_blur_window(h, y, x, radius)
                f =      make_blur_window(fltr, y,x, radius)
                mean = np.average( window[f==0] )
                #print( y, x, h[y,x], mean)
                result[y,x] = mean
            else:
                result[y,x] = h[y,x]
    return result

def large_deviation_blur(h_dict):
	res = h_dict.copy()
	res['hc_l'] = blur_pixels_with_large_deviation( np.array(h_dict['hc_l']), 1)
	res['hc_r'] = blur_pixels_with_large_deviation( np.array(h_dict['hc_r']), 1)
	return res



def FWHM(X,Y):
    half_max = max(Y) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #plot(X[0:len(d)],d) #if you are interested
    #find the left and right most indexes
    left = np.argwhere(d > 0)
    right = np.argwhere(d < 0)
    if len(left)>0: 
        left_idx = left[0]
    else:
        left_idx = 0
    if len(right)>0: 
        right_idx = right[0]
    else:
        right_idx = -1
    return X[right_idx] - X[left_idx] #return the difference (full width)

def FWHM2D(x,y,data):
    py = np.sum(data, axis=1)
    px = np.sum(data, axis=0)
    return ( FWHM(x,px), FWHM(y,py) )

def print_pol_stats(fitter):
    print('{:─^95}'.format("  Fit result  "))
    print("chi2/ndf = {:.3f} / {} = {:.3f} ".format(fitter.chi2, fitter.ndf, fitter.chi2/fitter.ndf), end='')
    prob = (1.0-scipy.stats.chi2.cdf(fitter.chi2, fitter.ndf))
    if  prob<1e-4 :
        print("{:>20} = {:<.5e}".format('prob(chi2,dnf)', prob))
    else:
        print("{:>20} = {:<.5f}".format('prob(chi2,dnf)',prob))
    minuit = fitter.minuit
    print(' P ={:>7.3f} ± {:1.3f}'.format(minuit.values['P'],minuit.errors['P']))
    print(' Q ={:>7.3f} ± {:1.3f}'.format(minuit.values['Q'],minuit.errors['Q']))
    print(' V ={:>7.3f} ± {:1.3f}'.format(minuit.values['V'],minuit.errors['V']))
    print(''.rjust(95,'─'))
