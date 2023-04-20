import scipy.signal as sg
import scipy.interpolate as interp
import scipy.optimize as opt
import numpy as np
from math import pi, exp, sqrt, atan2, sin, cos, acos


def make_gauss(rate_max=1, x_0=15.5, y_0=9.5, sigma_x=3.0, sigma_y=4.0, tilt=0, xc=32, yc=20):
    data = []
    for _y in range(yc):
        data.append([])
        for _x in range(xc):
            x = (_x - x_0) / sigma_x
            y = (_y - y_0) / sigma_y
            data[_y].append(rate_max*exp(-0.5*(x**2 + y**2 + tilt*x*y)))
    return np.array(data)

gauss = make_gauss()

def get_moments(h_dict):
    hc_l = h_dict['hc_l']
    hc_r = h_dict['hc_r']
    xc = h_dict['xc']
    yc = h_dict['yc']
    array_shape = hc_r.shape
    heatmap = {"diff" : hc_l - hc_r/hc_r.sum()*hc_l.sum(), 
               "sum"  : hc_l + hc_r/hc_r.sum()*hc_l.sum()}

    conv = sg.convolve(gauss, heatmap["sum"], mode="same")

    y_max, x_max = np.unravel_index(conv.argmax(), conv.shape)

    if conv[y_max-5:y_max+6,x_max-5:x_max+6].sum() < 2e6:
        return {k : 0 for k in ["dip_amp",  "dip_ang", "quad_amp",  "quad_ang",
                                "fft1_amp", "fft1_ang", "fft2_amp", "fft2_ang",
                                "gross_moments", "gross_moments_err"]}

    spline = interp.bisplrep(yc[y_max-5:y_max+6].repeat(5+6),
                             np.concatenate((xc[x_max-5:x_max+6],)*(5+6)),
                             conv[y_max-5:y_max+6,x_max-5:x_max+6],
                             s=0,)
    res = opt.minimize(lambda x: -interp.bisplev([x[0]], [x[1]], spline),
                       [yc[y_max], xc[x_max]], method='COBYLA',
                       options={'rhobeg' : 0.4, 'disp': False})

    xy_0 = res.x
    #print("x_0, y_0 by conv. after opt.:", xy_0[1], xy_0[0])

    charge = 0.
    total = heatmap["sum"].sum()
    #print(heatmap["diff"].sum())
    r_max = min(abs(xy_0[1] - xc[0]), abs(xy_0[1] - xc[-1]),
                abs(xy_0[0] - yc[0]), abs(xy_0[0] - yc[-1]))
    #print("r_max:", r_max)

    it = np.nditer(heatmap["diff"], flags=['multi_index'])
    n = 0
    while not it.finished:
        q = it[0]
        x = xc[it.multi_index[1]] - xy_0[1] + 1.0
        y = yc[it.multi_index[0]] - xy_0[0] + 0.5
        r2 = x**2. + y**2.
        if sqrt(r2)<=r_max:
            charge += q*r2
            n += 1
        it.iternext()

    #print("charge:", charge, "n:", n)
    it = np.nditer(heatmap["diff"] - charge/n**2*pi, flags=['multi_index'])
    charge = 0.
    dipole = [0.,0.]
    quad = np.zeros([3,3])
    polar = np.zeros(36)

    while not it.finished:
        q = it[0]
    #    print("idx:", it.multi_index)
        x = xc[it.multi_index[1]] - xy_0[1] + 1.0
        y = yc[it.multi_index[0]] - xy_0[0] + 0.5
        r2 = x**2. + y**2.
        if sqrt(r2)<=r_max:
#        if abs(x)<=r_max and abs(y)<=r_max: # квадрат
            dipole[0] += x*q
            dipole[1] += y*q
            charge += q*r2
            quad[0,0] += q*(3.*x**2 - r2)
            quad[0,1] += q*3.*x*y
            quad[1,0] += q*3.*x*y
            quad[1,1] += q*(3.*y**2 - r2)
            quad[2,2] += -q*r2
            polar[int(atan2(y, x)/pi*18+18)] += q
        it.iternext()
     
    
    dipole_amp = np.hypot(*dipole)/total                # = 2*q*R
    dipole_ang = atan2(dipole[1], dipole[0])
    quad_amp = 0.5*sqrt(4*quad[0,0]**2 + quad[0,1]**2)# /total
    quad_ang = 0.5*acos(quad[0,0]/quad_amp)

    fft = np.fft.rfft(polar)

    print("Moments:\n\tMonopole:\t{:.2f}".format(charge))
    print("\tDipole:\t\tamp = {:.2f},\tangle = {:.1f} deg.".format(dipole_amp, dipole_ang/pi*180))
    print("\tQuadrupole:\tamp = {:.2f},\tangle = {:.1f} deg.".format(quad_amp/10000, quad_ang/pi*180))
    print("\tQuadrupole tensor:\n", quad, "\n")

    print("FFTs:")
    print("\t1st (dipole):\t\tamp = {:.2f},\tangle = {:.1f} deg.".format(abs(fft[1])/total*10000,
                                                                         atan2(fft[1].imag, fft[1].real)/pi*180/(-2)))
    print("\t2nd (quadrupole):\tamp = {:.2f},\tangle = {:.1f} deg.".format(abs(fft[2])/total*10000,
                                                                         atan2(fft[2].imag, fft[2].real)/pi*90*(-1)))

    h_diff = np.absolute(hc_l-hc_r)
    gross_moments = np.sum(np.sum(h_diff))/np.sum(np.sum(hc_l+hc_r))
    gross_moments_err = 2./np.sqrt(np.sum(np.sum(hc_l+hc_r)))
    print('Gross moments:\n \tM={:>1.4f} ± {:1.4f}'.format(gross_moments, gross_moments_err))

    return {"dip_amp" : dipole_amp, 
           "dip_ang"  : dipole_ang/pi*180,
           "quad_amp" : quad_amp/10000, 
           "quad_ang" : quad_ang/pi*180,
           "fft1_amp" : abs(fft[1])/total*10000,
           "fft1_ang" : -atan2(fft[1].imag, fft[1].real)/pi*90,
           "fft2_amp" : abs(fft[2])/total*10000,
           "fft2_ang" : -atan2(fft[2].imag, fft[2].real)/pi*90,
           "gross_moments" :     gross_moments,
           "gross_moments_err" : gross_moments_err}

