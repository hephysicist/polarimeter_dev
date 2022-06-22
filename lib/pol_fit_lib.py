import numpy as np
from math import pi, sqrt, exp, cos, sin
from iminuit.util import make_func_code
from scipy import signal
import cupy as cp
import cusignal
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter, maximum_filter, generic_filter

def double_gausn2d(x, y, mx, sx, my, sy, mx2, sx2, my2, sy2, N):
    return (1./(2*pi*sx*sy) * np.exp(-np.power((x-mx),2)/(2.*sx**2)-np.power((y-my),2)/(2.*sy**2))+
            N/(2*pi*sx2*sy2) * np.exp(-np.power((x-mx2),2)/(2.*sx2**2)-np.power((y-my2),2)/(2.*sy2**2)))

def double_gausn2d2(x, y, sx, sy, mx2, sx2, my2, sy2, N):
    return (1./(2*pi*sx*sy) * np.exp(-np.power(x,2.)/(2.*sx**2.)-np.power(y,2.)/(2.*sy**2))+
            N/(2*pi*sx2*sy2) * np.exp(-np.power((x-mx2),2.)/(2.*sx2**2.)-np.power((y-my2),2.)/(2.*sy2**2)))/(1.+ N)

def double_gausn2d3(x, y, sx, sy, mx2, sx2, my2, sy2, N):
    return (1./(2*pi*sx*sy) * np.exp(-np.power(x,2.)/(2.*sx**2.)-np.power(y,2.)/(2.*sy**2))+
            N/(2*pi*np.sqrt(sx**2+sx2**2)*np.sqrt(sy**2+sy2**2)) * np.exp(-np.power(x,2.)/(2.*(sx**2+sx2**2.))-np.power(y,2.)/(2.*(sy**2 +sy2**2))))/(1.+ N)


def crystall_ball_1d(x, alpha, n1, n2):
    #helper params
    na1 = n1/alpha
    na2 = n2/alpha
    return np.where( x < alpha, 
                     np.where( x < -alpha,  
                               np.power(np.abs(na1/(na1+alpha-x)), n1) * np.exp(-0.5*alpha**2.),
                               np.exp(-0.5*x**2.) ),
                     np.power(np.abs(na2/(na2-alpha+x)), n2) * np.exp(-0.5*alpha**2.))

def crystall_ball_1d_cp(x, alpha, n1, n2):
    #helper params
    na1 = n1/alpha
    na2 = n2/alpha
    return np.where( x < alpha, 
                     cp.where( x < -alpha,  
                               cp.power((na1/(na1+alpha-x)), n1) * cp.exp(-0.5*alpha**2.),
                               cp.exp(-0.5*x**2.) ),
                     cp.power((na2/(na2-alpha+x)), n2) * cp.exp(-0.5*alpha**2.))

def crystall_ball_2d(x, y, sx,sy, alpha_x1,  alpha_x2, alpha_y1, alpha_y2, nx1, nx2, ny1, ny2, phi, p1, p2,p3):
    X =  x/sx*cos(phi) + y/sy*sin(phi)
    Y = -x/sx*sin(phi) + y/sy*cos(phi)
    Z =  np.sqrt(X*X+Y*Y)
    c = X/Z
    s = Y/Z
    #n     = np.sqrt( np.power(nx1*c - nx2, 2.0) + np.power(ny1*s - ny2, 2.0) )
    n     = np.sqrt( np.power(nx1*(c*cos(p2)+s*sin(p2)) - nx2, 2.0) + np.power(ny1*(s*cos(p2)-c*sin(p2)) - ny2, 2.0) )
    #alpha = np.sqrt( np.power(alpha_x1*c -alpha_x2, 2.0) + np.power(alpha_y1*s- alpha_y2, 2.0) )
    alpha = np.sqrt( np.power(alpha_x1*(c*cos(p1)+s*sin(p1)) -alpha_x2, 2.0) + np.power(alpha_y1*(s*cos(p1)-c*sin(p1))- alpha_y2, 2.0) )
    #return crystall_ball_1d(Z, alpha, n, n)*(1.0 + p1*y + p2*y*y + p3*np.power(y,3.0))
    return crystall_ball_1d(Z, alpha, n, n)

def crystall_ball_2d_cp(x, y, sx,sy, alpha_x1,  alpha_x2, alpha_y1, alpha_y2, nx1, nx2, ny1, ny2, phi, p1, p2,p3):
    X =  x/sx*cos(phi) + y/sy*sin(phi)
    Y = -x/sx*sin(phi) + y/sy*cos(phi)
    Z =  cp.sqrt(X*X+Y*Y)
    c = X/Z
    s = Y/Z
    n     = cp.sqrt( cp.power(nx1*c - nx2, 2.0) + cp.power(ny1*s - ny2, 2.0) )
    alpha = cp.sqrt( cp.power(alpha_x1*c -alpha_x2, 2.0) + cp.power(alpha_y1*s- alpha_y2, 2.0) )
    return crystall_ball_1d_cp(Z, alpha, n, n)*(1.0 + p1*y + p2*y*y + p3*cp.power(y,3.0))

def wrap_array(x_mid, n_p):
    step = x_mid[1]-x_mid[0]
    x_left = np.linspace(x_mid[0]-n_p*step,x_mid[0], num=n_p, endpoint=False)
    x_right = np.linspace(x_mid[-1]+step, x_mid[-1]+(n_p+1)*step, num=n_p, endpoint=False)
    return np.concatenate((x_left, x_mid, x_right))

def get_xsec_thetaphi(theta, phi, Ksi=0, phi_lin=0., P=0, V=0):
    m_e = 0.511*10**6   #eV
    gamma = 4730./0.511 
    omega_0 = 2.35 #eV
    re2 = 2.81794*10**-15
    
    kappa = 4.*gamma*omega_0/m_e
    eta = gamma*theta
    eta2 = np.power(eta,2)

    dsigma = 1/np.power((1+eta2+kappa),2)*(
    1 + 0.5*(kappa**2)/((1.+eta2+kappa)*(1.+eta2)) -
    - 2.*eta2/(1.+eta2)/(1.+eta2) * (1 - Ksi * np.cos(2.*(phi-phi_lin)))+
    + P*V*eta*kappa/(1+eta2+kappa)/(1.+eta2)*np.sin(phi))
    return dsigma

def get_xsec_xy(x,y, Ksi=0, phi_lin=0., P=0, V=0, E=4730, L=33000, alpha=0):
    m_e = 0.511*10**6   #eV
    gamma = E/0.511 
    omega_0 = 2.35 #eV
    
    r = np.hypot(x,y)
    phi = np.arctan2(y, x)
    theta = r/L
    
    
    kappa = 4.*gamma*omega_0/m_e
    eta = gamma*theta
    eta2 = np.power(eta,2)
    
    dsigma = 1/np.power((1+eta2+kappa),2)*(
    1 + 0.5*(kappa**2)/((1.+eta2+kappa)*(1.+eta2)) -
    - 2.*eta2/(1.+eta2)/(1.+eta2) * (1 - Ksi * np.cos(2.*(phi-phi_lin)))+
    + P*V*eta*kappa/(1+eta2+kappa)/(1.+eta2)*np.sin(phi-alpha))
    return dsigma

def get_xsec_xy_cp(x,y, Ksi=0, phi_lin=0., P=0, V=0, E=4730, L=33000, alpha=0):
    m_e = 0.511*10**6   #eV
    gamma = E/0.511 
    omega_0 = 2.35 #eV
    r = cp.hypot(x,y)
    phi = cp.arctan2(y, x)
    theta = r/L

    kappa = 4.*gamma*omega_0/m_e
    eta = gamma*theta
    eta2 = cp.power(eta,2)
    
    dsigma = 1/cp.power((1+eta2+kappa),2)*(
    1 + 0.5*(kappa**2)/((1.+eta2+kappa)*(1.+eta2)) -
    - 2.*eta2/(1.+eta2)/(1.+eta2) * (1 - Ksi * cp.cos(2.*(phi-phi_lin)))+
    + P*V*eta*kappa/(1+eta2+kappa)/(1.+eta2)*cp.sin(phi-alpha))
    return dsigma

def fit_func_gaus_(X, mx, sx, my, sy, mx2, sx2, my2, sy2, N_grel, Q, beta, P, V, E, L,alpha):
    x_mid = X[0]-mx
    y_mid = X[1]-my

    n_sp_x = 12
    n_sp_y = 20

    x_wrapped = wrap_array(x_mid,n_sp_x)
    y_wrapped = wrap_array(y_mid,n_sp_y)
    xx,yy = np.meshgrid(x_wrapped, y_wrapped)

    x_sec = get_xsec_xy(xx, yy, Q, beta, P, V, E, L,alpha)
    core = double_gausn2d3(xx, yy,  sx,  sy, mx2, sx2, my2, sy2, N_grel)
    res = signal.fftconvolve(x_sec, core, mode = 'same')
#    res = cp.asnumpy(cusignal.fftconvolve(cp.asarray(x_sec), cp.asarray(core), mode = 'same'))
    res = res[n_sp_y:-n_sp_y, n_sp_x:-n_sp_x]
    return res

def fit_func_gaus(x, y, par, inverse_pol=False):
	x_mid = (x[1:] + x[:-1])/2
	y_mid = (y[1:] + y[:-1])/2
	if inverse_pol:
		par[9] = - par[9]
		par[12] = - par[12]
	X = [x_mid,y_mid]
	res = fit_func_gaus_(X, 
						mx	   = par[0],
						sx	   = par[1],
						my	   = par[2],
						sy	   = par[3],
						mx2        = par[4],
						sx2        = par[5],
						my2        = par[6],
						sy2        = par[7],
						N_grel     = par[8],
						Q          = par[9],
						beta       = par[10],
						P	   = par[11],
						V	   = par[12],
						E	   = par[13],
						L	   = par[14],
                        alpha  = par[15])
	return res


def get_fit_func_(X, E, L, P, V, Q, beta, alpha_d,  mx, my, sx, sy, alpha_x1,alpha_x2,  alpha_y1, alpha_y2, nx1,nx2,ny1, ny2, phi, p1,p2,p3):
    x_mid = X[0]-mx
    y_mid = X[1]-my

    n_sp_x = 12
    n_sp_y = 20

    x_wrapped = wrap_array(x_mid,n_sp_x)
    y_wrapped = wrap_array(y_mid,n_sp_y)
    xx,yy = np.meshgrid(x_wrapped, y_wrapped)

    x_sec = get_xsec_xy(xx, yy, Q, beta, P, V, E, L,alpha_d)
    core = crystall_ball_2d(xx, yy,  sx,  sy, alpha_x1, alpha_x2,  alpha_y1, alpha_y2, nx1, nx2, ny1, ny2, phi, p1,p2,p3)
    res = signal.fftconvolve(x_sec, core, mode = 'same')
    res = res[n_sp_y:-n_sp_y, n_sp_x:-n_sp_x]
    return res


def get_fit_func(x, y, par, inverse_pol=False):
	x_mid = (x[1:] + x[:-1])/2
	y_mid = (y[1:] + y[:-1])/2
	X = [x_mid,y_mid]
	res = get_fit_func_(X, 
						E	    =  par[0],
						L	    =  par[1],
						P	    =  par[2],
						V	    = -par[3] if inverse_pol else par[3], 
						Q           = -par[4] if inverse_pol else par[4],
						beta        =  par[5],                        
						alpha_d     =  par[6],                        
						mx	    =  par[7],
						my	    =  par[8],
						sx	    =  par[9],
						sy	    =  par[10],
						alpha_x1 =  par[11],
						alpha_x2 =  par[12],
						alpha_y1 =  par[13],
						alpha_y2 =  par[14],
						nx1      =  par[15],
						nx2      =  par[16],
						ny1      =  par[17],
						ny2      =  par[18],
						phi      =  par[19],
						p1       =  par[20],
						p2       =  par[21],
						p3       =  par[22])
	return res

    
class Chi2:
    def __init__(self, model, x, z_l, z_r, tied_VQ = False):
        self.model = model	# model predicts z for given x()
        self.x = x
        self.z_l = z_l.flatten()
        self.z_r = z_r.flatten()
        fit_varnames  = list(model.__code__.co_varnames)[1:model.__code__.co_argcount]+['NL','NR']
        self.tied_VQ = tied_VQ
        self.func_code = make_func_code(fit_varnames)
        self.idxNL=fit_varnames.index('NL')
        self.idxNR=fit_varnames.index('NR')
        self.idxQ =fit_varnames.index('Q')
        self.idxV =fit_varnames.index('V')

    def __call__(self, *par):  
        NL = par[self.idxNL]
        NR = par[self.idxNR]
        Q  = par[self.idxQ]
        V  = np.sqrt(1.-Q*Q) if self.tied_VQ else par[self.idxV]
        
        par_l = np.array(par[:self.idxNL])
        par_l[self.idxQ] = Q
        par_l[self.idxV] = V

        par_r = np.array(par[:self.idxNL])
        par_r[self.idxQ] = -Q
        par_r[self.idxV]= -V

        zm_l = self.model(self.x, *par_l).flatten()
        zm_r = self.model(self.x, *par_r).flatten()
        zm  = zm_l - zm_r
        zm0 = NL*zm_l + NR*zm_r

        zd  = self.z_l/NL - self.z_r/NR
        zde = self.z_l/NL**2 + self.z_r/NR**2

        zd0 = self.z_l + self.z_r
        zde0 = self.z_l + self.z_r

        chi2_diff = np.sum(np.where(zde > 0,  np.power((zd-zm),2.)/zde, 0.))
        chi20     = np.sum(np.where(zde0 > 0, np.power((zd0-zm0),2.)/zde0, 0.))
        chi2 = chi20 + chi2_diff
        return chi2

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
	return {'hc_l': hc_l,
				'hc_r': hc_r,
				'hs_l': h_dict['hs_l'],
				'hs_r': h_dict['hs_r'],
				'xs': h_dict['xs'],
				'ys': h_dict['ys'],
				'xc': h_dict['xc'],
				'yc': h_dict['yc'],
				'vepp4E': h_dict['vepp4E'],
				'dfreq': h_dict['dfreq']}
				
				
