import numpy as np
from math import pi, sqrt, exp
from iminuit.util import make_func_code
from scipy import signal
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

#def crystall_ball(x, y, sx, sy, z0, n, N):
#    z = np.sqrt(x*x/(sx*sx) + 1./(sy*sy)*y*y)
#    if abs(z) < z0:
#        return 1.0/(2*pi*sx*sy)* np.exp(- 0.5*z**2.);
#    else:
#        return (n/np.abs(z0))**n * exp(-0.5*z0*z0)*np.power( n/z0 - z0  - z/sx)
#    return 

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
24 

def get_fit_func_(X, mx, sx, my, sy, mx2, sx2, my2, sy2, N_grel, Ksi, phi_lin, P, V, E, L,alpha):
    x_mid = X[0]-mx
    y_mid = X[1]-my

    n_sp_x = 12
    n_sp_y = 20

    x_wrapped = wrap_array(x_mid,n_sp_x)
    y_wrapped = wrap_array(y_mid,n_sp_y)
    xx,yy = np.meshgrid(x_wrapped, y_wrapped)

    x_sec = get_xsec_xy(xx, yy, Ksi, phi_lin, P, V, E, L,alpha)
    core = double_gausn2d3(xx, yy,  sx,  sy, mx2, sx2, my2, sy2, N_grel)
    res = signal.fftconvolve(x_sec, core, mode = 'same')
    res = res[n_sp_y:-n_sp_y, n_sp_x:-n_sp_x]
    return res


def get_fit_func(x, y, par, inverse_pol=False):
	x_mid = (x[1:] + x[:-1])/2
	y_mid = (y[1:] + y[:-1])/2
	if inverse_pol:
		par[9] = - par[9]
		par[12] = - par[12]
	X = [x_mid,y_mid]
	res = get_fit_func_(X, 
						mx	   = par[0],
						sx	   = par[1],
						my	   = par[2],
						sy	   = par[3],
						mx2    = par[4],
						sx2    = par[5],
						my2    = par[6],
						sy2    = par[7],
						N_grel = par[8],
						Ksi    = par[9],
						phi_lin= par[10],
						P	   = par[11],
						V	   = par[12],
						E	   = par[13],
						L	   = par[14],
                        alpha  = par[15])
	return res

# class Chi2_2d:
# 	def __init__(self, model, x, z_l, z_r):
# 		self.model = model	# model predicts z for given x()
# 		self.x = x
# 		self.z_l = z_l.flatten()
# 		self.z_r = z_r.flatten()
# 		self.z_l_err = np.where(self.z_l > 0, np.sqrt(np.abs(self.z_l)),-1)
# 		self.z_r_err = np.where(self.z_r > 0, np.sqrt(np.abs(self.z_r)),-1)
# 		self.func_code = make_func_code(fit_varnames)

# 	def __call__(self, *par):  
# 		par_l = np.array(par[:15])
# 		#par_l[12] = np.sqrt(1.-par_l[9]**2)
		
# 		zm_l = par[15] * self.model(self.x, *par_l).flatten()
		
# 		par_r = np.array(par[:15])
# 		par_r[9] = -par_r[9]
# 		#par_r[12] = -np.sqrt(1.-par_r[9]**2)
# 		par_r[12]= -par_r[12]
		
# 		zm_r = par[16] * self.model(self.x, *par_r).flatten()
# 		#print(self.z_l[self.z_l>20000],self.z_r[self.z_r>20000])
# 		chi2_l = np.sum(np.where(self.z_l > 0, 
# 									  np.power((self.z_l-zm_l),2)/np.power(self.z_l_err,2),
# 									  0))
# 		chi2_r = np.sum(np.where(self.z_r > 0, 
# 									  np.power((self.z_r-zm_r),2)/np.power(self.z_r_err,2),
# 									  0))
# 		print(chi2_l+chi2_r)
# 		return chi2_l+chi2_r

# class Chi2_2d:
# 	def __init__(self, model, x, z_l, z_r):
# 		self.model = model	# model predicts z for given x()
# 		self.x = x
# 		self.z_ly = np.sum(z_l, axis = 1)
# 		self.z_ry = np.sum(z_r, axis = 1)
# 		self.z_lx = np.sum(z_l, axis = 0)
# 		self.z_rx = np.sum(z_r, axis = 0)
# 		self.func_code = make_func_code(fit_varnames)

# 	def __call__(self, *par):  
# 		par_l = np.array(par[:15])
# 		#par_l[12] = np.sqrt(1.-par_l[9]**2)
# 		par_r = np.array(par[:15])
# 		par_r[9] = -par_r[9]
# 		#par_r[12] = -np.sqrt(1.-par_r[9]**2)
# 		par_r[12]= -par_r[12]
		
# 		zm_ly = par[15] * np.sum(self.model(self.x, *par_l), axis=1)
# 		zm_ry = par[16] * np.sum(self.model(self.x, *par_r), axis=1)
		
# 		zm_lx = par[15] * np.sum(self.model(self.x, *par_l), axis=0)
# 		zm_rx = par[16] * np.sum(self.model(self.x, *par_r), axis=0)
		
# 		diff2_lx = np.power((self.z_lx-zm_lx),2)/np.absolute(self.z_lx)
# 		diff2_rx = np.power((self.z_rx-zm_rx),2)/np.absolute(self.z_rx)
# 		#print(diff2_lx)
# 		chi2_lx = np.sum(diff2_lx[self.z_lx>0])
# 		chi2_rx = np.sum(diff2_rx[self.z_rx>0])
		
# 		#print(self.z_lx[self.z_lx>0]-zm_lx[self.z_lx>0])  
# 		diff2_ly = np.power((self.z_ly-zm_ly),2)/self.z_ly
# 		diff2_ry = np.power((self.z_ry-zm_ry),2)/self.z_ry
# 		chi2_ly = np.sum(diff2_ly[self.z_ly>0])
# 		chi2_ry = np.sum(diff2_ry[self.z_ry>0])
# 		#print(chi2_lx+chi2_rx + chi2_ly+chi2_ry)
# 		return chi2_lx+chi2_rx + chi2_ly+chi2_ry

#Try to fit the 2d differenece. Result is bad. Minuit unable to detetermine correct normalization.
class Chi2_2d:
    def __init__(self, model, x, z_l, z_r, tied_VQ = False):
        self.model = model	# model predicts z for given x()
        self.x = x
        self.z_l = z_l.flatten()
        self.z_r = z_r.flatten()
        fit_varnames = ['mx','sx','my','sy','mx2','sx2','my2','sy2','N_grel','Ksi','phi_lin','P','V','E','L','alpha','NL','NR']
        self.tied_VQ = tied_VQ
        self.func_code = make_func_code(fit_varnames)

    def __call__(self, *par):  
        NL = par[16]
        NR = par[17]
        Q  = par[9]
        V  = np.sqrt(1-Q*Q) if self.tied_VQ else par[12]
        
        par_l = np.array(par[:16])
        par_l[9] = Q
        par_l[12] = V

        par_r = np.array(par[:16])
        par_r[9] = -Q
        par_r[12]= -V

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
        #print(chi2)
        #print(self.x)
        return chi2
    
def make_blur(h_dict):
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
    hc_l = filter_(hc_l, sigma=0.8, mode='nearest')
    hc_r = filter_(hc_r, sigma=0.8, mode='nearest')
    return {'hc_l': hc_l,
                'hc_r': hc_r,
                'hs_l': h_dict['hs_l'],
                'hs_r': h_dict['hs_r'],
                'xs': h_dict['xs'],
                'ys': h_dict['ys'],
                'xc': h_dict['xc'],
                'yc': h_dict['yc']}


def nik_blur(h,radius):
	result = np.array(h)
	Nx, Ny = h.shape
	for x in range(0, Nx):
		for y in range(0, Ny):
			if h[x,y] > 0: 
				window = np.array(h[max(0,x-radius):min(x+radius+1, Nx), max(0,y-radius):min(y+radius+1, Ny)])
				#nx,ny = window.shape
				#result[x,y] = np.sum(window)/(nx*ny)
				result[x,y] = np.sum(np.where(window>0, window, 0.))/np.sum(np.where(window>0, 1, 0.))
			else:
				result[x,y]=0
				
	return result

def make_blur_nik(h_dict):
	hc_l = np.array(h_dict['hc_l'])
	hc_r = np.array(h_dict['hc_r'])
	#hc_l = mask_ch_map(hc_l, mask)
	#hc_r = mask_ch_map(hc_r, mask)
	hc_l = nik_blur(hc_l,1)
	hc_r = nik_blur(hc_r,1)
	return {'hc_l': hc_l,
				'hc_r': hc_r,
				'hs_l': h_dict['hs_l'],
				'hs_r': h_dict['hs_r'],
				'xs': h_dict['xs'],
				'ys': h_dict['ys'],
				'xc': h_dict['xc'],
				'yc': h_dict['yc']}
