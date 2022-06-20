import numpy as np
from itertools import chain
from iminuit import Minuit
from iminuit.util import make_func_code
from pol_fit_lib import wrap_array
from scipy import signal
from math import cos, sin
from pol_plot_lib_new import *
from pol_lib import get_coor_grid

class FitResult:
    def __init__(self, d, e, f):
        self.data = d
        self.error = e
        self.fit = f

class FitMethod1:

    def __init__(self, x, z_l, z_r):
        self.x = x
        self.z_l = z_l.flatten()
        self.z_r = z_r.flatten()
        self.shape = np.shape(z_l)
        fit_varnames  = list(self.PDF.__code__.co_varnames)[1:self.PDF.__code__.co_argcount]+['NL','NR']
        self.inipars = dict.fromkeys(fit_varnames, 0.0)
        self.func_code = make_func_code(fit_varnames)
        self.minuit = Minuit(self, **self.inipars)
        self.minuit.print_level = 0
        self.minuit.errordef=1
        self.tied_VQ = False
        self.idxNL=fit_varnames.index('NL')
        self.idxNR=fit_varnames.index('NR')
        self.idxQ =fit_varnames.index('Q')
        self.idxV =fit_varnames.index('V')
        self.ndf = np.shape(self.x[0])[0]*np.shape(self.x[1])[0]


    def get_xsec_xy(self, x,y, Ksi=0, phi_lin=0., P=0, V=0, E=4730, L=33000, alpha=0):
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
        
    def crystall_ball_1d(self, x, alpha, n1, n2):
        #helper params
        na1 = n1/alpha
        na2 = n2/alpha
        return np.where( x < alpha, 
                         np.where( x < -alpha,  
                                   np.power(np.abs(na1/(na1+alpha-x)), n1) * np.exp(-0.5*alpha**2.),
                                   np.exp(-0.5*x**2.) ),
                         np.power(np.abs(na2/(na2-alpha+x)), n2) * np.exp(-0.5*alpha**2.))
        
    def crystall_ball_2d(self, x, y, sx,sy, alpha_x1,  alpha_x2, alpha_y1, alpha_y2, nx1, nx2, ny1, ny2, phi, p1, p2,p3):
        X =  x/sx*np.cos(phi) + y/sy*np.sin(phi)
        Y = -x/sx*np.sin(phi) + y/sy*np.cos(phi)
        Z =  np.sqrt(X*X+Y*Y)
        c = X/Z
        s = Y/Z
        #    n     = np.sqrt( np.power(nx1*c - nx2, 2.0) + np.power(ny1*s - ny2, 2.0) )
        #    alpha = np.sqrt( np.power(alpha_x1*c -alpha_x2, 2.0) + np.power(alpha_y1*(s)- alpha_y2, 2.0) )
        n     = np.sqrt( np.power(nx1*(c*cos(p2)+s*sin(p2)) - nx2, 2.0) + np.power(ny1*(s*cos(p2)-c*sin(p2)) - ny2, 2.0) )
        #alpha = np.sqrt( np.power(alpha_x1*c -alpha_x2, 2.0) + np.power(alpha_y1*s- alpha_y2, 2.0) )
        alpha = np.sqrt( np.power(alpha_x1*(c*cos(p1)+s*sin(p1)) -alpha_x2, 2.0) + np.power(alpha_y1*(s*cos(p1)-c*sin(p1))- alpha_y2, 2.0) )
        return self.crystall_ball_1d(Z, alpha, n, n)
        
    def PDF(self, E, L, P, V, Q, beta, alpha_d,  mx, my, sx, sy, alpha_x1,alpha_x2,  alpha_y1, alpha_y2, nx1,nx2,ny1, ny2, phi, p1,p2,p3):
        x_mid = self.x[0]-mx
        y_mid = self.x[1]-my

        n_sp_x = 12
        n_sp_y = 20

        x_wrapped = wrap_array(x_mid,n_sp_x)
        y_wrapped = wrap_array(y_mid,n_sp_y)
        xx,yy = np.meshgrid(x_wrapped, y_wrapped)

        x_sec = self.get_xsec_xy(xx, yy, Q, beta, P, V, E, L,alpha_d)
        core = self.crystall_ball_2d(xx, yy,  sx,  sy, alpha_x1, alpha_x2,  alpha_y1, alpha_y2, nx1, nx2, ny1, ny2, phi, p1,p2,p3)
        res = signal.fftconvolve(x_sec, core, mode = 'same')
        res = res[n_sp_y:-n_sp_y, n_sp_x:-n_sp_x]
        return res


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

        zm_l = self.PDF(*par_l).flatten()
        zm_r = self.PDF(*par_r).flatten()
        zm  = zm_l - zm_r
        zm0 = NL*zm_l + NR*zm_r

        zd  = self.z_l/NL - self.z_r/NR
        zde = self.z_l/NL**2 + self.z_r/NR**2

        zd0 = self.z_l + self.z_r
        zde0 = self.z_l + self.z_r
        
        self.data_left = self.z_l
        self.data_left_error = np.sqrt(self.z_l)
        self.data_right = self.z_r
        self.data_right_error = np.sqrt(self.z_r)
        
        self.fit_left = zm_l*NL
        self.fit_right = zm_r*NR

        #sum and difference for data
        self.data_sum  = self.data_left + self.data_right
        self.data_sum_error = np.sqrt(self.data_left + self.data_right)

        self.data_diff  = zd
        self.data_diff_error = np.sqrt(zde)

        #expected values
        self.fit_diff   = zm
        self.fit_sum    = zm0

        chi2_diff = np.sum(np.where(zde > 0,  np.power((zd-zm),2.)/zde, 0.))
        chi20     = np.sum(np.where(zde0 > 0, np.power((zd0-zm0),2.)/zde0, 0.))
        chi2 = chi20 + chi2_diff
        return chi2
        
    def get_fit_result(self, cfg):
        grids = get_coor_grid()
        coors = [grids['xc'],grids['yc']]
        shape = (np.shape(grids['yc'])[0]-1, np.shape(grids['xc'])[0]-1)
        data_field_names = [ 'data_sum', 'data_diff', 'data_left', 'data_right',
                        'fit_sum', 'fit_diff', 'fit_left', 'fit_right']
        data_error_names = ['data_sum_error', 'data_diff_error', 'data_left_error', 'data_right_error']
        for field_name in chain(data_field_names, data_error_names):
            setattr(self, field_name, getattr(self, field_name).reshape(shape))
        data_field_dict = {}
        for field_name in data_field_names:
            if 'data' in field_name:
                this_data_type = 'dat'
                this_data_err = getattr(self,field_name + '_error')
                this_data_err_px = np.sqrt(np.sum(this_data_err**2, axis=0))
                this_data_err_py = np.sqrt(np.sum(this_data_err**2, axis=1))
            else:
                this_data_type = 'fit'
                this_data_err = None
                this_data_err_px = None
                this_data_err_py = None
            data_field_dict[field_name] = data_field(   coors,
                                                        getattr(self, field_name),
                                                        data_err = this_data_err,
                                                        label=field_name,
                                                        data_type = this_data_type)
            data_field_dict[field_name+'_px'] = data_field(   [grids['xc'], None],
                                                        np.sum(getattr(self, field_name), axis=0),
                                                        data_err = this_data_err_px,
                                                        label= field_name+'_px',
                                                        data_type = this_data_type)
            data_field_dict[field_name+'_py'] = data_field(   [None, grids['yc']],
                                                        np.sum(getattr(self, field_name), axis=1),
                                                        data_err = this_data_err_py,
                                                        label=field_name+'_py',
                                                        data_type = this_data_type)
        return data_field_dict
        
    def fix(self, parlist):
        for parname in parlist:
            self.minuit.fixed[parname]=True

    def unfix(self, parlist):
        for parname in parlist:
            self.minuit.fixed[parname]=False
    
    def fit(self, cfg):
        par_ini = cfg['initial_values']
        par_err = cfg['par_err']
        par_lim = cfg['par_lim']
        par_fix = cfg['fix_par']
        for parname in par_ini.keys():
            self.minuit.values[parname] = par_ini[parname]
            self.minuit.fixed[parname]  = par_fix[parname]
            self.minuit.errors[parname] = par_err[parname]
            self.minuit.limits[parname] = par_lim[parname]
        #first fit is to find beam shape
        print(self.minuit)
        self.minuit.migrad()
        self.minuit.hesse()
        print(self.minuit)

        #second fit is to determine polarization. Beam parameters are fixed
        if not self.minuit.valid:
        #for name in  ['sx', 'sy', 'alpha_x1', 'alpha_x2', 'alpha_y1', 'alpha_y2', 'nx1','nx2', 'ny1','ny2', 'phi', 'p1', 'p2', 'p3']:  
            for name in  ['alpha_x2', 'alpha_y2', 'nx2', 'ny2', 'phi', 'p1', 'p2', 'p3']:
                self.minuit.fixed[name]=True
            self.minuit.migrad()
            self.minuit.hesse()
        print(self.minuit)
        return self.minuit

