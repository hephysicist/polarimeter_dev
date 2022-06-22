import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code
from pol_fit_lib import wrap_array
from scipy import signal
from math import cos,sin
from pol_lib import get_coor_grid
from pol_plot_lib import *
from itertools import chain


class FitMethod2:

    def __init__(self, x, z_l, z_r):
        self.x = x
        self.data_l = z_l.flatten()
        self.data_r = z_r.flatten()
        self.shape = np.shape(z_l)
        fit_varnames  = list(self.PDF.__code__.co_varnames)[1:self.PDF.__code__.co_argcount]+['psum', 'NL','NR']
        self.inipars = dict.fromkeys(fit_varnames, 0.0)
        self.func_code = make_func_code(fit_varnames)
        self.minuit = Minuit(self, **self.inipars)
        self.minuit.print_level = 0
        self.minuit.errordef=1

        self.tied_VQ = True
        self.idxNL=fit_varnames.index('NL')
        self.idxNR=fit_varnames.index('NR')
        self.idxQ =fit_varnames.index('Q')
        self.idxV =fit_varnames.index('V')
        self.idxpsum =fit_varnames.index('psum')
        self.ndf = np.shape(self.x[0])[0]*np.shape(self.x[1])[0]


    def ComptonPDF(self, x,y, E=4730., L=33000., P = 0., V = 0., Q=0., beta=0.):
        #E - energy, MeV
        #L - photon flight length in mm
        #P - vertical beam polarization
        #V - Stokes parameter of circular polarization
        #Q - Stokes parameter of total linear polarization Q= sqrt(1-V^2)
        #beta - angle of linear polarization
        me = 0.5109989461*10**6   #electorn mass eV
        g = E*1.0e6/me            #gamma factor
        o = 2.3526413364          #eV Initial laser photon 527 nm
        
        r = np.hypot(x,y)
        phi = np.arctan2(y, x)

        eta = g*r/L
        eta2 = np.power(eta,2.0)
        
        kappa = 4.*g*o/me

        t = 1. + eta2 + kappa
        
        sigma = 1./np.power(t,2.)*( 1. + 0.5*(kappa**2.)/(t*(1.+eta2)) -
                - 2.*eta2/(1.+eta2)/(1.+eta2) * (1.0 - Q * np.cos(2.*(phi-beta)))+
                + P*V*eta*kappa / t /( 1. + eta2 )*np.sin(phi)
            )
        return sigma

    def CrystalBall(self, x, a, n1, n2):
        #a  - point where gauss connect with power function
        #n1, n2 - power left and right
        #helper params
        na1 = n1/a
        na2 = n2/a
        return np.where( x < a, 
                         np.where( x < -a,  
                                   np.power(np.abs(na1/(na1-a-x)), n1) * np.exp(-0.5*a**2.),
                                   np.exp(-0.5*x**2.) ),
                                   np.power(np.abs(na2/(na2-a+x)), n2) * np.exp(-0.5*a**2.))

    
    def Oval(self, x, dx, y, dy, c, s):
        result =  np.sqrt( (x*(c-dx))**2  + (y*(s-dy))**2 )
        return result


    def Gaus(self, x, y, A, mx, my, sx, sy):
        return  A*np.exp(  - np.power( (x-mx)/sx, 2.0) - np.power( (y-my)/sy , 2.0) )


    def BeamPDF(self, x, y, sx,sy, ax,  dax, ay, day, nx, dnx, ny, dny, alpha_s, alpha_a, alpha_n):
        X =  x/sx*cos(alpha_s) + y/sy*sin(alpha_s)
        Y = -x/sx*sin(alpha_s) + y/sy*cos(alpha_s)
        Z =  np.sqrt(X*X+Y*Y)
        c = X/Z
        s = Y/Z
        a = self.Oval(ax,dax, ay,day, c*cos(alpha_a) + s*sin(alpha_a), s*cos(alpha_a) - c*sin(alpha_a))
        n = self.Oval(nx,dnx, ny,dny, c*cos(alpha_n) + s*sin(alpha_n), s*cos(alpha_n) - c*sin(alpha_n))
        return self.CrystalBall(Z, a, n, n)


    def PDF(self, E, L, P, V, Q, beta, alpha_d,  mx, my, sx, sy, ax, dax,  ay, day, nx, dnx, ny, dny, alpha_s, alpha_a, alpha_n):
        x_mid = self.x[0]-mx
        y_mid = self.x[1]-my

        n_sp_x = 12
        n_sp_y = 20

        x_wrapped = wrap_array(x_mid,n_sp_x)
        y_wrapped = wrap_array(y_mid,n_sp_y)
        xx,yy = np.meshgrid(x_wrapped, y_wrapped)

        x_sec = self.ComptonPDF(xx, yy, E, L, P, V, Q, beta)
        core = self.BeamPDF(xx, yy,  sx,  sy, ax, dax,  ay, day, nx, dnx, ny, dny, alpha_s, alpha_a, alpha_n)
        res = signal.fftconvolve(x_sec, core, mode = 'same')
        res = res[n_sp_y:-n_sp_y, n_sp_x:-n_sp_x]
        self.beam_fit = core[n_sp_y:-n_sp_y, n_sp_x:-n_sp_x]
        self.compton_fit = x_sec[n_sp_y:-n_sp_y, n_sp_x:-n_sp_x]
        return res

    def fft_fix(self, D):
        D = np.vstack([D[10:,],D[0:10,]])
        D = np.hstack([D[:,16:],D[:,0:16]])
        return D

    def smooth(self, his, window=3): 
        smooth_kernel = np.blackman(window)
        smooth_kernel = [x*smooth_kernel  for x in smooth_kernel]
        return  signal.fftconvolve(his, smooth_kernel, 'same')




    def calc_chi2(self, data, fit, error):
        return np.sum( np.where(error > 0,  np.abs( np.power( (data-fit)/error,2.0)),  0.0) ) 




    def __call__(self, *par):  
            NL = par[self.idxNL]
            NR = par[self.idxNR]
            Q  = par[self.idxQ]
            V  = np.sqrt(1.-Q*Q) if self.tied_VQ else par[self.idxV]
            p = par[self.idxpsum]

            par_l = np.array(par[:self.idxpsum])
            par_l[self.idxQ] = Q
            par_l[self.idxV] = V

            par_r = np.array(par[:self.idxpsum])
            par_r[self.idxQ] = -Q
            par_r[self.idxV]= -V

            self.fit_left = self.PDF(*par_l).flatten()
            self.compton_fit_left = self.compton_fit
            self.fit_right = self.PDF(*par_r).flatten()
            self.compton_fit_right = self.compton_fit

            self.compton_fit_sum = self.compton_fit_left + self.compton_fit_right

            #print(ml)

            #sum and difference for model
            md  = self.fit_left - self.fit_right
            m0  = self.fit_left + self.fit_right

            #ratio of the model
            rm =  md / m0

            #NL = np.sum(self.z_l)

            #normalized data
            self.data_left = self.data_l/NL
            self.data_right = self.data_r/NR


            #sum and difference for data
            self.data_sum  = self.data_left + self.data_right
            self.data_sum_error = np.sqrt(self.data_left/NL + self.data_right/NR)


            self.data_diff  = self.data_left - self.data_right
            self.data_diff_error =  np.sqrt(1.0 + rm*rm)*self.data_sum_error

            #expected values
            self.fit_diff   = rm*self.data_sum
            self.fit_sum    = m0


            #main contribution into difference
            chi2_diff = self.calc_chi2(self.data_diff,  self.fit_diff, self.data_diff_error)
            chi2_sum = self.calc_chi2( self.data_sum,   self.fit_sum, self.data_sum_error)

            chi2 = chi2_sum*p + chi2_diff*(1.0-p)

            return chi2

    def set_result(self):
        self.data_left_error  = np.sqrt(self.data_left/self.minuit.values['NL'])
        self.data_right_error = np.sqrt(self.data_right/self.minuit.values['NR'])


    def calc_beam_pdf(self):
        self.compton_fit_sum = self.compton_fit_sum.reshape(self.shape)
        fC0  = np.fft.fft2(self.compton_fit_sum)
        fD0  = np.fft.fft2(self.data_sum)
        s = np.abs(np.sum(fC0))
        k=1e-3
        k=0
        fB = fD0/(fC0 + k*s)

        self.data_sum =  np.abs(np.fft.ifft2(fB))
        self.data_sum =  self.fft_fix(self.data_sum)
        #self.data_sum = self.smooth(self.data_sum, 7)



    def get_fit_result(self, cfg):
        grids = get_coor_grid()
        coors = [grids['xc'],grids['yc']]
        shape = (np.shape(grids['yc'])[0]-1, np.shape(grids['xc'])[0]-1)
        data_field_names = [ 'data_sum', 'data_diff', 'data_left', 'data_right',
                        'fit_sum', 'fit_diff', 'fit_left', 'fit_right']
        data_error_names = ['data_sum_error', 'data_diff_error', 'data_left_error', 'data_right_error']
        for field_name in chain(data_field_names, data_error_names):
            setattr(self, field_name, getattr(self, field_name).reshape(shape))

        #remove fit values when there is no data
        self.fit_sum[np.abs(self.data_sum)<1e-15]=0.0
        self.fit_diff[np.abs(self.data_sum)<1e-15]=0.0
        self.fit_left[np.abs(self.data_sum)<1e-15]=0.0
        self.fit_right[np.abs(self.data_sum)<1e-15]=0.0

        #self.calc_beam_pdf()



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

        self.tied_VQ =  self.minuit.fixed['V'] and np.abs(self.minuit.values['V'])<0.00001

        self.fix(['psum'])
        self.minuit.values['psum']=0.5
        self.minuit.values['P']=0
        self.minuit.values['Q']=0
        pol_par_list  = ['P','Q']
        #self.unfix(beam_par_list)
        print(self.minuit)
        self.minuit.migrad()

        #in order to print first fit result
        if self.tied_VQ:
            Q = self.minuit.values['Q']
            V = np.sqrt( 1.0 - Q**2)
            self.minuit.values['V']  = V
            self.minuit.errors['V'] = np.abs(Q/V*self.minuit.errors['Q']) 

        print(self.minuit)
        #return it back
        if self.tied_VQ:
            self.minuit.values['V']  = 0.0
            self.minuit.errors['V'] =  0.1


        #second fit is to determine polarization. Beam parameters are fixed
        self.minuit.values['psum']=0
        beam_par_list = [ 'mx','my', 'sx','sy', 'ax', 'dax', 'ay', 'day', 'nx', 'dnx', 'ny', 'dny', 'alpha_s', 'alpha_a', 'alpha_n', 'NL']
        self.fix(beam_par_list)
        self.unfix(pol_par_list)
        self.unfix(['NR'])

        self.minuit.migrad()
        self.minuit.hesse()

        if self.tied_VQ:
            Q = self.minuit.values['Q']
            V = np.sqrt( 1.0 - Q**2)
            self.minuit.values['V']  = V
            self.minuit.errors['V'] = np.abs(Q/V*self.minuit.errors['Q']) 

        print(self.minuit)
        self.set_result()
        return self.minuit

