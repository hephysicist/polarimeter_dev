import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code
from scipy import signal
from math import cos,sin
import matplotlib.pyplot as plt

import sys
sys.path.append('../lib')
from pol_fit_lib import wrap_array, FWHM2D
from pol_lib import get_coor_grid
from pol_plot_lib import data_field
from fit_method4 import *



#Fit by Fourier reconstruction of beam shape * Compton to normalized data difference
class FitMethod3:

    def __init__(self, x, z_l, z_r):
        self.fit_method=3
        self.x = x
        self.data_l = z_l
        self.data_r = z_r
        self.shape = np.shape(z_l)
        fit_varnames  = list(self.ComptonPDF.__code__.co_varnames)[3:self.ComptonPDF.__code__.co_argcount]+['N','DN','sx','sy','k','eps', 'ex','ey', 'psum']
        self.parlist = fit_varnames
        self.inipars = dict.fromkeys(fit_varnames, 0.0)
        self.func_code = make_func_code(fit_varnames)
        self.minuit = Minuit(self, **self.inipars)
        self.minuit.print_level = 0
        self.minuit.errordef=1

        self.idxE = fit_varnames.index('E')
        self.idxL = fit_varnames.index('L')
        self.idxP =fit_varnames.index('P')
        self.idxQ =fit_varnames.index('Q')
        self.idxV =fit_varnames.index('V')
        self.idxbeta = fit_varnames.index('beta')
        self.idxN=fit_varnames.index('N')
        self.idxDN=fit_varnames.index('DN')
        self.idxsx=fit_varnames.index('sx')
        self.idxsy=fit_varnames.index('sy')
        self.idxk=fit_varnames.index('k')
        self.idxeps=fit_varnames.index('eps')
        self.idxex=fit_varnames.index('ex')
        self.idxey=fit_varnames.index('ey')
        self.idxpsum=fit_varnames.index('psum')

        self.efficiency = np.ones(self.shape)

        def set_par(name, value, error, limits, fix):
            self.minuit.values[name] = value
            self.minuit.errors[name] = error
            self.minuit.limits[name] = limits
            self.minuit.fixed[name] = fix

        set_par('E'    , 4730  , 0   , [4000 , 5000] , True)
        set_par('L'    , 28e3  , 0   , [20e3 , 35e3] , True)
        set_par('P'    , 0.0   , 0.1 , [-1.0 , 1.0]  , False)
        set_par('Q'    , 0.0   , 0.1 , [-1.0 , 1.0]  , False)
        set_par('V'    , 0.0   , 0.1 , [-1.0 , 1.0]  , True)
        set_par('beta' , 0.0   , 0.1 , [-7.0 , +7.0] , False)
        set_par('N'   , 1     , 0.1 , [0.   , 1e10] ,  True)
        set_par('DN'   , 1     , 0.1 , [0.   , 1e10] , False)
        set_par('sx'   , 1     , 0.1 , [0.   , 1e10] , False)
        set_par('sy'   , 1     , 0.1 , [0.   , 1e10] , False)
        set_par('ey'   , 1    , 0.1 , [0.   , 1e10] , True)
        set_par('ex'   , 1    , 0.1 , [0.   , 1e10] , True)
        set_par('k'    , 6e-4  , 0.1 , [0.   , 1.]   , True)
        set_par('eps'  , 1e-14 , 0.1 , [0.   , 1.]   , True)
        set_par('psum'  , 0.0 , 0.1 , [0.   , 1.]   , True)

    def ComptonPDF(self, x,y, E=4730., L=33000., P = 0., V = 0., Q=0., beta=0.):
        #return np.exp( -np.power(x,2.)/24. -np.power(y,2.0)/12)
        me = 0.5109989461*10**6   #electorn mass eV
        g = E*1.0e6/me            #gamma factor
        o = 2.3526413364          #eV Initial laser photon 527 nm
        
        r = np.hypot(x,y)
        phi = np.arctan2(y, x)

        eta = g*r/L
        eta2 = np.power(eta,2.0)
        
        kappa = 4.*g*o/me

        t = 1. + eta2 + kappa

        A = np.power(1.0+kappa,2.0)/(1.0 + 0.5*kappa**2/(1+kappa))
        
        sigma = A/np.power(t,2.)*( 1. + 0.5*(kappa**2.)/(t*(1.+eta2)) -
                - 2.*eta2/(1.+eta2)/(1.+eta2) * (1.0 - Q * np.cos(2.*(phi-beta)))+
                + P*V*eta*kappa / t /( 1. + eta2 )*np.sin(phi)
            )
        return sigma


    def Gaus(self, x, y, mx, my, sx, sy):
        return  np.exp(  - 0.5*(np.power( (x-mx)/sx, 2.0) + np.power( (y-my)/sy , 2.0) ) )/(2.0*np.pi*sx*sy)


    def extend(self, data, add_shape):
        new  = np.zeros( ( data.shape[0]+add_shape[0]*2, data.shape[1]+add_shape[1]*2 ) )
        new[ add_shape[0]:-add_shape[0], add_shape[1]: -add_shape[1] ] = data
        return new

    def extend_from_base(self, data, base):
        new  = base.copy()
        add_shape = ( (base.shape[0]-data.shape[0])//2, (base.shape[1]-data.shape[1])//2 )
        new[ add_shape[0]:-add_shape[0], add_shape[1]: -add_shape[1] ] = data
        return new

    def arrange(self, data, base):
        add_shape = ( (base.shape[0]-data.shape[0])//2, (base.shape[1]-data.shape[1])//2 )
        #data[0:add_shape[0], 0: ]  = base[0:add_shape[0], 0: ]
        #data[-add_shape[0]:-1, 0: ]  = data[-add_shape[0]:-1, 0: ]
        new = base.copy()
        new[ add_shape[0]:-add_shape[0], add_shape[1]: -add_shape[1] ]  = data[ add_shape[0]:-add_shape[0], add_shape[1]: -add_shape[1] ]


    def shrink(self, data, minus_shape):
        return data[minus_shape[0]:-minus_shape[0], minus_shape[1]:-minus_shape[1]]

    #calculate Fourier image of beam pdf (regularized)
    def fBeamPDF(self, data, compton, add_shape, k=6e-4, eps=1e-9):
        fC  = np.fft.fft2(model) #Fourier image of model
        fD  = np.fft.fft2(data)  #Fourier image of data
        A2 = np.sum(np.abs(fC*fC)) #normalization constant
        R = np.abs(fC*fC) / ( np.abs(fC*fC) + k*A2) #regularization koeff
        fB = fD0/( fC0 + eps)*R
        return fB

    def calc_chi2(self, data, fit, error):
        return np.sum( np.where(error > 0,  np.abs( np.power( (data-fit)/error,2.0)),  0.0) ) 

    def extend_grid1(self, x, n):
        #    ...+9....#########...+9.... extend region +9+9
        step = x[1]-x[0]
        x_left  = np.linspace(x[0]-n*step,x[0], num=n, endpoint=False)
        x_right = np.linspace(x[-1]+step, x[-1]+(n+1)*step, num=n, endpoint=False)
        return np.concatenate((x_left, x, x_right))

    def extend_grid2(self, x, y, add_shape):
        ey = self.extend_grid1(y,add_shape[0])
        ex = self.extend_grid1(x,add_shape[1])
        #print('ey=', ey, ' shape =', ey.shape)
        #print('ex=', ex, ' shape = ', ex.shape)
        return  np.meshgrid(ex,ey)

    def print_his(self, data, width=6):
        print(''.rjust(width*data.shape[1],'━'))
        for y in range(0, data.shape[0]):
            for x in range(0, data.shape[1]):
                print( '{:+{}.1f}'.format(data[y,x], width), end='')
            print('')
        print(''.rjust(width*data.shape[1],'━'))

    def shift_phase2(self, data, phases):
        shape = np.shape(data)
        idx =  np.indices(shape)
        z = np.exp ( - 1.0j*np.pi*( idx[0]*phases[0] +  idx[1]*phases[1] ) )
        return data*z

    def smooth(self, his, window=3): 
        smooth_kernel = np.blackman(window)
        smooth_kernel = [x*smooth_kernel  for x in smooth_kernel]
        return  signal.fftconvolve(his, smooth_kernel, 'same')

    def fft_fix(self, D):
        n2y=D.shape[0]//2
        n2x=D.shape[1]//2
        D = np.vstack([D[n2y:,],D[0:n2y,]])
        D = np.hstack([D[:,n2x:],D[:,0:n2x]])
        return D

    def __call__(self, *par):  
        NL = par[self.idxN]*(1.0 + 0.5* par[self.idxDN])
        NR = par[self.idxN]*(1.0 - 0.5* par[self.idxDN])
        E  = par[self.idxE]
        L  = par[self.idxL]
        P  = par[self.idxP]
        Q  = par[self.idxQ]
        V  = np.sqrt(1.-Q*Q) if self.tied_VQ else par[self.idxV]

        beta  = par[self.idxbeta]
        eps   = par[self.idxeps]
        k     = par[self.idxk]

        sx = par[self.idxsx]
        sy = par[self.idxsy]

        xx, yy = self.extend_grid2(self.x[0], self.x[1], self.add_shape)

        #G = self.Gaus(xx,yy, 0.0, 0.0, sx, sy)
        #G = G/np.sum(G)

        self.fit_left  = self.ComptonPDF(xx, yy, E, L, P,  V,  Q, beta)
        self.fit_right = self.ComptonPDF(xx, yy, E, L, P, -V, -Q, beta)

        self.fit_sum = self.fit_left + self.fit_right
        self.fit_diff = self.fit_left - self.fit_right

        C0  = self.fit_sum
        C1  = self.fit_diff


        #normalized data
        self.data_left  = self.data_l/NL
        self.data_right = self.data_r/NR
        self.data_left_error =  np.sqrt(self.data_left/NL)
        self.data_right_error = np.sqrt(self.data_right/NR)

        #sum and difference for data
        self.data_sum  = self.data_left + self.data_right
        self.data_diff = self.data_left - self.data_right
        self.data_sum_error = np.sqrt(self.data_left/NL + self.data_right/NR)


        #D0 = self.extend(self.data_sum,  self.add_shape)
        #D1 = self.extend(self.data_diff, self.add_shape)
        D0 = self.extend_from_base(self.data_sum, self.cb0)
        D1 = self.extend_from_base(self.data_diff, self.cb1)


        fC0  = np.fft.fft2(C0)
        fC1  = np.fft.fft2(C1)
        fD0  = np.fft.fft2(D0)
        #fG   = np.fft.fft2(G)

        #Fourier image of the beam function
        A2 = np.sum(np.abs(fC0*fC0)) #normalization constant
        R = np.abs(fC0*fC0) / ( np.abs(fC0*fC0) + k*A2) #regularization koeff
        #RG =np.abs(fG*fG) / (np.abs(fG*fG) + kG*np.sum(np.abs(fG*fG)))
        fB = fD0/(fC0 + eps)*R 
        #fB = fB*RG/(fG + eps)

        fCB0 = fC0*fB
        fCB1 = fC1*fB

        fB = self.shift_phase2(fB,[1.0,1.0])

        CB0 = np.real(np.fft.ifft2(fCB0))
        CB1 = np.real(np.fft.ifft2(fCB1))

        B =  np.real(np.fft.ifft2(fB))
        #Bmax = np.amax(B)

        #B = np.where( B > par[self.idxxBcut]*Bmax, B, 0.0)

        
        #DG0 = signal.fftconvolve(G, D0, mode = 'same')
        #DG1 = signal.fftconvolve(G, D1, mode = 'same')

        self.CB0 = CB0.copy()
        self.CB1 = CB1.copy()


        D0  = self.shrink(D0, self.add_shape)
        D1  = self.shrink(D1, self.add_shape)
        #DG0 = self.shrink(DG0, self.add_shape)
        #DG1 = self.shrink(DG1, self.add_shape)
        CB1 = self.shrink(CB1, self.add_shape)
        CB0 = self.shrink(CB0, self.add_shape)
        #G   = self.shrink(G, self.add_shape)
        B   = self.shrink(B, self.add_shape)
        C0   = self.shrink(C0, self.add_shape)

        #DG_norm = np.sum(np.abs(D0))/np.sum(np.abs(DG0))
        #DG0 = DG0*DG_norm
        #DG1 = DG1*DG_norm

        self.fit_diff        =  D0*CB1
        self.data_diff       =  D1*CB0
        #self.data_diff_error =  np.sqrt( np.power( CB0 - CB1, 2.0) * self.data_left/NL + np.power( CB0 + CB1, 2.0) * self.data_right/NR)
        x =  np.sqrt( np.power( CB0 - CB1, 2.0) * self.data_left/NL + np.power( CB0 + CB1, 2.0) * self.data_right/NR)
        Left = self.data_l
        Right = self.data_r
        DL = self.data_left/NL
        DR = self.data_right/NR

        data_diff_error2 = 4.0*( CB1**2 * (DL+DR) + CB0**2 * ( Left*DL**2 + Right*DR**2)/D0**2  - 2.0*CB1*CB0*( DL**2 * NL - DR**2 * NR)/D0 )

        self.data_diff_error = np.sqrt(data_diff_error2)

        chi2_diff  = self.calc_chi2( self.data_diff ,  self.fit_diff, self.data_diff_error)   

        self.fit_sum  = CB0
        self.data_sum = D0
        self.data_sum_error = self.data_sum_error

        p = par[self.idxpsum]

        chi2_L = ((L-self.L)/self.Lerror)**2
        
        if np.abs(p) < 1e-9:
            chi2_sum = 0.0
        else:
            chi2_sum = self.calc_chi2( self.data_sum,  self.fit_sum, self.data_sum_error)


        chi2 = chi2_diff*(1.0-p)+chi2_sum*p + chi2_L

        if np.isnan(chi2): chi2_sum = 1e100

        self.beam_shape = B
        
        x = D0/CB0
        average = np.sum(x)/(x.shape[0]*x.shape[1])
        self.efficiency = np.where( np.abs(self.data_l + self.data_r)<1.0 , average, x)

        #self.remains  = (self.fit_diff - self.data_diff)/self.data_diff_error
        self.remains  = (self.fit_diff - self.data_diff)
        if np.isnan(chi2): return 1e100
        return chi2

    def perimeter_norm(self, data):
        shape = data.shape
        s = np.sum( data[0:,0:0] )
        s+= np.sum( data[0:, -1:-1] )
        s+= np.sum( data[0:0, 0:] )
        s+= np.sum( data[-1:-1,0:] )
        return s

    def make_projection(self, axis, his, his_error):
        return np.sum( his, axis = axis),  np.sqrt( np.sum(his_error**2, axis=axis) )


    def get_fit_result(self, cfg):
        grids = get_coor_grid()
        coors = [grids['xc'],grids['yc']]
        data_field_names = [ 'data_sum', 'data_diff', 'fit_diff', 'fit_sum']

        data_field_dict = {}

        for field_name in data_field_names:
            this_data = getattr(self, field_name)

            if 'data' in field_name:
                this_data_type = 'dat'
                this_data_err = getattr(self,field_name + '_error')
            else:
                this_data_type = 'fit'
                this_data_err = np.zeros(this_data.shape)

            this_data_x, this_data_x_error = self.make_projection(0, this_data, this_data_err)
            this_data_y, this_data_y_error = self.make_projection(1, this_data, this_data_err)

            data_field_dict[field_name] = data_field (coors, this_data, this_data_err, label=field_name, data_type = this_data_type)

            data_field_dict[field_name+'_px'] = data_field( [grids['xc'], None], this_data_x, this_data_x_error, label= field_name+'_px', data_type = this_data_type)
            data_field_dict[field_name+'_py'] = data_field( [None, grids['yc']], this_data_y, this_data_y_error, label= field_name+'_py', data_type = this_data_type)


        #data_field_dict['fit_sum_px']  = data_field([grids['xc'], None], np.sum(self.fit_sum, axis=0), None, label='fit_sum_px', data_type = "fit")
        #data_field_dict['fit_sum_py']  = data_field([None, grids['yc']], np.sum(self.fit_sum, axis=1), None, label='fit_sum_py', data_type = "fit")


        data_field_dict['beam_shape']  = data_field( coors,  self.beam_shape, data_err = None, label='Reconstructed beam shape', data_type='dat')
        data_field_dict['beam_shape'].interpolation='bicubic'
        data_field_dict['beam_shape'].palette=plt.cm.magma

        data_field_dict['efficiency'] = data_field( coors,  self.efficiency, data_err = None, label='Relative efficiency', data_type='dat')
        data_field_dict['efficiency'].palette=plt.cm.seismic
        data_field_dict['efficiency'].interpolation='bicubic'

        data_field_dict['remains'] = data_field( coors,  self.remains, data_err = None, label='Remains', data_type='dat')
        data_field_dict['remains'].palette=plt.cm.seismic
        data_field_dict['remains'].interpolation='bicubic'

        interp='bicubic'
        #interp='none'
        data_field_dict['data_sum'].interpolation=interp
        data_field_dict['data_diff'].interpolation=interp
        data_field_dict['fit_diff'].interpolation=interp
        data_field_dict['fit_sum'].interpolation=interp

        data_field_dict['data_sum'].palette=plt.cm.magma
        data_field_dict['fit_sum'].palette=plt.cm.magma
        #data_field_dict['data_diff'].palette=plt.cm.viridis
        data_field_dict['data_diff'].palette=plt.cm.seismic
        data_field_dict['fit_diff'].palette=plt.cm.seismic
        #data_field_dict['data_diff'].palette=plt.cm.magma
        #data_field_dict['data_diff'].palette=plt.cm.coolwarm
        #data_field_dict['data_diff'].palette=plt.cm.PRGn
        return data_field_dict

    def fixpar(self, parname, value):
        self.minuit.values[parname]=value
        self.minuit.fixed[parname] = True
    
    def fit(self, cfg):
        par_ini = cfg['model_params']
        par_val = {k: par_ini[k][0] for k, v in par_ini.items() if par_ini[k][0] != '*'}
        par_err = {k: par_ini[k][1]  for k, v in par_ini.items() if par_ini[k][1] != '*'}
        par_lim = {k: par_ini[k][2]  for k, v in par_ini.items() if par_ini[k][2] != '*'}
        par_fix = {k: par_ini[k][3] for k, v in par_ini.items() }
        
        for parname in par_val.keys():
            self.minuit.values[parname] = par_val[parname]
        for parname in par_fix.keys():
            self.minuit.fixed[parname]  = par_fix[parname]
        for parname in par_err.keys():
            self.minuit.errors[parname] = par_err[parname]
        for parname in par_lim.keys():
            self.minuit.limits[parname] = par_lim[parname]
        
        ex, ey =   int(self.minuit.values["ex"]),int(self.minuit.values["ey"])
        self.add_shape = (ey,ex)
        self.ext_shape = (self.shape[0] + 2*ey, self.shape[1] + 2*ex)

        self.Lerror = self.minuit.errors['L']
        self.L  = self.minuit.values['L']

        #gues expected data outside sensetive window
        if False:
            fm4 = FitMethod4(self.x, self.data_l, self.data_r)
            fm4.minuit.values['E'] = self.minuit.values['E']
            fm4.minuit.values['L'] = self.minuit.values['L']
            fm4.minuit.values['N'] = 0.5*np.amax( self.data_l + self.data_r)
            fm4.minuit.values['ex']= self.minuit.values['ex']
            fm4.minuit.values['ey']= self.minuit.values['ey']
            fwhmx, fwhmy = FWHM2D(self.x[0], self.x[1], self.data_l + self.data_r)
            print("sx = ", fwhmx*0.5, "sy = ", fwhmy*0.5)
            fm4.minuit.values['sx'] = fwhmx*0.5
            fm4.minuit.values['sy'] = fwhmy*0.5
            fm4.minuit.fixed['sx']=True
            fm4.minuit.fixed['sy']=True
            minuit4 = fm4.fit()
            print(minuit4)
            self.cb0 = fm4.fit_sum
            self.cb1 = np.zeros(self.ext_shape)
        else:
            self.cb0 = np.zeros(self.ext_shape)
            self.cb1 = np.zeros(self.ext_shape)




        self.tied_VQ =  self.minuit.fixed['V'] and np.abs(self.minuit.values['V'])<0.00001
        #determine maximum in the sum of the left and right data
        self.minuit.values['N'] = 0.5*np.amax( self.data_l + self.data_r)
        #self.minuit.fixed['N'] = True
        print("Fixing N = {:.3}".format(self.minuit.values['N']))

        print("Fit parameters initial configuration...")
        print(self.minuit)

        #ext_shape = (self.shape[0] + 2*int(self.minuit.values[self.idxey]), self.shape[1] + 2*int(self.minuit.values[self.idxex]))
        #self.cb0 = np.zeros(ext_shape)
        #self.cb1 = np.zeros(ext_shape)

        def migrad(k,psum=0.0):
            self.minuit.values['k']  = k 
            self.minuit.migrad()
            self.minuit.values['psum']=psum
            self.cb0 = self.CB0.copy()#*np.amax(self.data_sum)/np.amax(self.CB0)
            self.cb1 = self.CB1.copy()

        self.minuit.fixed['k']=True
        for i in range(0,5):
            migrad(0.1e-4)

        self.minuit.strategy=2
        self.minuit.fixed['k']=True
        migrad(0.1e-4,0.0)


        self.minuit.hesse()


        #print(self.minuit)
        #if np.abs(np.abs(self.minuit.values['P']) - np.abs(self.minuit.limits['P'][0])) < 0.01:
        #    for parname in ['P','Q', 'beta']:
        #        self.fixpar(parname, 0.0)
        #    self.fixpar('xBcut', 0.5)
        #    migrad(0.1e-4,0.0)
        #    for parname in ['P','Q', 'beta']:
        #        self.minuit.fixed[parname]=False
        #    self.fixpar('xBcut', 0.7)
        #    migrad(1e-4,0.0)

        if self.tied_VQ:
            Q = self.minuit.values['Q']
            V = np.sqrt( 1.0 - Q**2)
            self.minuit.values['V']  = V
            self.minuit.errors['V'] = np.abs(Q/V*self.minuit.errors['Q']) 
        print(self.minuit)
        self.ndf = np.shape(self.x[0])[0]*np.shape(self.x[1])[0]-self.minuit.nfit
        self.chi2 = self.minuit.fval

        return self.minuit


