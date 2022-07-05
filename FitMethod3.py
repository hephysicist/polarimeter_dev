import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code
from pol_fit_lib import wrap_array
from pol_lib import get_coor_grid
from scipy import signal
from math import cos,sin
from pol_plot_lib import *


class FitMethod3:

    def __init__(self, x, z_l, z_r):
        self.x = x
        self.data_l = z_l
        self.data_r = z_r
        self.shape = np.shape(z_l)
        fit_varnames  = list(self.ComptonPDF.__code__.co_varnames)[3:self.ComptonPDF.__code__.co_argcount]+['N','DN','sx','sy','k','eps', 'ex','ey']
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

        self.ndf = np.shape(self.x[0])[0]*np.shape(self.x[1])[0]

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
        return  np.exp(  - np.power( (x-mx)/sx, 2.0) - np.power( (y-my)/sy , 2.0) )


    def extend(self, data, add_shape):
        new  = np.zeros( ( data.shape[0]+add_shape[0]*2, data.shape[1]+add_shape[1]*2 ) )
        new[ add_shape[0]:-add_shape[0], add_shape[1]: -add_shape[1] ] = data
        return new

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

    def print_his(self, data):
        print(''.rjust(7*data.shape[1],'━'))
        for y in range(0, data.shape[0]):
            for x in range(0, data.shape[1]):
                print( '{:+7.1f}'.format(data[y,x]), end='')
            print('')
        print(''.rjust(5*data.shape[1],'━'))

    def shift_phase2(self, data, phases):
        shape = np.shape(data)
        idx =  np.indices(shape)
        z = np.exp ( - 1.0j*np.pi*( idx[0]*phases[0] +  idx[1]*phases[1] ) )
        return data*z

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
            #print(NL,NR,E,L,P,Q,V,beta, eps, k)

            #print('initial shape = ', self.shape)
            add_shape = ( int(par[self.idxey]), int(par[self.idxex]) )
            #print('add_shape = ', add_shape)

            sx = self.idxsx
            sy = self.idxsy

            #print("self.x0 = ", self.x[0])
            #xx, yy = self.extend_grid2(self.x[0], self.x[1], add_shape)
            xx,yy = np.meshgrid(self.x[0], self.x[1])
            #print('xx shape =', xx.shape)
            #print('yy shape =', yy.shape)
            #print('xx = ', xx)
            #print('yy = ', yy)

            G = self.Gaus(xx,yy, 0.0, 0.0, sx, sy)
            G = self.extend(G, add_shape)
            G = self.fft_fix(G)
            G = G/np.sum(G)

            self.fit_left  = self.ComptonPDF(xx, yy, E, L, P,  V,  Q, beta)
            self.fit_right = self.ComptonPDF(xx, yy, E, L, P, -V, -Q, beta)

            self.fit_left = self.extend(self.fit_left, add_shape)
            self.fit_right = self.extend(self.fit_right, add_shape)

            #print('max of fit_left = ', np.amax(self.fit_left))
            #print('max of fit_right = ', np.amax(self.fit_right))



            self.fit_sum = self.fit_left + self.fit_right
            self.fit_diff = self.fit_left - self.fit_right

            C0  = self.fit_sum
            C1  = self.fit_diff

            #print("C0 shape = ", C0.shape)

            #normalized data
            self.data_left  = self.data_l/NL
            self.data_right = self.data_r/NR
            self.data_left_error =  np.sqrt(self.data_left/NL)
            self.data_right_error = np.sqrt(self.data_right/NR)

            #sum and difference for data
            self.data_sum  = self.data_left + self.data_right
            self.data_sum_error = np.sqrt(self.data_left/NL + self.data_right/NR)


            #print('max of fit_sum = ', np.amax(self.fit_sum))
            #print('max of data_sum = ', np.amax(self.data_sum))

            #for x in range(0, self.shape[1]):
            #    for y in range(0, self.shape[0]):
            #        print( '{:3.1f} '.format(self.fit_sum[y,x]), end=' ')
            #    print('')

            D0 = self.extend(self.data_sum, add_shape)
            #print("data_sum")
            #self.print_his(D0)
            #print("fit_sum")
            #self.print_his(self.fit_sum)
            #print("(data_sum-fit_sum)/fit_sum")
            #self.print_his((D0-self.fit_sum)/self.fit_sum)
            #print(D0)
            #print("D0 shape = ", D0.shape)

            fC0  = np.fft.fft2(C0)
            fC1  = np.fft.fft2(C1)
            fD0  = np.fft.fft2(D0)
            fG   = np.fft.fft2(G)

            #Fourier image of the beam function
            #k=0
            #eps =1e-9
            #print("k=",k, " eps = ", eps)
            A2 = np.sum(np.abs(fC0*fC0)) #normalization constant
            R = np.abs(fC0*fC0) / ( np.abs(fC0*fC0) + k*A2) #regularization koeff
            #print(R.shape)
            #self.print_his(R)
            fB = fD0/(fC0 + eps)*R

            fG=1.0

            fCBG1 = fC1*fB*fG
            fCBG0 = fC0*fB*fG


            CB1 = self.shrink(np.real(np.fft.ifft2(fCBG1)), add_shape)
            CB0 = self.shrink(np.abs(np.fft.ifft2(fCBG0)), add_shape)
            #print("CB1.shape = ", CB1.shape)

            self.fit_diff        =  self.data_sum*CB1
            self.data_diff       = (self.data_left - self.data_right)*CB0
            self.data_diff_error =  np.sqrt( np.power( CB0 - CB1, 2.0) * self.data_left/NL + np.power( CB0 + CB1, 2.0) * self.data_right/NR)

            chi2  = self.calc_chi2( self.data_diff,  self.fit_diff, self.data_diff_error)    

            self.fit_sum = CB0

            
            #self.data_sum = self.data_sum/CB0  #To determine registration efficiency

            
            #this is to show beam function 
            fB = self.shift_phase2(fB,[1.0,1.0])
            #self.data_sum = np.abs(   self.shrink(np.fft.ifft2(fB),add_shape) )

            return chi2

    def get_fit_result(self, cfg):
        grids = get_coor_grid()
        coors = [grids['xc'],grids['yc']]
        shape = (np.shape(grids['yc'])[0]-1, np.shape(grids['xc'])[0]-1)
        data_field_names = [ 'data_sum', 'data_diff', 'data_left', 'data_right',
                        'fit_sum', 'fit_diff', 'fit_left', 'fit_right']
        data_error_names = ['data_sum_error', 'data_diff_error', 'data_left_error', 'data_right_error']

        #remove fit values when there is no data
        #self.fit_sum[np.abs(self.data_sum)<1e-14]=0.0
        #self.fit_diff[np.abs(self.data_sum)<1e-14]=0.0
        #self.fit_left[np.abs(self.data_sum)<1e-14]=0.0
        #self.fit_right[np.abs(self.data_sum)<1e-14]=0.0


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
        interp='bicubic'
        #interp='none'
        data_field_dict['data_sum'].interpolation=interp
        data_field_dict['data_diff'].interpolation=interp
        data_field_dict['data_sum'].palette=plt.cm.magma
        #data_field_dict['data_diff'].palette=plt.cm.viridis
        data_field_dict['data_diff'].palette=plt.cm.seismic
        #data_field_dict['data_diff'].palette=plt.cm.magma
        #data_field_dict['data_diff'].palette=plt.cm.coolwarm
        #data_field_dict['data_diff'].palette=plt.cm.PRGn
        return data_field_dict
    
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



        self.tied_VQ =  self.minuit.fixed['V'] and np.abs(self.minuit.values['V'])<0.00001
        #determine maximum in the sum of the left and right data
        self.minuit.values['N'] = 0.5*np.amax( self.data_l + self.data_r)
        self.minuit.fixed['N'] = True
        print("Fixing N = {:.3}".format(self.minuit.values['N']))

        print("Fit parameters initial configuration...")
        print(self.minuit)
        self.minuit.migrad()
        self.minuit.hesse()

        if self.tied_VQ:
            Q = self.minuit.values['Q']
            V = np.sqrt( 1.0 - Q**2)
            self.minuit.values['V']  = V
            self.minuit.errors['V'] = np.abs(Q/V*self.minuit.errors['Q']) 

        print(self.minuit)
        print(self.minuit.values['eps'])
        #self.set_result()
        return self.minuit


