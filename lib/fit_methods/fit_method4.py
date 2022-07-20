import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code
from scipy import signal
from math import cos,sin
from itertools import chain
import sys
sys.path.append('../lib')
from pol_fit_lib import wrap_array
from pol_lib import get_coor_grid
#from pol_plot_lib import *




#Fit by 2D gaus * Compton to data sum (no polarization)
class FitMethod4:

    def __init__(self, x, z_l, z_r):
        self.fit_method=4
        self.x = x
        self.data_l = z_l
        self.data_r = z_r
        self.shape = np.shape(z_l)
        fit_varnames  = list(self.PDF.__code__.co_varnames)[1:self.PDF.__code__.co_argcount]+['N']
        self.inipars = dict.fromkeys(fit_varnames, 0.0)
        self.func_code = make_func_code(fit_varnames)
        self.minuit = Minuit(self, **self.inipars)
        self.minuit.print_level = 0
        self.minuit.errordef=1

        self.idxE = fit_varnames.index('E')
        self.idxL = fit_varnames.index('L')
        self.idxN=fit_varnames.index('N')
        self.idxex=fit_varnames.index('ex')
        self.idxey=fit_varnames.index('ey')
        self.idxP =fit_varnames.index('P')
        self.idxQ =fit_varnames.index('Q')
        self.idxV =fit_varnames.index('V')
        self.idxbeta = fit_varnames.index('beta')
        self.idxN=fit_varnames.index('N')
        self.idxsx=fit_varnames.index('sx')
        self.idxsy=fit_varnames.index('sy')
        self.idxmx=fit_varnames.index('mx')
        self.idxmy=fit_varnames.index('my')

        def set_par(name, value, error, limits, fix):
            self.minuit.values[name] = value
            self.minuit.errors[name] = error
            self.minuit.limits[name] = limits
            self.minuit.fixed[name] = fix

        set_par('E'    , 4730  , 0   , [4000 , 5000] , True)
        set_par('L'    , 28e3  , 0   , [20e3 , 35e3] , True)
        set_par('P'    , 0.0   , 0.1 , [-1.0 , 1.0]  , True)
        set_par('Q'    , 0.0   , 0.1 , [-1.0 , 1.0]  , True)
        set_par('V'    , 0.0   , 0.1 , [-1.0 , 1.0]  , True)
        set_par('beta' , 0.0   , 0.1 , [-7.0 , +7.0] , True)
        set_par('N'   , 1     , 0.1 , [0.   , 1e10] ,  False)
        set_par('sy'   , 2.0     , 1.0 , [0.   , 10.0] , False)
        set_par('sx'   , 3.0     , 1.0 , [0.   , 20.0] , False)
        set_par('ey'   , 1    , 0.1 , [0.   , 1e10] , True)
        set_par('ex'   , 1    , 0.1 , [0.   , 1e10] , True)


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



    #Gaus beam pdf
    def BeamPDF(self, x, y, mx, my, sx,sy):
        return  np.exp(  - np.power( (x-mx)/sx, 2.0) - np.power( (y-my)/sy , 2.0) )/(2.0*np.pi**sx*sy)

    



    def extend_grid1(self, x, n):
        #    ...+9....#########...+9.... extend region +9+9
        step = x[1]-x[0]
        x_left  = np.linspace(x[0]-n*step,x[0], num=n, endpoint=False)
        x_right = np.linspace(x[-1]+step, x[-1]+(n+1)*step, num=n, endpoint=False)
        return np.concatenate((x_left, x, x_right))

    def extend_grid2(self, x, y, add_shape):
        ey = self.extend_grid1(y,add_shape[0])
        ex = self.extend_grid1(x,add_shape[1])
        return  np.meshgrid(ex,ey)

    def extend(self, data, add_shape):
        new  = np.zeros( ( data.shape[0]+add_shape[0]*2, data.shape[1]+add_shape[1]*2 ) )
        new[ add_shape[0]:-add_shape[0], add_shape[1]: -add_shape[1] ] = data
        return new


    def shrink(self, data, minus_shape):
        return data[minus_shape[0]:-minus_shape[0], minus_shape[1]:-minus_shape[1]]


    def PDF(self, E, L, P, V, Q, beta, mx, my, sx, sy, ex,ey):
        x, y =  self.extend_grid2(self.x[0],self.x[1], (ey,ex))
        C = self.ComptonPDF(x, y, E, L, P, V, Q, beta)
        #print("C=")
        #self.print_his(C)
        B = self.BeamPDF(x, y, mx, my, sx, sy)
        #print("B=")
        #self.print_his(B)
        CB = signal.fftconvolve(C, B, mode = 'same')
        #print("CB=")
        #self.print_his(CB*1e3)
        return CB

    def calc_chi2(self, data, fit, error):
        return np.sum( np.where(error > 0,  np.abs( np.power( (data-fit)/error,2.0)),  0.0) ) 

    def print_his(self, data, width=6):
        print(''.rjust(width*data.shape[1],'━'))
        for y in range(0, data.shape[0]):
            for x in range(0, data.shape[1]):
                print( '{:+{}.1f}'.format(data[y,x], width), end='')
            print('')
        print(''.rjust(width*data.shape[1],'━'))

    def __call__(self, *par):  
            N = par[self.idxN]
            E  = par[self.idxE]
            L  = par[self.idxL]
            P  = 0.0
            Q  = 0.0
            V  = 0.0
            beta  = 0.0
            mx  = par[self.idxmx]
            my  = par[self.idxmy]
            sx  = par[self.idxsx]
            sy  = par[self.idxsy]
            ex  = int(par[self.idxex])
            ey  = int(par[self.idxey])


            self.fit_sum = self.PDF(E,L,P,V,Q,beta, mx, my, sx, sy, ex, ey)

            fit_sum = self.shrink(self.fit_sum, (ey,ex))

            self.data_sum  = (self.data_l + self.data_r)/(2.0*N)
            self.data_sum_error = np.sqrt(self.data_sum/(2.0*N))
            #self.print_his(self.data_sum)
            #self.print_his(self.fit_sum)

            chi2 = self.calc_chi2(self.data_sum,  fit_sum, self.data_sum_error)
            #print(chi2)
            return chi2
    
    def fit(self):
        self.minuit.migrad()
        self.minuit.hesse()
        self.ndf = np.shape(self.x[0])[0]*np.shape(self.x[1])[0]-self.minuit.nfit
        self.chi2 = self.minuit.fval
        return self.minuit

