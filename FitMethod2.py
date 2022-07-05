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
        self.shift_x = 0.0
        self.shift_y = 0.0
        self.smooth_value=3
        with  open('shift.txt','r') as f:
            self.shift_x = float(f.readline())
            self.shift_y = float(f.readline())
            self.smooth_value = float(f.readline())
        self.x = x
        z_l = self.smooth(z_l,self.smooth_value)
        z_r = self.smooth(z_r,self.smooth_value)
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

    def extend_grid1(self, x, n):
        #    ...+9....#########...+9.... extend region +9+9
        step = x[1]-x[0]
        x_left  = np.linspace(x[0]-n*step,x[0], num=n, endpoint=False)
        x_right = np.linspace(x[-1]+step, x[-1]+(n+1)*step, num=n, endpoint=False)
        return np.concatenate((x_left, x, x_right))

    def extend_grid2(self, y, x, add_shape):
        return  np.meshgrid(
                self.extend_grid1(x,add_shape[1]),
                self.extend_grid1(y,add_shape[0]
                ) )

    def extend(self, data, add_shape):
        new  = np.zeros( ( data.shape[0]+add_shape[0]*2, data.shape[1]+add_shape[1]*2 ) )
        new[ add_shape[0]:-add_shape[0], add_shape[1]: -add_shape[1] ] = data
        return new


    def shrink(self, data, minus_shape):
        return data[minus_shape[0]:-minus_shape[0], minus_shape[1]:-minus_shape[1]]




    def PDF(self, E, L, P, V, Q, beta, alpha_d,  mx, my, sx, sy, ax, dax,  ay, day, nx, dnx, ny, dny, alpha_s, alpha_a, alpha_n):
        x_mid = self.x[0]-mx
        y_mid = self.x[1]-my

        shape_diff = (20,32)
        x, y =  self.extend_grid2(y_mid,x_mid, shape_diff)

        C = self.ComptonPDF(x, y, E, L, P, V, Q, beta)
        B =    self.BeamPDF(x, y, sx, sy, ax, dax,  ay, day, nx, dnx, ny, dny, alpha_s, alpha_a, alpha_n)

        CB = signal.fftconvolve(C, B, mode = 'same')

        self.compton_fit = self.shrink(C, shape_diff) 
        self.beam_fit    = self.shrink(B, shape_diff) 
        return self.shrink(CB, shape_diff)

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

    #def calc_phase(self, data):
    #    return np.sum( np.angle(data[0:self.shape[0]//2, 0:self.shape[1]//2]) )/(self.shape[0]*self.shape[1]/4)

    def calc_phase_y(self, data):
        return np.sum( np.angle( data[ 0 : data.shape[0]//2 ,0 :] ) )/(data.shape[0]*data.shape[1]/2)

    def calc_phase_x(self, data):
        return np.sum( np.angle(data[0:data.shape[0] , 0:data.shape[1]//2]) )/(data.shape[0]*data.shape[1]/2)

    def calc_phase(self, data):
        shape = np.shape(data)
        N0 = shape[0]//2
        N1 = shape[1]//2
        N = 2*N0*N1
        return [ np.sum( np.angle(data[0:N0, 0:]) )/N, np.sum( np.angle(data[0: , 0:N1]) )/N ]

    def shift_phase(self, data, phases):
        shape = np.shape(data)
        idx =  np.indices(shape)
        z = np.exp ( - 1j* ( idx[0]*phases[0]/shape[0] +  idx[1]*phases[1]/shape[1] ) )
        return data*z

    def shift_phase2(self, data, phases):
        shape = np.shape(data)
        idx =  np.indices(shape)
        z = np.exp ( - 1.0j*np.pi*( idx[0]*phases[0] +  idx[1]*phases[1] ) )
        return data*z

    #def gaus_filter(self,data):
    #    def gaus(x,sx):
    #        return np.exp( - 0.5*np.power(x/sx) )/(np.sqrt(2.0*np.pi)*sx)

    #    def prepare_idx(i, N):
    #        if i 

#            
#        shape = np.shape(data)
#        idx =  np.indices(shape)
#




    def calc_beam_pdf(self):
        self.data_diff = self.data_sum.copy()
        self.compton_fit_sum = self.compton_fit_sum.reshape(self.shape)
        self.is_extend = True
        #self.add_shape = (40,64)
        self.add_shape = (100,100)
        D0 = self.data_sum
        if self.is_extend:
            self.compton_fit_sum = self.extend(self.compton_fit_sum, self.add_shape)
            D0 = self.extend(D0, self.add_shape)

        fC0  = np.fft.fft2(self.compton_fit_sum)
        fD0  = np.fft.fft2(D0)
        sum2 = np.sum(np.abs(fC0*fC0))
        k=6e-4
        eps=1e-9
        eps = 1e-14
        reg = np.abs(fC0*fC0) / ( np.abs(fC0*fC0) + k*sum2)
        fB = fD0/( fC0 + eps)*reg

        phi_fB = self.calc_phase(fB)
        def print_phase(title, data):
            print("average complex phase for {:^10} degree [{:10.3f} , {:10.3f} ] degree".format(title, self.calc_phase_x(data)*180.0/np.pi, self.calc_phase_x(data)*180.0/np.pi) )
        print_phase("fB", fB)
        print_phase("fC0", fC0)
        print_phase("fD0", fD0)

        fB = self.shift_phase2(fB,[self.shift_y,self.shift_x])

        self.data_sum = np.abs(np.fft.ifft2(fB))

        if self.is_extend:
            self.data_sum  = self.shrink(self.data_sum,self.add_shape)
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

        self.calc_beam_pdf()



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
        data_field_dict['data_sum'].interpolation='bicubic'
        data_field_dict['data_diff'].interpolation='bicubic'
        data_field_dict['data_sum'].palette=plt.cm.magma
        #data_field_dict['data_diff'].palette=plt.cm.viridis
        #data_field_dict['data_diff'].palette=plt.cm.seismic
        data_field_dict['data_diff'].palette=plt.cm.magma
        #data_field_dict['data_diff'].palette=plt.cm.coolwarm
        #data_field_dict['data_diff'].palette=plt.cm.PRGn
        #data_field_dict['data_sum'].x = self.extend_grid1(grids['xc'],self.add_shape[1])
        #data_field_dict['data_sum'].y = self.extend_grid1(grids['yc'], self.add_shape[0])
        #print( data_field_dict['data_diff'].y)
        #data_field_dict['data_diff'].x = self.grids['xc']
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
        print("Fit parameters initial configuration...")
        print(self.minuit)
        print("Performing first fit for beam shape determination...")
        beam_par_list = [ 'mx','my', 'sx','sy', 'ax', 'dax', 'ay', 'day', 'nx', 'dnx', 'ny', 'dny', 'alpha_s', 'alpha_a', 'alpha_n']
        #self.fix(beam_par_list)
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
        print("Performing second fit with fixed beam shape and free polarization")

        #self.minuit.migrad()
        #self.minuit.hesse()

        if self.tied_VQ:
            Q = self.minuit.values['Q']
            V = np.sqrt( 1.0 - Q**2)
            self.minuit.values['V']  = V
            self.minuit.errors['V'] = np.abs(Q/V*self.minuit.errors['Q']) 

        print(self.minuit)
        self.set_result()
        return self.minuit

