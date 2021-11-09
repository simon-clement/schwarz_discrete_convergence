#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import numpy as np
from numpy import pi
from memoisation import memoised, FunMem
import matplotlib.pyplot as plt
import functools
import discretizations
from simulator import simulation_cv_rate_linear, simulation_cv_rate_matrixlinear
from simulator import frequency_simulation
from simulator import matrixlinear_frequency_simulation
from simulator import eigenvalues_matrixlinear_frequency_simulation
from simulator import simulation_firstlevels

# If set to True, the simulations will run, taking multiple hours.
REAL_FIG = False

def fig_lambda_Pade():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    ###########################################
    npts   = 1000
    wwdt   = np.linspace(0.05, np.pi, npts)
    rr     = 0.1
    nu1    = 0.5
    nu2    = 1.
    dt     = 1.
    aa     = 1. + np.sqrt(2.)
    bb     = 1. + 1./np.sqrt(2.)    
#####========================================================
    sBEdt  = np.zeros(npts,dtype=np.complex_)   
    sBEdt  = (np.exp(1j*wwdt)-1.)/np.exp(1j*wwdt)  
#####========================================================
    cff1   = 1./( aa * np.sqrt(dt*nu1) ) 
    cff2   = 1./( aa * np.sqrt(dt*nu2) ) 
    
    term1  = np.zeros(npts,dtype=np.complex_) 
    term2  = np.zeros(npts,dtype=np.complex_) 
    
    term1  = 1.+aa*sBEdt+aa*aa*rr*dt
    term2  = np.sqrt(1.-sBEdt)*np.sqrt(1.-aa*aa*sBEdt)
#####   
    lambda1  = np.zeros(npts,dtype=np.complex_) 
    lambda2  = np.zeros(npts,dtype=np.complex_) 
    lambda3  = np.zeros(npts,dtype=np.complex_) 
    lambda4  = np.zeros(npts,dtype=np.complex_) 
 
    lambda1 =   cff1 * np.sqrt(term1-term2)
    lambda2 = - cff2 * np.sqrt(term1-term2)    
    lambda3 =   cff1 * np.sqrt(term1+term2)
    lambda4 = - cff2 * np.sqrt(term1+term2)    
#####    
    sigma1   =  np.zeros(npts,dtype=np.complex_) 
    sigma2   =  np.zeros(npts,dtype=np.complex_) 
    sigma1   =  np.sqrt( (1j*wwdt + rr*dt)/(nu1*dt) )
    sigma2   = -np.sqrt( (1j*wwdt + rr*dt)/(nu2*dt) )
#####    
    fig  = plt.figure()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,figsize=(10,2))
    ax = axes[0]
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)
    ax.set_xlabel (r'$\omega\Delta t$', fontsize=14)
    ax.set_xticklabels(ax.get_xticks(),fontsize=14)
    ax.set_yticklabels(ax.get_yticks(),fontsize=14)  
    ax.set_ylim(-0.15,1.85)
    ax.set_title(r"$\left|\mathcal{R}(\lambda_j^{(p)})\right|$") 
    ax.semilogx( wwdt, np.abs(np.real(lambda1)),linewidth=2.,color='k', linestyle='dashed' ,label=r'$\lambda_1$')   
    ax.semilogx( wwdt, np.abs(np.real(lambda3)),linewidth=2.,color='k', linestyle='dashdot' ,label=r'$\lambda_3$')     
    ax.semilogx( wwdt, np.abs(np.real(sigma1)),linewidth=2.,color='k', linestyle='solid' ,label=r'$\sqrt{\frac{s_c \Delta t}{\nu_1 \Delta t}}$')     
    ax.semilogx( wwdt, np.abs(np.real(lambda2)),linewidth=2.,color='0.5', linestyle='dashed' ,label=r'$\lambda_2$')    
    ax.semilogx( wwdt, np.abs(np.real(lambda4)),linewidth=2.,color='0.5', linestyle='dashdot' ,label=r'$\lambda_4$')      
    ax.semilogx( wwdt, np.abs(np.real(sigma2)),linewidth=2.,color='0.5', linestyle='solid' ,label=r'$-\sqrt{\frac{s_c \Delta t}{\nu_2 \Delta t}}$')   
    ax = axes[1]
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)
    ax.set_xlabel (r'$\omega\Delta t$', fontsize=14)
    ax.set_xticklabels(ax.get_xticks(),fontsize=14)
    ax.set_yticklabels(ax.get_yticks(),fontsize=14) 
    ax.set_ylim(-0.15,1.85)
    ax.set_title(r"$\left|\mathcal{I}(\lambda_j^{(p)})\right|$") 
    ax.semilogx( wwdt, np.abs(np.imag(lambda1)),linewidth=2.,color='k', linestyle='dashed' ,label=r'$\lambda_1^{(1)}$')    
    ax.semilogx( wwdt, np.abs(np.imag(lambda3)),linewidth=2.,color='k', linestyle='dashdot' ,label=r'$\lambda_1^{(3)}$')   
    ax.semilogx( wwdt, np.abs(np.imag(sigma1)),linewidth=2.,color='k', linestyle='solid' ,label=r'$\sqrt{\frac{s_c \Delta t}{\nu_1 \Delta t}}$')    
    ax.semilogx( wwdt, np.abs(np.imag(lambda2)),linewidth=2.,color='0.5', linestyle='dashed' ,label=r'$\lambda_2^{(2)}$')      
    ax.semilogx( wwdt, np.abs(np.imag(lambda4)),linewidth=2.,color='0.5', linestyle='dashdot' ,label=r'$\lambda_2^{(4)}$')         
    ax.semilogx( wwdt, np.abs(np.imag(sigma2)),linewidth=2.,color='0.5', linestyle='solid' ,label=r'$-\sqrt{\frac{s_c \Delta t}{\nu_2 \Delta t}}$')  
    ax.legend(loc=2,prop={'size':9.5},ncol=2,handlelength=2)  
    fig.tight_layout()   
    show_or_save("fig_lambda_Pade")

def fig_rhoDNPade():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, sharex=False, sharey=True,figsize=(7,3))
    #plt.subplots_adjust(bottom=.18, top=.85)
    ax   = axes
    wmax = np.pi
    wmin = wmax / 200
    ax.set_xlim(wmin,wmax)
    ax.set_ylim(0.53,1.05)
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)
    ax.set_xlabel (r'$\omega$', fontsize=18)
    ax.set_xticklabels(ax.get_xticks(),fontsize=12)
#    ax.set_yticklabels(ax.get_yticks(),fontsize=12)  
    ax.set_title(r"$\rho_{\rm DN}^{\rm (P2,c)}$",fontsize=16) 


    builder = Builder()
    builder.R = 0.
    builder.D1 = .5

    b = 1+1/np.sqrt(2)

    def get_z_s(w):
        z = np.exp(-1j*w*builder.DT)
        return z, (z - 1)/(z*builder.DT)

    def gamma_highTilde(w):
        z, _ = get_z_s(w)
        return z - b*(z-1)

    def gamma_lowTilde(w):
        z, _ = get_z_s(w)
        return z - b*(z-1) - b * (b-1)**2 * (z-1)**2

    w, varrho = wAndRhoPadeRR(builder, gamma=gamma_highTilde)
    ax.semilogx(w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='k', linestyle='solid' ,label=r'$r=0\;{\rm s}^{-1}, \gamma = z - \beta (z-1)$')   
    w, varrho = wAndRhoPadeRR(builder, gamma=gamma_lowTilde)

    ax.semilogx( w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='k', linestyle='dashed' ,label=r'$r=0\;{\rm s}^{-1}, \gamma = z - \beta (z-1) - \beta(\beta-1)^2 (z-1)^2$')       

    builder.R = .1

    w, varrho = wAndRhoPadeRR(builder, gamma=gamma_highTilde)
    ax.semilogx( w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='0.5', linestyle='solid' ,label=r'$r=0.1\;{\rm s}^{-1}, \gamma = z - \beta (z-1)$')   

    w, varrho = wAndRhoPadeRR(builder, gamma=gamma_lowTilde)
    ax.semilogx(w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='0.5', linestyle='dashed' ,label=r'$r=0.1\;{\rm s}^{-1}, \gamma = z - \beta (z-1) - \beta (\beta-1)^2 (z-1)^2$')    

    rho_continuous = np.sqrt(builder.D1/builder.D2) * np.ones_like(w)
    ax.semilogx(w*builder.DT, rho_continuous ,linewidth=2.,color='r', linestyle='dashed' ,label=r'$\sqrt{\frac{\nu_1}{\nu_2}}$') 
    ax.set_xlabel(r"$\omega\Delta t$")
    ax.legend(loc=2,prop={'size':9},ncol=1,handlelength=2)
    fig.tight_layout()

    show_or_save("fig_rhoDNPade")

def fig_rhoDN_space():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    npts   = 1000
    ww     = np.logspace(-6, 3, npts)
    nu1    = 0.5
    nu2    = 1.
    continuous = ww*0. + np.sqrt(nu1/nu2)
    print(np.sqrt(nu1/nu2))   
    dx     = 0.1
    rr     = 0.0
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)   
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn1 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc1 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV21 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #
    dx     = 1.
    rr     = 0.0
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)   
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn2 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc2 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV22 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #      
    dx     = 5.
    rr     = 0.0
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)     
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn3 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc3 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV23 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #      
    dx     = 10.
    rr     = 0.0
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)   
    #
    phi1p  = ( 12.+5.*chi1+2.*np.sqrt(6)*np.sqrt( chi1*(6.+chi1) ) ) / (12.-chi1)
    phi2m  = ( 12.+5.*chi2-2.*np.sqrt(6)*np.sqrt( chi2*(6.+chi2) ) ) / (12.-chi2)  
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn4 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc4 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV24 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #  
    #####========================================================    
    fig  = plt.figure()
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(8,4.2))    
    ax = axes[0,0]  
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25) 
    ax.set_title(r"$\rho_{\rm DN}^{({\rm c,FD})}(\kappa_c=0, r=0\;{\rm s}^{-1})$",fontsize=10) 
    ax.set_xlim(9.e-06,1000)
    ax.set_ylim(0.45,1.05)
    ax.semilogx(ww,rhoDNFDn1,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$' )    
    ax.semilogx(ww,rhoDNFDn2,linewidth=2.,color='k'   , linestyle='solid' ,label=r'$h = 1\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDn3,linewidth=2.,color='0.5'   , linestyle='dashed' ,label=r'$h = 5\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDn4,linewidth=2.,color='0.5'   , linestyle='solid' ,label=r'$h = 10\;{\rm m}$' )  
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' ) 
    ax.legend(loc=2,prop={'size':7}, handlelength=2)
    txt = ax.annotate('(a)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))  
    
    ax  = axes[0,1] 
    ax.set_ylim(0.45,1.05)
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)     
    ax.set_title(r"$\rho_{\rm DN}^{({\rm c,FD})}(\kappa_c=1, r=0\;{\rm s}^{-1})$",fontsize=10)
    ax.semilogx(ww,rhoDNFDc1,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$' )    
    ax.semilogx(ww,rhoDNFDc2,linewidth=2.,color='k'   , linestyle='solid' ,label=r'$h = 1\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDc3,linewidth=2.,color='0.5'   , linestyle='dashed' ,label=r'$h = 5\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDc4,linewidth=2.,color='0.5'   , linestyle='solid' ,label=r'$h = 10\;{\rm m}$' )  
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' )  
    txt = ax.annotate('(b)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))  

    ax = axes[0,2]   
    ax.set_ylim(0.45,1.05)
    ax.set_title(r"$\rho_{\rm DN}^{({\rm c,FV})}(r=0\;{\rm s}^{-1})$",fontsize=10)
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25) 
    ax.semilogx(ww,rhoDNFV21,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$' )    
    ax.semilogx(ww,rhoDNFV22,linewidth=2.,color='k'   , linestyle='solid' ,label=r'$h = 1\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFV23,linewidth=2.,color='0.5'   , linestyle='dashed' ,label=r'$h = 5\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFV24,linewidth=2.,color='0.5'   , linestyle='solid' ,label=r'$h = 10\;{\rm m}$' )  
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' )  
    txt = ax.annotate('(c)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))  
    #####========================================================
    dx     = 0.1
    rr     = 0.1
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)   
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn1 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc1 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV21 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #
#####========================================================
    dx     = 1.
    rr     = 0.1
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)   
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn2 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc2 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV22 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #      
#####========================================================    
    dx     = 5.
    rr     = 0.1
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)     
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn3 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc3 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV23 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #      
###========================================================  
    dx     = 10.
    rr     = 0.1
    #
    chi1   = dx*dx*(rr+1j*ww)/nu1
    chi2   = dx*dx*(rr+1j*ww)/nu2  
    #
    sig1p  = 0.5 * ( 2.+chi1+np.sqrt( chi1*(4.+chi1) ) )
    sig2m  = 0.5 * ( 2.+chi2-np.sqrt( chi2*(4.+chi2) ) )
    #
    del1p  = ( 6.+2.*chi1+np.sqrt( 3.*chi1*(12.+chi1) ) ) / (6.-chi1)
    del2m  = ( 6.+2.*chi2-np.sqrt( 3.*chi2*(12.+chi2) ) ) / (6.-chi2)   
    #
    phi1p  = ( 12.+5.*chi1+2.*np.sqrt(6)*np.sqrt( chi1*(6.+chi1) ) ) / (12.-chi1)
    phi2m  = ( 12.+5.*chi2-2.*np.sqrt(6)*np.sqrt( chi2*(6.+chi2) ) ) / (12.-chi2)  
    #
    kappac    = 0.
    eta1FDneu = nu1*(1.-1./sig1p)
    eta2FDneu = nu2*( sig2m - 1.)
    rhoDNFDn4 = np.abs( eta1FDneu/eta2FDneu )  
    #
    kappac    = 1.
    eta1FDneu = nu1*(1.-1./sig1p+0.5*kappac*chi1)
    eta2FDneu = nu2*( sig2m - 1.-0.5*kappac*chi2)
    rhoDNFDc4 = np.abs( eta1FDneu/eta2FDneu )  
    #
    eta1FV2   = ( (1./3.+1./chi1)+(1./6.-1./chi1) / del1p )
    eta2FV2   = ( (1./chi2-1./6.)*del2m-(1./chi2+1./3.)   )
    rhoDNFV24 = np.abs( nu1*eta2FV2/(nu2*eta1FV2) )
    #  
    ax = axes[1,0]  
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25) 
    ax.set_title(r"$\rho_{\rm DN}^{({\rm c,FD})}(\kappa_c=0, r=0.1\;{\rm s}^{-1})$",fontsize=10) 
    ax.set_xlim(9.e-06,1000)
    ax.set_ylim(0.45,1.05)
    ax.set_xlabel (r'$\omega$', fontsize=12)
    ax.semilogx(ww,rhoDNFDn1,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$' )    
    ax.semilogx(ww,rhoDNFDn2,linewidth=2.,color='k'   , linestyle='solid' ,label=r'$h = 1\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDn3,linewidth=2.,color='0.5'   , linestyle='dashed' ,label=r'$h = 5\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDn4,linewidth=2.,color='0.5'   , linestyle='solid' ,label=r'$h = 10\;{\rm m}$' )  
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' ) 
    txt = ax.annotate('(d)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))  
    
    ax  = axes[1,1] 
    ax.set_ylim(0.45,1.05)
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)     
    ax.set_title(r"$\rho_{\rm DN}^{({\rm c,FD})}(\kappa_c=1, r=0.1\;{\rm s}^{-1})$",fontsize=10) 
    ax.set_xlabel (r'$\omega$', fontsize=12)
    ax.semilogx(ww,rhoDNFDc1,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$' )    
    ax.semilogx(ww,rhoDNFDc2,linewidth=2.,color='k'   , linestyle='solid' ,label=r'$h = 1\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDc3,linewidth=2.,color='0.5'   , linestyle='dashed' ,label=r'$h = 5\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFDc4,linewidth=2.,color='0.5'   , linestyle='solid' ,label=r'$h = 10\;{\rm m}$' )  
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' )  
    txt = ax.annotate('(e)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))  

    ax = axes[1,2]   
    ax.set_ylim(0.45,1.05)
    ax.set_xlabel (r'$\omega$', fontsize=12)
    ax.set_title(r"$\rho_{\rm DN}^{({\rm c,FV})}(r=0.1\;{\rm s}^{-1})$",fontsize=10) 
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25) 

    ax.semilogx(ww,rhoDNFV21,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$' )    
    ax.semilogx(ww,rhoDNFV22,linewidth=2.,color='k'   , linestyle='solid' ,label=r'$h = 1\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFV23,linewidth=2.,color='0.5'   , linestyle='dashed' ,label=r'$h = 5\;{\rm m}$' ) 
    ax.semilogx(ww,rhoDNFV24,linewidth=2.,color='0.5'   , linestyle='solid' ,label=r'$h = 10\;{\rm m}$' )  
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' )  
    txt = ax.annotate('(f)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))  
    fig.tight_layout()   
    show_or_save("fig_rhoDN_space")

def fig_DNInteraction():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.space.FD_corr import FiniteDifferencesCorr
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.LAMBDA_1 = 1e9  # extremely high lambda is a Dirichlet condition
    builder.LAMBDA_2 = 0. # lambda=0 is a Neumann condition
    # builder.LAMBDA_1 = 1.11 # optimal parameters for corr=0, N=3000
    # builder.LAMBDA_2 = -0.76
    builder.D1 = 1.
    builder.D2 = 2.
    builder.R = 0.
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(8,4.2))    
    plt.subplots_adjust(left=.04, bottom=.10, right=.99, top=.92, wspace=0.05, hspace=0.4)
    dt = builder.DT
    REACTION_COEFF = 0.1
        
    discretizations = {}
    time_scheme = BackwardEuler

    theorical_convergence_factors = {}
    theorical_convergence_factors["BEFV"] = {}
    theorical_convergence_factors["BEFD"] = {}
    theorical_convergence_factors["BEFD_R"] = {}
    theorical_convergence_factors["SimplePade"] = {}
    theorical_convergence_factors["LowTildePade"] = {}
    theorical_convergence_factors["LowTildePade_R"] = {}

    def low_tilde_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1) - b/2 * (z-1)**2
    def simple_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1)

    all_dt = (1e-3, 1e-2, 1e-1, 1e0, 1e1)
    style = {all_dt[4] : {'col':'k', 'ls':'solid', 'lw':1.4, 'legend':r"$\Gamma=10$"},
            all_dt[3] : {'col':'.1', 'ls':'dashed', 'lw':1.55, 'legend':r"$\Gamma=1$"},
            all_dt[2] : {'col':'.5', 'ls':'solid', 'lw':1.7, 'legend':r"$\Gamma=10^{-1}$"},
            all_dt[1] : {'col':'.5', 'ls':'dashed', 'lw':1.9, 'legend':r"$\Gamma=10^{-2}$"},
            all_dt[0] : {'col':'r', 'ls':'solid', 'lw':1.2, 'legend':r"$\Gamma=\nu_1\frac{\Delta t}{h^2}=10^{-3}$"}}

    for dt in all_dt:
        builder.DT = dt
        assert REACTION_COEFF * builder.DT <= 1
        axis_freq = get_discrete_freq(1000/dt, builder.DT)

        dis = builder.build(BackwardEuler, FiniteDifferencesNaive)
        theorical_convergence_factors["BEFD"][dt] = dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))

        dis.R = REACTION_COEFF
        theorical_convergence_factors["BEFD_R"][dt] = dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))

        dis = builder.build(BackwardEuler, QuadSplinesFV)
        theorical_convergence_factors["BEFV"][dt] = dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))

        theorical_convergence_factors["SimplePade"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=simple_gamma)

        builder.R = REACTION_COEFF
        theorical_convergence_factors["LowTildePade_R"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=low_tilde_gamma)
        builder.R = 0.

        theorical_convergence_factors["LowTildePade"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=low_tilde_gamma)

        
        col = style[dt]['col']
        ls = style[dt]['ls']
        lw = style[dt]['lw']
        legend = style[dt]['legend']
        axes[0,0].semilogx(axis_freq, theorical_convergence_factors["BEFV"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[0,1].semilogx(axis_freq, theorical_convergence_factors["BEFD"][dt], linestyle=ls, color=col, label=legend, linewidth=lw)
        axes[0,2].semilogx(axis_freq, theorical_convergence_factors["BEFD_R"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[1,0].semilogx(axis_freq, theorical_convergence_factors["SimplePade"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[1,1].semilogx(axis_freq, theorical_convergence_factors["LowTildePade"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[1,2].semilogx(axis_freq, theorical_convergence_factors["LowTildePade_R"][dt], linestyle=ls, color=col, linewidth=lw)

    axes[0,0].set_title(r"${\rho}^{(BE, FV)}_{DN}$", fontsize=10)
    axes[0,1].set_title(r"${\rho}^{(BE, FD)}_{DN}$", fontsize=10)
    axes[0,2].set_title(r"${\rho}^{(BE, FD)}_{DN}, r=0.1 s^{-1}$", fontsize=10)
    axes[1,0].set_title(r"${\rho}^{(P2, FD)}_{DN}, \gamma=\gamma_{\rm extr}$", fontsize=10)
    axes[1,1].set_title(r"${\rho}^{(P2, FD)}_{DN}, \gamma=\gamma_{\rm imit}$", fontsize=10)
    axes[1,2].set_title(r"${\rho}^{(P2, FD)}_{DN}, \gamma=\gamma_{\rm imit}, r=0.1 s^{-1}$", fontsize=10)

    for i in (0,1,2):
        axes[1,i].set_xlabel(r"$\omega$", fontsize=12)

    letter_fig = {(0,0): '(a)', 
            (0,1): '(b)', 
            (0,2): '(c)', 
            (1,0): '(d)', 
            (1,1): '(e)', 
            (1,2): '(f)', }
    for i,j in ((0,0), (1,0), (0,1), (1,1), (0,2), (1,2)):
        txt = axes[i,j].annotate(letter_fig[(i,j)],xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
        txt.set_bbox(dict(facecolor='white',alpha=1.))  
        axes[i,j].grid(color='k', linestyle=':', linewidth=0.25)
    axes[0,1].legend(loc="upper right",prop={'size':7}, handlelength=2)
    fig.tight_layout()   
    show_or_save("fig_interactionsDN")

def robin_parameters_discrete_space(builder, N, dt, scheme="FD"):
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.time.backward_euler import BackwardEuler # silent class
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    if scheme == "FD":
        space_dis = FiniteDifferencesNaive
    elif scheme == "FV":
        space_dis = QuadSplinesFV
    else:
        raise

    from scipy.optimize import minimize_scalar, minimize
    axis_freq = get_discrete_freq(N, dt)
    def to_minimize_onesided(p):
        dis = builder.build(BackwardEuler, space_dis)
        dis.LAMBDA_1 = p
        dis.LAMBDA_2 = -p
        return np.max(np.abs(dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_operators=float('inf'), order_equations=float('inf'))))

    def to_minimize_twosided(parameters):
        dis = builder.build(BackwardEuler, space_dis)
        dis.LAMBDA_1 = parameters[0]
        dis.LAMBDA_2 = parameters[1]
        return np.max(np.abs(dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_operators=float('inf'), order_equations=float('inf'))))

    res_onesided = minimize_scalar(fun=to_minimize_onesided)
    p_opti = res_onesided.x
    res_twosided = minimize(fun=to_minimize_twosided,
            x0=np.array((p_opti, -p_opti)), method='Nelder-Mead')
    return res_twosided

def fig_RRInteraction():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.space.FD_corr import FiniteDifferencesCorr
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.D1 = 1.
    builder.D2 = 2.
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(8,4.2))    
    plt.subplots_adjust(left=.04, bottom=.10, right=.99, top=.92, wspace=0.05, hspace=0.4)
    axes[0,0].set_ylim(top=0.6, bottom=0.) #sharey activated : see ax[1].set_xlim
    dt = builder.DT
    REACTION_COEFF = 0.1 # for cases with reaction != 0
    all_dt = (1e1, 1e0, 1e-1, 1e-2, 1e-3)
    style = {all_dt[0] : {'col':'k', 'ls':'solid', 'lw':1.4, 'legend':r"$\Gamma=10$"},
            all_dt[1] : {'col':'.1', 'ls':'dashed', 'lw':1.55, 'legend':r"$\Gamma=1$"},
            all_dt[2] : {'col':'.5', 'ls':'solid', 'lw':1.7, 'legend':r"$\Gamma=10^{-1}$"},
            all_dt[3] : {'col':'.5', 'ls':'dashed', 'lw':1.9, 'legend':r"$\Gamma=10^{-2}$"},
            all_dt[4] : {'col':'r', 'ls':'solid', 'lw':1., 'legend':r"$\Gamma=\nu_1\frac{\Delta t}{h^2}=10^{-3}$"}}
        
    discretizations = {}
    time_scheme = BackwardEuler

    theorical_convergence_factors = {}
    theorical_convergence_factors["BEFV"] = {}
    theorical_convergence_factors["BEFD"] = {}
    theorical_convergence_factors["BEFD_R"] = {}
    theorical_convergence_factors["SimplePade"] = {}
    theorical_convergence_factors["LowTildePade"] = {}
    theorical_convergence_factors["LowTildePade_R"] = {}

    ############## OPTIMISATION #####################
    dtmin = 1e-3
    builder.R = 0.
    builder.DT = dtmin
    #optimisation for FD:
    res_twosided = memoised(robin_parameters_discrete_space, builder=builder, N=1000/dtmin, dt=dtmin)
    theorical_convergence_factors["BEFD"]["robin"] = res_twosided.x
    #optimisation for FD with Reaction:

    builder.R = REACTION_COEFF
    theorical_convergence_factors["BEFD_R"]["robin"] =memoised(robin_parameters_discrete_space, builder=builder, N=1000/dtmin, dt=dtmin).x

    #optimisation for FV:
    builder.R = 0.
    theorical_convergence_factors["BEFV"]["robin"] = memoised(robin_parameters_discrete_space, builder=builder, N=1000/dtmin, dt=dtmin, scheme="FV").x

    #optimisation for FD with r:

    def low_tilde_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1) - b/2 * (z-1)**2
    def simple_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1)

    for dt in all_dt:
        builder.DT = dt
        assert REACTION_COEFF * builder.DT <= 1
        axis_freq = get_discrete_freq(5000/dt, builder.DT)

        dis = builder.build(BackwardEuler, FiniteDifferencesNaive)
        dis.LAMBDA_1, dis.LAMBDA_2 = theorical_convergence_factors["BEFD"]["robin"]
        theorical_convergence_factors["BEFD"][dt] = dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))

        col = style[dt]['col']
        ls = style[dt]['ls']
        lw = style[dt]['lw']
        legend = style[dt]['legend']

        dis.R = REACTION_COEFF
        dis.LAMBDA_1, dis.LAMBDA_2 = theorical_convergence_factors["BEFD_R"]["robin"]
        theorical_convergence_factors["BEFD_R"][dt] = dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))

        dis = builder.build(BackwardEuler, QuadSplinesFV)
        dis.LAMBDA_1, dis.LAMBDA_2 = theorical_convergence_factors["BEFV"]["robin"]
        theorical_convergence_factors["BEFV"][dt] = dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))

        builder.LAMBDA_1, builder.LAMBDA_2 = theorical_convergence_factors["BEFD"]["robin"]
        theorical_convergence_factors["SimplePade"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=simple_gamma)

        builder.R = REACTION_COEFF
        builder.LAMBDA_1, builder.LAMBDA_2 = theorical_convergence_factors["BEFD_R"]["robin"]
        theorical_convergence_factors["LowTildePade_R"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=low_tilde_gamma)
        builder.R = 0.

        builder.LAMBDA_1, builder.LAMBDA_2 = theorical_convergence_factors["BEFD"]["robin"]
        theorical_convergence_factors["LowTildePade"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=low_tilde_gamma)

        axes[0,0].semilogx(axis_freq, theorical_convergence_factors["BEFV"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[0,1].semilogx(axis_freq, theorical_convergence_factors["BEFD"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[0,2].semilogx(axis_freq, theorical_convergence_factors["BEFD_R"][dt], linestyle=ls, color=col, label=legend, linewidth=lw)
        axes[1,0].semilogx(axis_freq, theorical_convergence_factors["SimplePade"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[1,1].semilogx(axis_freq, theorical_convergence_factors["LowTildePade"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[1,2].semilogx(axis_freq, theorical_convergence_factors["LowTildePade_R"][dt], linestyle=ls, color=col, linewidth=lw)

    axes[0,0].set_title(r"${\rho}^{(BE, FV)}_{RR}$", fontsize=10)
    axes[0,1].set_title(r"${\rho}^{(BE, FD)}_{RR}$", fontsize=10)
    axes[0,2].set_title(r"${\rho}^{(BE, FD)}_{RR}, r=0.1 s^{-1}$", fontsize=10)
    axes[1,0].set_title(r"${\rho}^{(P2, FD)}_{RR}, \gamma=\gamma_{\rm extr}$", fontsize=10)
    axes[1,1].set_title(r"${\rho}^{(P2, FD)}_{RR}, \gamma=\gamma_{\rm imit}$ ", fontsize=10)
    axes[1,2].set_title(r"${\rho}^{(P2, FD)}_{RR}, \gamma=\gamma_{\rm imit}, r=0.1 s^{-1}$", fontsize=10)

    for i in (0,1,2):
        axes[1,i].set_xlabel(r"$\omega$", fontsize=12)
    letter_fig = {(0,0): '(a)', 
            (0,1): '(b)', 
            (0,2): '(c)', 
            (1,0): '(d)', 
            (1,1): '(e)', 
            (1,2): '(f)', }

    for i,j in ((0,0), (1,0), (0,1), (1,1), (0,2), (1,2)):
        txt = axes[i,j].annotate(letter_fig[(i,j)],xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
        axes[0,0].set_xlim(3.1415/1000,9e2)
        txt.set_bbox(dict(facecolor='white',alpha=1.))  
        axes[i,j].grid(color='k', linestyle=':', linewidth=0.25)
    handles, labels = axes[0,2].get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = list(reversed(labels)), list(reversed(handles))
    axes[0,2].legend(handles, labels, loc="upper right",prop={'size':7}, handlelength=2)

    fig.tight_layout()
    show_or_save("fig_interactionsRR")

def wAndRhoPadeRR(builder, gamma=None):
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.R
    nu_1 = builder.D1
    nu_2 = builder.D2
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2

    def get_z_s(w):
        z = np.exp(-1j*w*dt)
        return z, (z - 1)/(z*dt)

    if gamma is None:
        def gamma(w):
            z, _ = get_z_s(w)
            return z - b*(z-1) - b/2 * (z-1)**2

    def square_root_interior(w):
        z, s = get_z_s(w)
        return 1j*np.sqrt(-1*(1+(a*dt*s)**2 - (a**2+1)*dt*s))

    def sigma_plus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s +a**2*dt*r + square_root_interior(w))/(a*np.sqrt(dt*nu))

    def sigma_minus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s +a**2*dt*r - square_root_interior(w))/(a*np.sqrt(dt*nu))

    N = 300
    w = get_discrete_freq(N, dt)

    sigma_1 = sigma_minus(w, nu_1)
    sigma_2 = - sigma_minus(w, nu_2)
    sigma_3 = sigma_plus(w, nu_1)
    sigma_4 = -sigma_plus(w, nu_2)
    assert (np.real(sigma_1) > 0).all()
    assert (np.real(sigma_2) < 0).all()
    assert (np.real(sigma_3) > 0).all()
    assert (np.real(sigma_4) < 0).all()

    z, s = get_z_s(w)
    mu_1 = z*(1 + r*dt*b - b*dt*nu_1*sigma_1**2)
    mu_2 = z*(1 + r*dt*b - b*dt*nu_2*sigma_2**2)
    mu_3 = z*(1 + r*dt*b - b*dt*nu_1*sigma_3**2)
    mu_4 = z*(1 + r*dt*b - b*dt*nu_2*sigma_4**2)
    assert (np.linalg.norm(mu_1 - mu_2) < 1e-10) # mu_1 == mu_2
    assert (np.linalg.norm(mu_3 - mu_4) < 1e-10) # mu_3 == mu_4
    gamma_t = (mu_1 - gamma(w))/(mu_1 - mu_3)

    varrho = ((L1 + nu_2*sigma_2)/(L2 + nu_2*sigma_2) * (1 - gamma_t) + \
             (L1 + nu_2*sigma_4)/(L2 + nu_2*sigma_4) * gamma_t) * \
             ((L2 + nu_1*sigma_1)/(L1 + nu_1*sigma_1) * (1 - gamma_t) + \
             (L2 + nu_1*sigma_3)/(L1 + nu_1*sigma_3) * gamma_t)

    return w, np.abs(varrho)


def fig_compare_discrete_modif():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma as Pade
    from discretizations.time.backward_euler import BackwardEuler as BE
    from discretizations.space.FD_naive import FiniteDifferencesNaive as FD
    from discretizations.space.FD_extra import FiniteDifferencesExtra as FD
    from discretizations.space.quad_splines_fv import QuadSplinesFV as FV
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra
    fig, axes = plt.subplots(2, 2, figsize=[6.4, 4.4], sharex=False, sharey=True)
    plt.subplots_adjust(left=.11, bottom=.28, right=.99, top=.92, wspace=0.19, hspace=0.34)
    COLOR_CONT = '#FF3333FF'
    COLOR_CONT_FD = '#AA0000FF'
    COLOR_MODIF = '#000000FF'

    for r, axes in ((0, axes[0,:]), (.1, axes[1,:])):
        setting = Builder()
        setting.R = r

        setting.LAMBDA_1 = 1.
        setting.LAMBDA_2 = -1.
        setting.M1 = 200
        setting.M2 = 200
        setting.D1 = 1.
        setting.D2 = 1.
        dt = setting.DT
        # N = 30
        # axis_freq = get_discrete_freq(N, setting.DT)
        axis_freq = np.exp(np.linspace(-5, np.log(pi), 10000))/dt

        #########################################################
        # LEFT CANVA: TIME COMPARISON
        #########################################################

        space_dis = FD
        dis = setting.build(Pade, space_dis)

        cont_time = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=0, order_operators=0) #continuous in time
        modif_time = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=2, order_equations=0, order_operators=0) # modified in time

        b = 1+1/np.sqrt(2)
        def gamma_order2(z):
            return z - b*(z-1) - b/2 * (z-1)**2

        def gamma_order1(z):
            return z - b*(z-1)

        ######################
        # TIME SCHEME : GAMMA ORDER 2:
        ######################
        axis = axes[1]

        full_discrete = rho_Pade_c(setting, w=axis_freq, gamma=gamma_order2) # disccrete in time
        lineg2, = axis.semilogx(axis_freq*dt, np.abs(full_discrete - modif_time)/np.abs(full_discrete), linewidth='1.1',
                color=COLOR_MODIF, linestyle='solid')
        axis.semilogx(axis_freq*dt, np.abs(full_discrete - cont_time)/np.abs(full_discrete), linewidth='1.1',
                color=COLOR_CONT, linestyle='solid')

        ######################
        # TIME SCHEME : GAMMA ORDER 1:
        ######################

        full_discrete = rho_Pade_c(setting, w=axis_freq, gamma=gamma_order1) # disccrete in time

        lineg1, = axis.semilogx(axis_freq*dt, np.abs(full_discrete - modif_time)/np.abs(full_discrete), linewidth='1.1',
                color=COLOR_MODIF, linestyle='dashed')
        axis.semilogx(axis_freq*dt, np.abs(full_discrete - cont_time)/np.abs(full_discrete), linewidth='1.1',
                color=COLOR_CONT, linestyle='dashed')

        ########################
        # TIME SCHEME : Backward Euler
        #########################
        dis = setting.build(BE, space_dis)

        modif_time = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=2, order_equations=0, order_operators=0) # modified in time
        full_discrete = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=float('inf'), order_equations=0, order_operators=0) # discrete in time

        linebe, = axis.semilogx(axis_freq*dt, np.abs(full_discrete - modif_time)/np.abs(full_discrete),
                color=COLOR_MODIF, linestyle=':', linewidth="2.3")
        axis.semilogx(axis_freq*dt, np.abs(full_discrete - cont_time)/np.abs(full_discrete),
                color=COLOR_CONT, linestyle=':', linewidth="2.3")

        axis.grid(True,color='k', linestyle='dotted', linewidth=0.25)
        axis.set_xlim(left=0.9e-2, right=.7)
        #axis.set_ylim(top=0.1, bottom=0.) #sharey activated : see axis.set_xlim
        Title = r'Semi-discrete in time' #Title = r'$d\rho_{\rm RR}^{\rm (\cdot,c)}$'
        #x_legend= r'$\left| \rho_{\rm RR}^{\rm (\cdot,c)} - \rho_{\rm RR}^{\rm (Discrete,c)}\right|/\left|\rho_{\rm RR}^{\rm (Discrete,c)}\right| $'
        if r == 0:
            axis.set_title(Title)
            axis.set_xticklabels([])
        else:
            axis.set_xlabel(r'$\omega\Delta t$')

        #########################################################
        # RIGHT CANVA: SPACE COMPARISON
        #########################################################
        time_dis = BE # we don't really care, since everything is continuous in time now

        ######################
        # SPACE SCHEME : FV
        ######################
        dis = setting.build(time_dis, FV)

        cont_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=0, order_operators=float('inf')) #continuous in time

        modif_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=2, order_operators=float('inf')) # modified in time

        full_discrete = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=float('inf'), order_operators=float('inf'))

        axis = axes[0]

        axis.semilogx(axis_freq*dt, np.abs(full_discrete - modif_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_MODIF, linestyle='solid')
        axis.semilogx(axis_freq*dt, np.abs(full_discrete - cont_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_CONT, linestyle='solid')

        ######################
        # SPACE SCHEME : FD
        ######################
        dis = setting.build(time_dis, FD)

        cont_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=0, order_operators=float('inf')) #continuous in time

        modif_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=2, order_operators=float('inf')) # modified in time

        full_discrete = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=float('inf'), order_operators=float('inf'))

        axis.semilogx(axis_freq*dt, np.abs(full_discrete - modif_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_MODIF, linestyle='dashed')
        axis.semilogx(axis_freq*dt, np.abs(full_discrete - cont_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_CONT_FD, linestyle='dashed')

        axis.grid(True,color='k', linestyle='dotted', linewidth=0.25)
        axis.set_xlim(left=2e-2, right=3)
        axis.set_ylim(top=0.03, bottom=0.)
        Title = r'Semi-discrete in space' #r'$d\rho_{\rm RR}^{\rm (c, \cdot)}$'
        axis.set_ylabel(r'$r=' + str(r) + r'\;{\rm s}^{-1}$')
        if r == 0:
            #axis.legend()
            axis.set_title(Title)
            axis.set_xticklabels([])
        else:
            axis.set_xlabel(r'$\omega$')


    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    custom_lines = [
                    Patch(facecolor=COLOR_MODIF),
                    Patch(facecolor=COLOR_CONT),
                    Line2D([0],[0],color="w"),
                    Line2D([0], [0], lw=2., color='black'),
                    Line2D([0], [0], linestyle='dashed', lw=2., color='black'),
                    Line2D([0],[0],color="w"),
                    Line2D([0], [0], lw=1.2, color='black'),
                    Line2D([0], [0], linestyle='dashed', lw=1.2, color='black'),
                    Line2D([0], [0], linestyle='dotted', lw=1.6, color='black'),
                    ]

    custom_labels = [
            r'$(\delta \rho)_\mathbf{m}^{\rm (\cdot, \cdot)}$',
            r'$(\delta \rho)_\mathbf{c}^{\rm (\cdot, \cdot)}$',
            r"",
            r"$(\delta\rho)^{\rm (c, \mathbf{FV})}$", r"$(\delta\rho)^{\rm (c, \mathbf{FD})}$",
            r"",
            r"$(\delta\rho)^{\rm (\mathbf{P2}, c)}$" + ", " + r"$\gamma = z - \beta (z-1)$" + r"$- \beta(\beta-1)^2(z-1)^2$",
            r"$(\delta\rho)^{\rm (\mathbf{P2}, c)}$" + ", " + r"$\gamma = z - \beta (z-1)$",
            r"$(\delta\rho)^{\rm (\mathbf{BE}, c)}$",]
    fig.legend(custom_lines, custom_labels, loc=(0.10, 0.), ncol=3)
    fig.tight_layout()   
    fig.subplots_adjust(bottom=.28, wspace=.2)

    show_or_save("fig_compare_discrete_modif")

def fig_optiRates():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    fig, axes = plt.subplots(2, 2, figsize=[6.4*1.4, 4.8*1.4], sharex=True, sharey=True)
    axes[0,0].grid()
    axes[1,0].grid()
    axes[0,1].grid()
    axes[1,1].grid()
    axes[1,1].set_ylim(bottom=0.095, top=0.3) # all axis are shared

    caracs = {}
    caracs["continuous"] = {'color':'#00AF80', 'width':0.7, 'nb_+':9}
    caracs["modified"] = {'color':'#0000FF', 'width':1.3, 'nb_+':12}
    caracs["discrete"] = {'color':'#000000', 'width':0.7, 'nb_+':15}


    fig.suptitle("Optimized convergence rates with different methods")
    fig.subplots_adjust(left=0.07, bottom=0.12, right=0.98, top=0.92, wspace=0.13, hspace=0.16)
    #####################################
    # PADE
    ####################################
    from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra

    time_dis, space_dis = PadeLowTildeGamma, FiniteDifferencesExtra

    def rho_c_FD_extra(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=0, order_operators=float('inf'),
                    order_equations=float('inf'))

    def rho_m_FD(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=1, order_operators=float('inf'),
                    order_equations=float('inf'))

    optiRatesGeneral(axes[0,0], rho_c_FD_extra, rho_m_FD, rho_Pade_FD_extra, time_dis, space_dis, "Pade", caracs=caracs)

    ########################################
    # BE
    ###########################################
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    time_dis, space_dis = BackwardEuler, FiniteDifferencesExtra
    def rho_c_FD_extra(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=0, order_operators=float('inf'),
                    order_equations=float('inf'))
    def rho_BE_FD_extra(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=float('inf'), order_operators=float('inf'),
                    order_equations=float('inf'))

    def rho_m_FD(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=1, order_operators=float('inf'),
                    order_equations=float('inf'))
    
    optiRatesGeneral(axes[0,1], rho_c_FD_extra, rho_m_FD, rho_BE_FD_extra, time_dis, space_dis, "Backward Euler", caracs=caracs)

    #############################################
    # FD
    #######################################

    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    time_dis, space_dis = BackwardEuler, FiniteDifferencesExtra
    def rho_BE_c(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=float('inf'), order_operators=0,
                    order_equations=0)
    def rho_BE_FD_extra(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=float('inf'), order_operators=float('inf'),
                    order_equations=float('inf'))

    def rho_m_FD(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=float('inf'), order_operators=1,
                    order_equations=1)
    
    optiRatesGeneral(axes[1,0], rho_BE_c, rho_m_FD, rho_BE_FD_extra, time_dis, space_dis, "Finite Differences", caracs=caracs)

    ###########################
    # FV
    ##########################

    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    time_dis, space_dis = BackwardEuler, QuadSplinesFV
    def rho_BE_c(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=float('inf'), order_operators=0,
                    order_equations=0)
    def rho_BE_FV(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=float('inf'), order_operators=float('inf'),
                    order_equations=float('inf'))

    def rho_m_FV(builder, axis_freq):
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
                    order_time=float('inf'), order_operators=1,
                    order_equations=2)
    
    optiRatesGeneral(axes[1,1], rho_BE_c, rho_m_FV, rho_BE_FV, time_dis, space_dis, "Finite Volumes", caracs=caracs)


    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=caracs["continuous"]["color"], lw=caracs["continuous"]["width"]),
                    Line2D([0], [0], color=caracs["modified"]["color"], lw=caracs["modified"]["width"]),
                    Line2D([0], [0], color=caracs["discrete"]["color"], lw=caracs["discrete"]["width"]),
                    Line2D([0], [0], marker="^", markersize=6., linewidth=0.,
                        color=caracs["discrete"]["color"]) ]
    custom_labels = ["Continuous", "Modified", "Discrete", "Prediction"]
    fig.legend(custom_lines, custom_labels, loc=(0.2, 0.), ncol=5, handlelength=2)

    show_or_save("fig_optiRates")

def optiRatesGeneral(axes, continuous_rate, modified_rate, discrete_rate, time_dis, space_dis, name_method="Unknown discretization", caracs={}, **args_for_discretization):
    """
        Creates a figure comparing analysis methods for a discretization.
        the functions "rate" should be:
        def discrete_rate(builder, axis_freq):
            dis = builder.build(time_dis, space_dis)
            return dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))
    """

    setting = Builder()
    setting.M1 = 200
    setting.M2 = 200
    setting.D1 = .5
    setting.D2 = 1.
    setting.R = 1e-2
    setting.DT = 1.
    N = 30000
    axis_freq = get_discrete_freq(N, setting.DT)

    axes.set_xlabel("$\\omega \\Delta t$")
    axes.set_ylabel(r"${\rho}_{RR}$")
    #axes.set_title("Optimized convergence rates with different methods (" + name_method + ")")
    axes.set_title(name_method)

    def rate_onesided(lam):
        builder = setting.copy()
        builder.LAMBDA_1 = lam
        builder.LAMBDA_2 = -lam
        return np.max(continuous_rate(builder, axis_freq))
    from scipy.optimize import minimize_scalar, minimize
    optimal_lam = minimize_scalar(fun=rate_onesided)
    x0_opti = (optimal_lam.x, -optimal_lam.x)

    for discrete_factor, names in zip((continuous_rate, modified_rate, discrete_rate), caracs):
        if names == "modified":
            freq_opti = axis_freq # get_discrete_freq(N//10, setting.DT*10.)
        else:
            freq_opti = axis_freq


        def rate_twosided(lam):
            builder = setting.copy()
            builder.LAMBDA_1 = lam[0]
            builder.LAMBDA_2 = lam[1]
            return np.max(discrete_factor(builder, freq_opti))

        optimal_lam = minimize(method='Nelder-Mead',
                fun=rate_twosided, x0=x0_opti)
        if names == "continuous":
            x0_opti = optimal_lam.x
        print(optimal_lam)
        setting.LAMBDA_1 = optimal_lam.x[0]
        setting.LAMBDA_2 = optimal_lam.x[1]

        builder = setting.copy()
        if REAL_FIG:
            alpha_w = memoised(Builder.frequency_cv_factor, builder,
                    time_dis, space_dis, N=N, number_samples=32, NUMBER_IT=2) # WAS 7 IN CACHE
            convergence_factor = np.abs((alpha_w[2] / alpha_w[1]))
        else:
            convergence_factor = discrete_rate(setting, axis_freq)


        axis_freq_predicted = np.exp(np.linspace(np.log(min(np.abs(axis_freq))), np.log(axis_freq[-1]), caracs[names]["nb_+"]))

        # LESS IMPORTANT CURVE : WHAT IS PREDICTED

        axes.semilogx(axis_freq * setting.DT, convergence_factor, linewidth=caracs[names]["width"], label= "$p_1, p_2 =$ ("+ str(optimal_lam.x[0])[:4] +", "+ str(optimal_lam.x[1])[:5] + ")", color=caracs[names]["color"]+"90")
        if names =="discrete":
            axes.semilogx(axis_freq_predicted * setting.DT, discrete_factor(setting, axis_freq_predicted), marker="^", markersize=6., linewidth=0., color=caracs[names]["color"])# , label="prediction")
        else:
            axes.semilogx(axis_freq_predicted * setting.DT, discrete_factor(setting, axis_freq_predicted), marker="^", markersize=6., linewidth=0., color=caracs[names]["color"])

        axes.semilogx(axis_freq * setting.DT, np.ones_like(axis_freq)*max(convergence_factor), linestyle="dashed", linewidth=caracs[names]["width"], color=caracs[names]["color"]+"90")


    axes.legend( loc=(0., 0.), ncol=1 )
    axes.set_xlim(left=1e-4, right=3.4)
    #axes.set_ylim(bottom=0)



######################################################
# Utilities for analysing, representing discretizations
######################################################

class Builder():
    """
        interface between the discretization classes and the plotting functions.
        The main functions is build: given a space and a time discretizations,
        it returns a class which can be used with all the available functions.

        The use of anonymous classes forbids to use a persistent cache.
        To shunt this problem, function @frequency_cv_factor allows to
        specify the time and space discretizations at the last time, so
        the function @frequency_cv_factor can be stored in cache.

        To use this class, instanciate builder = Builder(),
        choose appropriate arguments of builder:
        builder.DT = 0.1
        builder.LAMBDA_2 = -0.3
        and then build all the schemes you want with theses parameters:
        dis_1 = builder.build(BackwardEuler, FiniteDifferencesNaive)
        dis_2 = builder.build(ThetaMethod, QuadSplinesFV)
        The comparison is thus then quite easy
    """
    def __init__(self): # changing defaults will result in needing to recompute all cache
        self.COURANT_NUMBER = 1.
        self.M1 = 200
        self.M2 = 200
        self.SIZE_DOMAIN_1 = 200
        self.SIZE_DOMAIN_2 = 200
        self.D1 = 1.
        self.D2 = 1.
        self.DT = self.COURANT_NUMBER * (self.SIZE_DOMAIN_1 / self.M1)**2 / self.D1
        self.A = 0.
        self.R = 0.
        self.LAMBDA_1 = 1e9
        self.LAMBDA_2 = 0.

    def build_scheme(self, scheme):
        """
            To use with a space-time discretization
        """
        return scheme(A=self.A, C=self.R,
                              D1=self.D1, D2=self.D2,
                              M1=self.M1, M2=self.M2,
                              SIZE_DOMAIN_1=self.SIZE_DOMAIN_1,
                              SIZE_DOMAIN_2=self.SIZE_DOMAIN_2,
                              LAMBDA_1=self.LAMBDA_1,
                              LAMBDA_2=self.LAMBDA_2,
                              DT=self.DT)

    def build(self, time_discretization, space_discretization):
        """
            Given two abstract classes of a time and space discretization,
            build a scheme.
        """
        class AnonymousScheme(time_discretization, space_discretization):
            def __init__(self, *args, **kwargs):
                space_discretization.__init__(self, *args, **kwargs)
                time_discretization.__init__(self, *args, **kwargs)

        return AnonymousScheme(A=self.A, C=self.R,
                              D1=self.D1, D2=self.D2,
                              M1=self.M1, M2=self.M2,
                              SIZE_DOMAIN_1=self.SIZE_DOMAIN_1,
                              SIZE_DOMAIN_2=self.SIZE_DOMAIN_2,
                              LAMBDA_1=self.LAMBDA_1,
                              LAMBDA_2=self.LAMBDA_2,
                              DT=self.DT)

    def frequency_cv_factor_spacetime_scheme(self, discretization, linear=True, **kwargs):
        discretization = self.build_scheme(discretization)
        if linear:
            return frequency_simulation(discretization, **kwargs)
        else:
            raise NotImplementedError("Just a linear regression in 2D")
            return matrixlinear_frequency_simulation(discretization, **kwargs)

    def frequency_cv_rate(self, time_discretization, space_discretization, linear=True, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        if linear:
            return simulation_cv_rate_linear(discretization, **kwargs)
        else:
            return simulation_cv_rate_matrixlinear(discretization, **kwargs)

    def simulation_firstlevels(self, time_discretization, space_discretization, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        return simulation_firstlevels(discretization, **kwargs)

    def frequency_cv_factor(self, time_discretization, space_discretization, linear=True, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        if linear:
            return frequency_simulation(discretization, **kwargs)
        else:
            return matrixlinear_frequency_simulation(discretization, **kwargs)

    def frequency_eigenvalues(self, time_discretization, space_discretization, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        return eigenvalues_matrixlinear_frequency_simulation(discretization, **kwargs)

    def robin_robin_theorical_cv_factor(self, time_discretization, space_discretization, *args, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        return discretization.analytic_robin_robin_modified(*args, **kwargs)

    def copy(self):
        ret = Builder()
        ret.__dict__ = self.__dict__.copy()
        return ret

    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())))

    def __repr__(self):
        return repr(sorted(self.__dict__.items()))

DEFAULT = Builder()


def get_discrete_freq(N, dt, avoid_zero=True):
    """
        Computation of the frequency axis.
        Z transform gives omega = 2 pi k T / (N).
    """
    N = N + 1 # actually, the results of the simulator contains one more point
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    # Usually, we don't want the zero frequency so we use instead 1/T:
    if avoid_zero:
        all_k[int(N//2)] = .5
    return 2 * np.pi*all_k / N / dt

#############################################
# Utilities for saving, visualizing, calling functions
#############################################


def set_save_to_png():
    global SAVE_TO_PNG
    SAVE_TO_PNG = True
    assert not SAVE_TO_PDF and not SAVE_TO_PGF

def set_save_to_pdf():
    global SAVE_TO_PDF
    SAVE_TO_PDF = True
    assert not SAVE_TO_PGF and not SAVE_TO_PNG

def set_save_to_pgf():
    global SAVE_TO_PGF
    SAVE_TO_PGF = True
    assert not SAVE_TO_PDF and not SAVE_TO_PNG

SAVE_TO_PNG = False
SAVE_TO_PGF = False
SAVE_TO_PDF = False
def show_or_save(name_func):
    """
    By using this function instead plt.show(),
    the user has the possibiliy to use ./figsave name_func
    name_func must be the name of your function
    as a string, e.g. "fig_comparisonData"
    """
    name_fig = name_func[4:]
    directory = "figures_out/"
    if SAVE_TO_PNG:
        print("exporting to directory " + directory)
        import os
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.png')
    elif SAVE_TO_PGF:
        print("exporting to directory " + directory)
        import os
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.pgf')
    elif SAVE_TO_PDF:
        print("exporting to directory " + directory)
        import os
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.pdf')
    else:
        try:
            import matplotlib as mpl
            import os
            os.makedirs(directory, exist_ok=True)
            mpl.rcParams['savefig.directory'] = directory
            fig = plt.get_current_fig_manager()
            fig.canvas.set_window_title(name_fig) 
        except:
            print("cannot set default directory or name")
        plt.show()

"""
    The dictionnary all_figures contains all the functions
    of this module that begins with "fig_".
    When you want to add a figure,
    follow the following rule:
        if the figure is going to be labelled as "fig:foo"
        then the function that generates it should
                                        be named (fig_foo())
    The dictionnary is filling itself: don't try to
    manually add a function.
"""
all_figures = {}

##################################################################################
# Filling the dictionnary all_figures with the functions beginning with "fig_":  #
##################################################################################
# First take all globals defined in this module:
for key, glob in globals().copy().items():
    # Then select the names beginning with fig.
    # Note that we don't check if it is a function,
    # So that a user can give a callable (for example, with functools.partial)
    if key[:3] == "fig":
        all_figures[key] = glob
