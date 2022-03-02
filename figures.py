#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
"""
import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import bisect
import multiprocessing
from scipy.optimize import minimize_scalar, minimize
from memoisation import memoised
from simulator import frequency_simulation, simulation_L2norm
from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FV
from cv_factor_onestep import rho_BE_FD, rho_BE_FV, rho_c_FD, rho_c_FV
from cv_factor_onestep import rho_BE_c, rho_c_c
from ocean_models.ocean_BE_FD import OceanBEFD
from ocean_models.ocean_BE_FV import OceanBEFV
from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
from atmosphere_models.atmosphere_BE_FV import AtmosphereBEFV
from ocean_models.ocean_Pade_FD import OceanPadeFD
from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD

# If set to True, the simulations will run, taking ~2 days.
REAL_FIG = False

def fig_lambda_Pade():
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
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
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,figsize=(10,2))
    ax = axes[0]
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)
    ax.set_xlabel (r'$\omega\Delta t$', fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
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
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
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
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    fig, axes = plt.subplots(1, 1, sharex=False, sharey=True,figsize=(7,3))
    #plt.subplots_adjust(bottom=.18, top=.85)
    ax   = axes
    wmax = np.pi
    wmin = wmax / 200
    ax.set_xlim(wmin,wmax)
    ax.set_ylim(0.53,1.05)
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)
    ax.set_xlabel (r'$\omega$', fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_title(r"$\rho_{\rm DN}^{\rm (P2,c)}$",fontsize=16)


    builder = Builder()
    builder.R = 0.
    builder.D1 = .5
    builder.M1 *= 100
    builder.M2 *= 100
    N=300

    b = 1+1/np.sqrt(2)

    def get_z_s(w):
        z = np.exp(-1j*w*builder.DT)
        return z, (z - 1)/(z*builder.DT)

    def gammaz_highTilde(z):
        return z - b*(z-1)

    def gammaz_lowTilde(z):
        return z - b*(z-1) - b * (b-1)**2 * (z-1)**2

    def gammaw_highTilde(w):
        z, _ = get_z_s(w)
        return gammaz_highTilde(z)

    def gammaw_lowTilde(w):
        z, _ = get_z_s(w)
        return gammaz_lowTilde(z)


    axis_freq = get_discrete_freq(N, builder.DT)

    from ocean_models.ocean_Pade_FD import OceanPadeFD
    from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD
    ocean_r0, atmosphere_r0 = builder.build(OceanPadeFD,
            AtmospherePadeFD)
    builder.R = 0.1
    ocean_r1, atmosphere_r1 = builder.build(OceanPadeFD,
            AtmospherePadeFD)
    builder.R = 0.
    if REAL_FIG:
        alpha_w = memoised(frequency_simulation,
                atmosphere_r0, ocean_r0, number_samples=10,
                NUMBER_IT=1, laplace_real_part=0,
                T=N*builder.DT, gamma="lowTilde")
        cv_factor_r0_low = np.abs((alpha_w[2] / alpha_w[1]))
        alpha_w = memoised(frequency_simulation,
                atmosphere_r0, ocean_r0, number_samples=10,
                NUMBER_IT=1, laplace_real_part=0,
                T=N*builder.DT, gamma="simple")
        cv_factor_r0_high = np.abs((alpha_w[2] / alpha_w[1]))
        alpha_w = memoised(frequency_simulation,
                atmosphere_r1, ocean_r1, number_samples=10,
                NUMBER_IT=1, laplace_real_part=0,
                T=N*builder.DT, gamma="lowTilde")
        cv_factor_r1_low = np.abs((alpha_w[2] / alpha_w[1]))
        alpha_w = memoised(frequency_simulation,
                atmosphere_r1, ocean_r1, number_samples=10,
                NUMBER_IT=1, laplace_real_part=0,
                T=N*builder.DT, gamma="simple")
        cv_factor_r1_high = np.abs((alpha_w[2] / alpha_w[1]))
    else:
        theory = rho_Pade_FD_corr0
        builder.R = 0.1
        cv_factor_r1_low = np.abs(theory(builder,
            axis_freq, gamma=gammaz_lowTilde))
        cv_factor_r1_high = np.abs(theory(builder,
            axis_freq, gamma=gammaz_highTilde))
        builder.R = 0.
        cv_factor_r0_low = np.abs(theory(builder,
            axis_freq, gamma=gammaz_lowTilde))
        cv_factor_r0_high = np.abs(theory(builder,
            axis_freq, gamma=gammaz_highTilde))

    #### validation ####
    nb_subpoints = 12 # plotting only 12 points
    indices = find_indices(axis_freq[N//2+1:], nb_subpoints) + N//2
    lw_observed = 0.45

    ax.semilogx(axis_freq[indices], cv_factor_r0_high[indices],
            'o', color='k', fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
    ax.semilogx(axis_freq[indices], cv_factor_r1_high[indices],
            'o', color='0.5', fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
    ax.semilogx(axis_freq[indices], cv_factor_r0_low[indices],
            'o', color='k', fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
    ax.semilogx(axis_freq[indices], cv_factor_r1_low[indices],
            'o', color='0.5', fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)

    w, varrho = wAndRhoPadeRR(builder, gamma=gammaw_highTilde, N=N)
    ax.semilogx(w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='k', linestyle='solid' ,label=r'$r=0\;{\rm s}^{-1}, \gamma = z - \beta (z-1)$')

    w, varrho = wAndRhoPadeRR(builder, gamma=gammaw_lowTilde, N=N)
    ax.semilogx( w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='k', linestyle='dashed' ,label=r'$r=0\;{\rm s}^{-1}, \gamma = z - \beta (z-1) - \beta(\beta-1)^2 (z-1)^2$')

    builder.R = .1

    w, varrho = wAndRhoPadeRR(builder, gamma=gammaw_highTilde, N=N)
    ax.semilogx( w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='0.5', linestyle='solid' ,label=r'$r=0.1\;{\rm s}^{-1}, \gamma = z - \beta (z-1)$')

    w, varrho = wAndRhoPadeRR(builder, gamma=gammaw_lowTilde, N=N)
    ax.semilogx(w*builder.DT, np.abs(varrho ) ,linewidth=2.,color='0.5', linestyle='dashed' ,label=r'$r=0.1\;{\rm s}^{-1}, \gamma = z - \beta (z-1) - \beta (\beta-1)^2 (z-1)^2$')

    rho_continuous = np.sqrt(builder.D1/builder.D2) * np.ones_like(w)
    ax.semilogx(w*builder.DT, rho_continuous ,linewidth=2.,color='r', linestyle='dashed' ,label=r'$\sqrt{\frac{\nu_1}{\nu_2}}$')
    ax.set_xlabel(r"$\omega\Delta t$")
    ax.legend(loc=2,prop={'size':9},ncol=1,handlelength=2)
    fig.tight_layout()

    show_or_save("fig_rhoDNPade")

def fig_rhoDN_space():
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    builder = Builder()
    builder.R = 0.
    builder.D1 = .5
    builder.D2 = 1.
    builder.DT /= 100
    N = 100000
    axis_freq = get_discrete_freq(N, builder.DT)[N//2+1:]
    def validation(builder, h, atmclass, oceanclass, theory,
            ignore_cached=False, **eventual_k_c):
        local_builder = builder.copy()
        local_builder.set_h(h, vertical_levels=100)
        ocean, atmosphere = local_builder.build(oceanclass,
                atmclass, **eventual_k_c)
        if REAL_FIG:
            alpha_w = memoised(frequency_simulation, atmosphere, ocean,
                    number_samples=10, NUMBER_IT=1, T=N*builder.DT,
                    ignore_cached=ignore_cached)
            convergence_factor = np.abs((alpha_w[2] / alpha_w[1]))
            convergence_factor = convergence_factor[N//2 + 1:]
        else:
            convergence_factor = np.abs(theory(local_builder,
                                        axis_freq, **eventual_k_c))
        return convergence_factor

    nb_subpoints = 12 # plotting only 12 points
    indices = find_indices(axis_freq, nb_subpoints)
    lw_observed = 0.45


    from ocean_models.ocean_BE_FD import OceanBEFD
    from ocean_models.ocean_BE_FV import OceanBEFV
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
    from atmosphere_models.atmosphere_BE_FV import AtmosphereBEFV

    npts   = 1000
    ww     = np.logspace(-6, 3, npts)
    nu1    = 0.5
    nu2    = 1.
    continuous = ww*0. + np.sqrt(nu1/nu2)
    ########### h = 0.1 #################
    dx     = 0.1
    builder.R = 0.1
    validFD0 = {}
    validFD1 = {}
    validFV = {}
    for r in (0., .1):
        for h in (0.1, 1, 5, 10):
            builder.R = r
            validFD0[(h, r)] = validation(builder, h, AtmosphereBEFD,
                    OceanBEFD, rho_BE_FD, k_c=0.)
            validFD1[(h, r)] = validation(builder, h, AtmosphereBEFD,
                    OceanBEFD, rho_BE_FD, k_c=1., ignore_cached=False)
            validFV[(h, r)] = validation(builder, h, AtmosphereBEFV,
                    OceanBEFV, rho_BE_FV)

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
    colors_h = {0.1 : 'k', 1 : 'k', 5 : '0.5', 10 : '0.5'}

    for h in (0.1, 1, 5, 10):
        ax.semilogx(axis_freq[indices],validFD0[(h, 0.)][indices],
            'o', color=colors_h[h], fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' )
    ax.legend(loc=2,prop={'size':7}, handlelength=2)
    txt = ax.annotate('(a)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))

    ax  = axes[0,1]
    ax.set_ylim(0.45,1.05)
    ax.grid(True,color='k', linestyle='dotted', linewidth=0.25)
    ax.set_title(r"$\rho_{\rm DN}^{({\rm c,FD})}(\kappa_c=1, r=0\;{\rm s}^{-1})$",fontsize=10)
    for h in (0.1, 1, 5, 10):
        ax.semilogx(axis_freq[indices],validFD1[(h, 0.)][indices],
            'o', color=colors_h[h], fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)

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
    for h in (0.1, 1, 5, 10):
        ax.semilogx(axis_freq[indices],validFV[(h, 0.)][indices],
            'o', color=colors_h[h], fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
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
    for h in (0.1, 1, 5, 10):
        ax.semilogx(axis_freq[indices],validFD0[(h, 0.1)][indices],
            'o', color=colors_h[h], fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
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
    for h in (0.1, 1, 5, 10):
        ax.semilogx(axis_freq[indices],validFD1[(h, 0.1)][indices],
            'o', color=colors_h[h], fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
    ax.semilogx(ww,rhoDNFDc1,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$')
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

    for h in (0.1, 1, 5, 10):
        ax.semilogx(axis_freq[indices],validFV[(h, 0.1)][indices],
            'o', color=colors_h[h], fillstyle="none",
            markeredgewidth=lw_observed, zorder=0)
    ax.semilogx(ww,rhoDNFV21,linewidth=2.,color='k', linestyle='dashed' ,label=r'$h = 10^{-1}\;{\rm m}$' )
    ax.semilogx(ww,rhoDNFV22,linewidth=2.,color='k'   , linestyle='solid' ,label=r'$h = 1\;{\rm m}$' )
    ax.semilogx(ww,rhoDNFV23,linewidth=2.,color='0.5'   , linestyle='dashed' ,label=r'$h = 5\;{\rm m}$' )
    ax.semilogx(ww,rhoDNFV24,linewidth=2.,color='0.5'   , linestyle='solid' ,label=r'$h = 10\;{\rm m}$' )
    ax.semilogx(ww,continuous,linewidth=1.,color='red'   , linestyle='solid' ,label=r'continuous' )
    txt = ax.annotate('(f)',xy=(0.03,0.075), xycoords='axes fraction',color='k',fontsize=10)
    txt.set_bbox(dict(facecolor='white',alpha=1.))
    fig.tight_layout()
    show_or_save("fig_rhoDN_space")


def find_indices(frequencies, nbpts):
    """
        returns an array of indices such that
        frequencies[indices] is approximately
            geomspace(min, max, nbpts).
        make sure that frequencies is sorted and >0.
    """
    assert frequencies[0] > 0
    ideal_logspace = np.geomspace(frequencies[0],
            frequencies[-2], num=nbpts)
    return np.array([bisect.bisect(frequencies, x)\
                for x in ideal_logspace])

def frequencies_for_optim(N, dt, nbpts):
    axis_freq = get_discrete_freq(N, dt)
    indices = find_indices(axis_freq[N//2:], nbpts) + N//2-1
    return np.copy(axis_freq[indices])


def fig_DNInteraction():
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.LAMBDA_1 = 1e9  # extremely high lambda is a Dirichlet condition
    builder.LAMBDA_2 = 0. # lambda=0 is a Neumann condition
    builder.D1 = .5
    builder.D2 = 1.
    builder.R = 0.
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(8,4.2))
    plt.subplots_adjust(left=.04, bottom=.10, right=.99, top=.92, wspace=0.05, hspace=0.4)
    REACTION_COEFF = 0.1
    T = 1000

    theorical_convergence_factors = {}
    theorical_convergence_factors["BEFV"] = {}
    theorical_convergence_factors["BEFD"] = {}
    theorical_convergence_factors["BEFD_R"] = {}
    theorical_convergence_factors["SimplePade"] = {}
    theorical_convergence_factors["LowTildePade"] = {}
    theorical_convergence_factors["LowTildePade_R"] = {}

    observed_convergence_factors = {}
    observed_convergence_factors["BEFV"] = {}
    observed_convergence_factors["BEFD"] = {}
    observed_convergence_factors["BEFD_R"] = {}
    observed_convergence_factors["SimplePade"] = {}
    observed_convergence_factors["LowTildePade"] = {}
    observed_convergence_factors["LowTildePade_R"] = {}

    def low_tilde_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1) - b/2 * (z-1)**2
    def simple_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1)

    all_dt = (2e-3, 2e-2, 2e-1, 2e0, 2e1)
    style = {all_dt[4] : {'col':'k', 'ls':'solid', 'lw':1.4, 'legend':r"$\Gamma=10$"},
            all_dt[3] : {'col':'.1', 'ls':'dashed', 'lw':1.55, 'legend':r"$\Gamma=1$"},
            all_dt[2] : {'col':'.5', 'ls':'solid', 'lw':1.7, 'legend':r"$\Gamma=10^{-1}$"},
            all_dt[1] : {'col':'.5', 'ls':'dashed', 'lw':1.9, 'legend':r"$\Gamma=10^{-2}$"},
            all_dt[0] : {'col':'r', 'ls':'solid', 'lw':1.2, 'legend':r"$\Gamma=\nu_1\frac{\Delta t}{h^2}=10^{-3}$"}}
    nb_validation_pts = {all_dt[4] : 10,
            all_dt[3] : 6,
            all_dt[2] : 4,
            all_dt[1] : 3,
            all_dt[0] : 3 }

    def validation(builder, N, oceanclass, atmclass, theory, gamma="simple", **possible_kc):
        ocean, atmosphere = builder.build(oceanclass,
                atmclass, **possible_kc)
        axis_freq = get_discrete_freq(N, builder.DT)
        if REAL_FIG:
            alpha_w = memoised(frequency_simulation, atmosphere, ocean,
                    number_samples=10, NUMBER_IT=1, T=N*builder.DT,
                    gamma=gamma)
            convergence_factor = np.abs((alpha_w[2] / alpha_w[1]))
        else:
            if gamma != "simple":
                convergence_factor = np.abs(theory(builder,
                                axis_freq, gamma=low_tilde_gamma))
            else:
                convergence_factor = np.abs(theory(builder,
                                            axis_freq, **possible_kc))
        return convergence_factor

    for dt in all_dt:
        builder.DT = dt
        # assert REACTION_COEFF * builder.DT <= 1
        N = int(T/dt)
        N_validation = 10000
        axis_freq = get_discrete_freq(N, builder.DT)

        theorical_convergence_factors["BEFD"][dt] = np.abs(rho_BE_FD(builder, w=axis_freq, k_c=0))
        observed_convergence_factors["BEFD"][dt] = validation(builder,
                N_validation, OceanBEFD, AtmosphereBEFD, rho_BE_FD, k_c=0)

        builder.R = REACTION_COEFF
        theorical_convergence_factors["BEFD_R"][dt] = np.abs(rho_BE_FD(builder, w=axis_freq, k_c=0))
        observed_convergence_factors["BEFD_R"][dt] = validation(builder,
                N_validation, OceanBEFD, AtmosphereBEFD, rho_BE_FD, k_c=0)
        builder.R = 0.

        theorical_convergence_factors["BEFV"][dt] = np.abs(rho_BE_FV(builder, w=axis_freq))
        observed_convergence_factors["BEFV"][dt] = validation(builder,
                N_validation, OceanBEFV, AtmosphereBEFV, rho_BE_FV)

        theorical_convergence_factors["SimplePade"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=simple_gamma)
        observed_convergence_factors["SimplePade"][dt] = validation(builder,
                N_validation, OceanPadeFD, AtmospherePadeFD, rho_Pade_FD_corr0)

        builder.R = REACTION_COEFF
        theorical_convergence_factors["LowTildePade_R"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=low_tilde_gamma)
        observed_convergence_factors["LowTildePade_R"][dt] = validation(builder,
                N_validation, OceanPadeFD, AtmospherePadeFD, rho_Pade_FD_corr0,
                gamma="lowTilde")
        builder.R = 0.

        theorical_convergence_factors["LowTildePade"][dt] = rho_Pade_FD_corr0(builder, axis_freq, gamma=low_tilde_gamma)
        observed_convergence_factors["LowTildePade"][dt] = validation(builder,
                N_validation, OceanPadeFD, AtmospherePadeFD, rho_Pade_FD_corr0,
                gamma="lowTilde")


        col = style[dt]['col']
        ls = style[dt]['ls']
        lw = style[dt]['lw']
        lw_observed = 0.45
        legend = style[dt]['legend']

        nb_subpoints = nb_validation_pts[dt]
        w_validation = get_discrete_freq(N_validation, dt)
        indices = find_indices(w_validation[N_validation//2+1:],
                nb_subpoints) + N_validation//2
        axes[0,0].semilogx(w_validation[indices],
                observed_convergence_factors["BEFV"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[0,1].semilogx(w_validation[indices],
                observed_convergence_factors["BEFD"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[0,2].semilogx(w_validation[indices],
                observed_convergence_factors["BEFD_R"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[1,0].semilogx(w_validation[indices],
                observed_convergence_factors["SimplePade"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[1,1].semilogx(w_validation[indices],
                observed_convergence_factors["LowTildePade"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[1,2].semilogx(w_validation[indices],
                observed_convergence_factors["LowTildePade_R"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)

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
    axes[0,0].set_xlim(3.1415/1000,9e2)
    axes[0,1].legend(loc="upper right",prop={'size':7}, handlelength=2)
    fig.tight_layout()   
    show_or_save("fig_interactionsDN")

def robin_parameters_discrete_space(builder, N, dt, scheme="FD"):
    if scheme == "FD":
        rho = rho_c_FD
    elif scheme == "FV":
        rho = rho_c_FV
    else:
        raise NotImplementedError("unknown scheme")

    # axis_freq = get_discrete_freq(N, dt)
    axis_freq = frequencies_for_optim(N, dt, 1000)
    # indices = find_indices(axis_freq[N//2+1:], 100) + N//2
    # axis_freq = axis_freq[indices]
    def to_minimize_onesided(p):
        builder_cp = builder.copy()
        builder_cp.LAMBDA_1, builder_cp.LAMBDA_2 = p, -p
        return np.max(np.abs(rho(builder=builder_cp, w=axis_freq)))

    def to_minimize_twosided(parameters):
        builder_cp = builder.copy()
        builder_cp.LAMBDA_1 = parameters[0]
        builder_cp.LAMBDA_2 = parameters[1]
        return np.max(np.abs(rho(builder=builder_cp, w=axis_freq)))

    # res_onesided = minimize_scalar(fun=to_minimize_onesided)
    # p_opti = res_onesided.x
    res_twosided_eye1 = minimize(fun=to_minimize_twosided,
            x0=np.array((0.35, -0.04)), method='Nelder-Mead')
    res_twosided_eye2 = minimize(fun=to_minimize_twosided,
            x0=np.array((0.06, -0.21)), method='Nelder-Mead')
    if res_twosided_eye2.fun > res_twosided_eye1.fun:
        return res_twosided_eye1
    return res_twosided_eye2

def fig_RRInteraction():
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.D1 = .5
    builder.D2 = 1.
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(8,4.2))
    plt.subplots_adjust(left=.04, bottom=.10, right=.99, top=.92, wspace=0.05, hspace=0.4)
    axes[0,0].set_ylim(top=0.42, bottom=0.) #sharey activated : see ax[1].set_xlim
    dt = builder.DT
    REACTION_COEFF = 0.1 # for cases with reaction != 0
    all_dt = (2e1,2e0,2e-1,2e-2,2e-3,)
    style = {all_dt[0] : {'col':'k', 'ls':'solid', 'lw':1.4, 'legend':r"$\Gamma=10$"},
            all_dt[1] : {'col':'.1', 'ls':'dashed', 'lw':1.55, 'legend':r"$\Gamma=1$"},
            all_dt[2] : {'col':'.5', 'ls':'solid', 'lw':1.7, 'legend':r"$\Gamma=10^{-1}$"},
            all_dt[3] : {'col':'.5', 'ls':'dashed', 'lw':1.9, 'legend':r"$\Gamma=10^{-2}$"},
            all_dt[4] : {'col':'r', 'ls':'solid', 'lw':1., 'legend':r"$\Gamma=\nu_1\frac{\Delta t}{h^2}=10^{-3}$"}}
    nb_validation_pts = {all_dt[0] : 9,
            all_dt[1] : 7,
            all_dt[2] : 6,
            all_dt[3] : 5,
            all_dt[4] : 5 }
        
    theorical_convergence_factors = {}
    theorical_convergence_factors["BEFV"] = {}
    theorical_convergence_factors["BEFD"] = {}
    theorical_convergence_factors["BEFD_R"] = {}
    theorical_convergence_factors["SimplePade"] = {}
    theorical_convergence_factors["LowTildePade"] = {}
    theorical_convergence_factors["LowTildePade_R"] = {}

    observed_convergence_factors = {}
    observed_convergence_factors["BEFV"] = {}
    observed_convergence_factors["BEFD"] = {}
    observed_convergence_factors["BEFD_R"] = {}
    observed_convergence_factors["SimplePade"] = {}
    observed_convergence_factors["LowTildePade"] = {}
    observed_convergence_factors["LowTildePade_R"] = {}

    ############## OPTIMISATION #####################
    dtmin = 2e-3
    builder.R = 0.
    builder.DT = dtmin
    T = 1000
    #optimisation for FD:
    res_twosided = memoised(robin_parameters_discrete_space,
            builder=builder, N=int(T/dtmin), dt=dtmin, scheme="FD")
    theorical_convergence_factors["BEFD"]["robin"] = res_twosided.x
    theorical_convergence_factors["SimplePade"]["robin"] = res_twosided.x
    theorical_convergence_factors["LowTildePade"]["robin"] = res_twosided.x

    #optimisation for FD with Reaction:
    builder.R = REACTION_COEFF
    res_twosided = memoised(robin_parameters_discrete_space,
            builder=builder, N=int(T/dtmin), dt=dtmin, scheme="FD")

    theorical_convergence_factors["BEFD_R"]["robin"] = res_twosided.x
    theorical_convergence_factors["LowTildePade_R"]["robin"] = res_twosided.x

    #optimisation for FV:
    builder.R = 0.
    res_twosided = memoised(robin_parameters_discrete_space,
            builder=builder, N=int(T/dtmin), dt=dtmin, scheme="FV")
    theorical_convergence_factors["BEFV"]["robin"] = res_twosided.x

    def validation(builder, N, oceanclass, atmclass, theory, gamma="simple", **possible_kc):
        ocean, atmosphere = builder.build(oceanclass,
                atmclass, **possible_kc)
        axis_freq = get_discrete_freq(N, builder.DT)
        if REAL_FIG:
            alpha_w = memoised(frequency_simulation, atmosphere, ocean,
                    number_samples=10, NUMBER_IT=1, T=N*builder.DT,
                    gamma=gamma, ignore_cached=False)
            convergence_factor = np.abs((alpha_w[2] / alpha_w[1]))
        else:
            if gamma != "simple":
                convergence_factor = np.abs(theory(builder,
                                axis_freq, gamma=low_tilde_gamma))
            else:
                convergence_factor = np.abs(theory(builder,
                                            axis_freq, **possible_kc))
        return convergence_factor


    def low_tilde_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1) - b/2 * (z-1)**2
    def simple_gamma(z):
        b = 1+1/np.sqrt(2)
        return z - b*(z-1)

    for dt in all_dt:
        col = style[dt]['col']
        ls = style[dt]['ls']
        lw = style[dt]['lw']
        lw_observed = 0.45
        legend = style[dt]['legend']
        builder.DT = dt
        # assert REACTION_COEFF * builder.DT <= 1
        N = int(T/dt)
        N_validation = 10000
        axis_freq = get_discrete_freq(N, builder.DT)
        freq_theory = get_discrete_freq(5*N, builder.DT)
        indices = find_indices(freq_theory[5*N//2+1:], 1000) + 5*N//2
        freq_theory = freq_theory[indices]

        builder.LAMBDA_1, builder.LAMBDA_2 = theorical_convergence_factors["BEFD"]["robin"]
        theorical_convergence_factors["BEFD"][dt] = \
                np.abs(rho_BE_FD(builder=builder, w=freq_theory))
        observed_convergence_factors["BEFD"][dt] = validation(builder,
                N_validation, OceanBEFD, AtmosphereBEFD,
                rho_BE_FD, k_c=0)

        builder.R = REACTION_COEFF
        builder.LAMBDA_1, builder.LAMBDA_2 = \
                    theorical_convergence_factors["BEFD_R"]["robin"]
        theorical_convergence_factors["BEFD_R"][dt] = \
                np.abs(rho_BE_FD(builder=builder, w=freq_theory))
        observed_convergence_factors["BEFD_R"][dt] = validation(builder,
                N_validation, OceanBEFD, AtmosphereBEFD,
                rho_BE_FD, k_c=0)
        builder.R = 0.

        builder.LAMBDA_1, builder.LAMBDA_2 = \
                theorical_convergence_factors["BEFV"]["robin"]
        theorical_convergence_factors["BEFV"][dt] = \
                np.abs(rho_BE_FV(builder=builder, w=freq_theory))
        observed_convergence_factors["BEFV"][dt] = validation(builder,
                N_validation, OceanBEFV, AtmosphereBEFV,
                rho_BE_FV)

        builder.LAMBDA_1, builder.LAMBDA_2 = theorical_convergence_factors["SimplePade"]["robin"]
        theorical_convergence_factors["SimplePade"][dt] = rho_Pade_FD_corr0(builder, freq_theory, gamma=simple_gamma)
        observed_convergence_factors["SimplePade"][dt] = validation(builder,
                N_validation, OceanPadeFD, AtmospherePadeFD,
                rho_Pade_FD_corr0)

        builder.R = REACTION_COEFF
        builder.LAMBDA_1, builder.LAMBDA_2 = theorical_convergence_factors["LowTildePade_R"]["robin"]
        theorical_convergence_factors["LowTildePade_R"][dt] = rho_Pade_FD_corr0(builder, freq_theory, gamma=low_tilde_gamma)
        observed_convergence_factors["LowTildePade_R"][dt] = validation(builder,
                N_validation, OceanPadeFD, AtmospherePadeFD,
                rho_Pade_FD_corr0,
                gamma="lowTilde")
        builder.R = 0.

        builder.LAMBDA_1, builder.LAMBDA_2 = theorical_convergence_factors["LowTildePade"]["robin"]
        theorical_convergence_factors["LowTildePade"][dt] = rho_Pade_FD_corr0(builder, freq_theory, gamma=low_tilde_gamma)
        observed_convergence_factors["LowTildePade"][dt] = validation(builder,
                N_validation, OceanPadeFD, AtmospherePadeFD,
                rho_Pade_FD_corr0,
                gamma="lowTilde")

        nb_subpoints = nb_validation_pts[dt]
        w_validation = get_discrete_freq(N_validation, dt)
        indices = find_indices(w_validation[N_validation//2+1:], nb_subpoints) + N_validation//2
        # indices = indices[nb_subpoints:]
        # indices = np.array(range(w_validation.shape[0]))
        w_validation = w_validation[indices]
        axes[0,0].semilogx(w_validation,
                observed_convergence_factors["BEFV"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[0,1].semilogx(w_validation,
                observed_convergence_factors["BEFD"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[0,2].semilogx(w_validation,
                observed_convergence_factors["BEFD_R"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[1,0].semilogx(w_validation,
                observed_convergence_factors["SimplePade"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[1,1].semilogx(w_validation,
                observed_convergence_factors["LowTildePade"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)
        axes[1,2].semilogx(w_validation,
                observed_convergence_factors["LowTildePade_R"][dt][indices],
                'o', color=col, fillstyle="none", markeredgewidth=lw_observed, zorder=0)

        axes[0,0].semilogx(freq_theory, theorical_convergence_factors["BEFV"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[0,1].semilogx(freq_theory, theorical_convergence_factors["BEFD"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[0,2].semilogx(freq_theory, theorical_convergence_factors["BEFD_R"][dt], linestyle=ls, color=col, label=legend, linewidth=lw)
        axes[1,0].semilogx(freq_theory, theorical_convergence_factors["SimplePade"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[1,1].semilogx(freq_theory, theorical_convergence_factors["LowTildePade"][dt], linestyle=ls, color=col, linewidth=lw)
        axes[1,2].semilogx(freq_theory, theorical_convergence_factors["LowTildePade_R"][dt], linestyle=ls, color=col, linewidth=lw)

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

def wAndRhoPadeRR(builder, gamma=None, N=300):
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

def fig_optiRates():
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    fig, axes = plt.subplots(1, 2, figsize=[6.4*1.4, 4.4], sharex=True, sharey=True)
    axes[0].grid()
    axes[1].grid()
    #axes[1].set_ylim(bottom=0.095, top=0.3) # all axis are shared

    caracs = {}
    caracs["continuous"] = {'color':'#00AF80', 'width':0.7, 'nb_+':9}
    caracs["semi-discrete"] = {'color':'#FF0000', 'width':.9, 'nb_+':15}
    caracs["discrete, FV"] = {'color':'#000000', 'width':.9, 'nb_+':15}
    caracs["discrete, FD"] = {'color':'#0000FF', 'width':.9, 'nb_+':15}


    fig.suptitle("Optimized convergence rates with different methods")
    fig.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.92, wspace=0.13, hspace=0.16)
    #############################################
    # BE
    #######################################

    from ocean_models.ocean_BE_FD import OceanBEFD
    from ocean_models.ocean_BE_FV import OceanBEFV
    from atmosphere_models.atmosphere_BE_FV import AtmosphereBEFV
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD

    all_rates = rho_c_c, rho_BE_c, rho_BE_FV, rho_BE_FD
    all_ocean = OceanBEFV, OceanBEFV, OceanBEFV, OceanBEFD
    all_atmosphere = AtmosphereBEFV, AtmosphereBEFV, AtmosphereBEFV, AtmosphereBEFD
    
    optiRatesGeneral(axes[0], all_rates, all_ocean, all_atmosphere, "BE", caracs=caracs)

    ###########################
    # Pade
    ##########################

    from ocean_models.ocean_Pade_FD import OceanPadeFD
    from ocean_models.ocean_Pade_FV import OceanPadeFV
    from atmosphere_models.atmosphere_Pade_FV import AtmospherePadeFV
    from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD

    all_rates = rho_c_c, rho_Pade_c, rho_Pade_FV, rho_Pade_FD_corr0
    all_ocean = OceanPadeFV, OceanPadeFV, OceanPadeFV, OceanPadeFD
    all_atmosphere = AtmospherePadeFV, AtmospherePadeFV, AtmospherePadeFV, AtmospherePadeFD
    optiRatesGeneral(axes[1], all_rates, all_ocean, all_atmosphere, "P2", caracs=caracs)


    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=caracs["continuous"]["color"], lw=caracs["continuous"]["width"]),
                    Line2D([0], [0], color=caracs["semi-discrete"]["color"], lw=caracs["semi-discrete"]["width"]),
                    Line2D([0], [0], color=caracs["discrete, FV"]["color"], lw=caracs["discrete, FV"]["width"]),
                    Line2D([0], [0], color=caracs["discrete, FD"]["color"], lw=caracs["discrete, FD"]["width"]),
                    Line2D([0], [0], marker="^", markersize=6., linewidth=0.,
                        color="000000") ]
    custom_labels = ["Continuous", "Semi-discrete", "Discrete, FV", "Discrete, FD", "Theoretical prediction"]
    fig.legend(custom_lines, custom_labels, loc=(0.1, 0.), ncol=5, handlelength=2)
    show_or_save("fig_optiRates")

def optiRatesGeneral(axes, all_rates, all_ocean, all_atmosphere,
        name_method="Unknown discretization", caracs={}, **args_for_discretization):
    """
        Creates a figure comparing analysis methods for a discretization.
    """

    setting = Builder()
    setting.D1 = .5
    setting.R = 1e-3
    setting.DT = 1.
    N = 1000000
    axis_freq = get_discrete_freq(N, setting.DT)
    freq_for_optim = frequencies_for_optim(N, setting.DT, 200)

    axes.set_xlabel("$\\omega \\Delta t$")
    axes.set_ylabel(r"${\rho}_{RR}^{"+name_method+r"}$")

    def rate_onesided(lam):
        builder = setting.copy()
        builder.LAMBDA_1 = lam
        builder.LAMBDA_2 = -lam
        return np.max(np.abs(all_rates[0](builder, freq_for_optim)))

    from scipy.optimize import minimize_scalar, minimize
    optimal_lam = minimize_scalar(fun=rate_onesided)
    x0_opti = (optimal_lam.x, -optimal_lam.x)

    for discrete_factor, oce_class, atm_class, names in zip(all_rates,
            all_ocean, all_atmosphere, caracs):
        def rate_twosided(lam):
            builder = setting.copy()
            builder.LAMBDA_1 = lam[0]
            builder.LAMBDA_2 = lam[1]
            return np.max(np.abs(discrete_factor(builder, freq_for_optim)))

        optimal_lam = minimize(method='Nelder-Mead',
                fun=rate_twosided, x0=x0_opti)
        optimal_lam_new = minimize(method='Nelder-Mead',
                fun=rate_twosided, x0=(0.4, -0.05))
        if optimal_lam.fun > optimal_lam_new.fun:
            optimal_lam = optimal_lam_new
        if names == "continuous":
            x0_opti = optimal_lam.x
        setting.LAMBDA_1 = optimal_lam.x[0]
        setting.LAMBDA_2 = optimal_lam.x[1]

        builder = setting.copy()
        ocean, atmosphere = builder.build(oce_class, atm_class)
        if REAL_FIG:
            alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=10, NUMBER_IT=1, laplace_real_part=0, T=N*builder.DT)
            convergence_factor = np.abs((alpha_w[2] / alpha_w[1]))
        else:
            convergence_factor = np.abs(ocean.discrete_rate(setting, axis_freq))


        axis_freq_predicted = np.exp(np.linspace(np.log(min(np.abs(axis_freq))), np.log(axis_freq[-1]), caracs[names]["nb_+"]))

        # LESS IMPORTANT CURVE : WHAT IS PREDICTED

        axes.semilogx(axis_freq * setting.DT, convergence_factor, linewidth=caracs[names]["width"], label= "$p_1, p_2 =$ ("+ str(optimal_lam.x[0])[:4] +", "+ str(optimal_lam.x[1])[:5] + ")", color=caracs[names]["color"]+"90")
        if names =="discrete":
            axes.semilogx(axis_freq_predicted * setting.DT, np.abs(discrete_factor(setting, axis_freq_predicted)), marker="^", markersize=6., linewidth=0., color=caracs[names]["color"])# , label="prediction")
        else:
            axes.semilogx(axis_freq_predicted * setting.DT, np.abs(discrete_factor(setting, axis_freq_predicted)), marker="^", markersize=6., linewidth=0., color=caracs[names]["color"])

    axes.legend( loc=(0., 0.), ncol=1 )
    #axes.set_xlim(left=1e-3, right=3.4)
    axes.set_ylim(bottom=0)

def fig_optiRatesL2():
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

    caracs = {}
    caracs["continuous"] = {'color':'#00AF80', 'width':0.7, 'nb_+':9}
    caracs["semi-discrete"] = {'color':'#FF0000', 'width':.9, 'nb_+':15}
    caracs["discrete, FV"] = {'color':'#000000', 'width':.9, 'nb_+':15}
    caracs["discrete, FD"] = {'color':'#0000FF', 'width':.9, 'nb_+':15}

    #############################################
    # BE
    #######################################

    from ocean_models.ocean_BE_FD import OceanBEFD
    from ocean_models.ocean_BE_FV import OceanBEFV
    from atmosphere_models.atmosphere_BE_FV import AtmosphereBEFV
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD

    all_rates = rho_c_c, rho_BE_c, rho_BE_FV, rho_BE_FD
    all_ocean = OceanBEFV, OceanBEFV, OceanBEFV, OceanBEFD
    all_atmosphere = AtmosphereBEFV, AtmosphereBEFV, AtmosphereBEFV, AtmosphereBEFD
    p = multiprocessing.Process(target=optiRatesGeneralL2,
            args=(all_rates, all_ocean, all_atmosphere,
            "BE", caracs))
    p.start()

    ###########################
    # Pade
    ##########################

    from ocean_models.ocean_Pade_FD import OceanPadeFD
    from ocean_models.ocean_Pade_FV import OceanPadeFV
    from atmosphere_models.atmosphere_Pade_FV import AtmospherePadeFV
    from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD

    all_rates = rho_c_c, rho_Pade_c, rho_Pade_FV, rho_Pade_FD_corr0
    all_ocean = OceanPadeFV, OceanPadeFV, OceanPadeFV, OceanPadeFD
    all_atmosphere = AtmospherePadeFV, AtmospherePadeFV, AtmospherePadeFV, AtmospherePadeFD
    optiRatesGeneralL2(all_rates, all_ocean, all_atmosphere, "P2", caracs=caracs)

def optiRatesGeneralL2(all_rates, all_ocean, all_atmosphere,
        name_method="Unknown discretization", caracs={}, **args_for_discretization):
    """
        Creates a figure comparing analysis methods for a discretization.
    """

    setting = Builder()
    setting.D1 = .5
    setting.R = 1e-3
    setting.DT = 1.
    N = 1000000
    axis_freq = get_discrete_freq(N, setting.DT)
    freq_for_optim = frequencies_for_optim(N, setting.DT, 200)

    def rate_onesided(lam):
        builder = setting.copy()
        builder.LAMBDA_1 = lam
        builder.LAMBDA_2 = -lam
        return np.max(np.abs(all_rates[0](builder, freq_for_optim)))

    from scipy.optimize import minimize_scalar, minimize
    optimal_lam = minimize_scalar(fun=rate_onesided)
    x0_opti = (optimal_lam.x, -optimal_lam.x)

    for discrete_factor, oce_class, atm_class, names in zip(all_rates,
            all_ocean, all_atmosphere, caracs):
        def rate_twosided(lam):
            builder = setting.copy()
            builder.LAMBDA_1 = lam[0]
            builder.LAMBDA_2 = lam[1]
            return np.max(np.abs(discrete_factor(builder, freq_for_optim)))

        optimal_lam = minimize(method='Nelder-Mead',
                fun=rate_twosided, x0=x0_opti)
        optimal_lam_new = minimize(method='Nelder-Mead',
                fun=rate_twosided, x0=(0.4, -0.05))
        if optimal_lam.fun > optimal_lam_new.fun:
            optimal_lam = optimal_lam_new
        if names == "continuous":
            x0_opti = optimal_lam.x
        setting.LAMBDA_1 = optimal_lam.x[0]
        setting.LAMBDA_2 = optimal_lam.x[1]

        def print_cv_rate(builder, oce_class, atm_class, names):
            ocean, atmosphere = builder.build(oce_class, atm_class)

            L2_norm = memoised(simulation_L2norm, atmosphere, ocean,
                    number_samples=1, NUMBER_IT=1, laplace_real_part=0,
                    T=N*builder.DT)
            convergence_rate = L2_norm[2]/L2_norm[1]
            print(name_method, names, convergence_rate)

        p = multiprocessing.Process(target=print_cv_rate,
                args=(setting.copy(), oce_class, atm_class, names))
        p.start()

######################################################
# Utilities for analysing, representing discretizations
######################################################

class Builder():
    """
        interface between the discretization classes and the plotting functions.
        The main functions is build: given a space and a time discretizations,
        it returns a class which can be used with all the available functions.

        To use this class, instanciate builder = Builder(),
        choose appropriate arguments of builder:
        builder.DT = 0.1
        builder.LAMBDA_2 = -0.3
        and then build all the schemes you want with the models
        contained in ocean_models and atmosphere_models.
        The comparison is thus then quite easy
    """
    def __init__(self): # changing defaults will result in needing to recompute all cache
        self.R = 0.
        self.D1=1.
        self.D2=1.
        self.M1=100
        self.M2=100
        self.LAMBDA_1=1e9
        self.LAMBDA_2=0.
        self.SIZE_DOMAIN_1=100
        self.SIZE_DOMAIN_2=100
        COURANT_NUMBER = 1.
        self.DT = COURANT_NUMBER * (self.SIZE_DOMAIN_1 / self.M1)**2 / self.D1

    def set_h(self, h, vertical_levels=100):
        self.SIZE_DOMAIN_1 = self.SIZE_DOMAIN_2 = \
                float(h*(vertical_levels-1))
        self.M1 = vertical_levels
        self.M2 = vertical_levels

    def copy(self):
        ret = Builder()
        ret.__dict__ = self.__dict__.copy()
        return ret

    def build(self, ocean_discretisation, atm_discretisation, **kwargs):
        """ build the models and returns tuple (ocean_model, atmosphere_model)"""
        ocean = ocean_discretisation(r=self.R, nu=self.D1,
                LAMBDA=self.LAMBDA_1, M=self.M1,
                SIZE_DOMAIN=self.SIZE_DOMAIN_1, DT=self.DT, **kwargs)
        atmosphere = atm_discretisation(r=self.R, nu=self.D2,
                LAMBDA=self.LAMBDA_2, M=self.M2,
                SIZE_DOMAIN=self.SIZE_DOMAIN_2, DT=self.DT, **kwargs)
        return ocean, atmosphere


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
    import os
    name_fig = name_func[4:]
    directory = "figures_out/"
    if SAVE_TO_PNG:
        print("exporting to directory " + directory)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.png')
    elif SAVE_TO_PGF:
        print("exporting to directory " + directory)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.pgf')
    elif SAVE_TO_PDF:
        print("exporting to directory " + directory)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.pdf')
    else:
        try:
            os.makedirs(directory, exist_ok=True)
            mpl.rcParams['savefig.directory'] = directory
            fig = plt.get_current_fig_manager()
            fig.set_window_title(name_fig)
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
