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
from simulator import frequency_simulation


def fig_firstBulkAnalysis():
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.space.FD_corr import FiniteDifferencesCorr
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.Manfredi import Manfredi
    from cv_factor_pade import rho_Pade_FD_corr0
    from cv_factor_pade import rho_Pade_FD_extra
    from cv_factor_pade import rho_Pade_c
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.LAMBDA_1 = 0.
    builder.LAMBDA_2 = 0.
    builder.M1 = 100
    builder.M2 = 100
    builder.D1 = 1.
    builder.D2 = 1.
    builder.R = 0.
    N = 300
    dt = builder.DT
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)

    axis_freq = get_discrete_freq(N, dt)
    fig, axes = plt.subplots(1, 1, figsize=[6.4, 4.8])

    from discretizations.BE_FD_bulk import be_fd_bulk
    theta=0.
    ratio_density=1.
    alpha=1.
    D2=builder.D2
    M1 = builder.M1
    M2 = builder.M2
    for theta in (0., 1.):
        for M1 in (1000, 10000):
        #for M1 in (10, 100,1000):
            dis =  be_fd_bulk(alpha=alpha, theta=theta, ratio_density=ratio_density,
                    A=builder.A, C=builder.R,
                                      D1=builder.D1, D2=D2,
                                      M1=M1, M2=M2,
                                      SIZE_DOMAIN_1=builder.SIZE_DOMAIN_1,
                                      SIZE_DOMAIN_2=builder.SIZE_DOMAIN_2,
                                      LAMBDA_1=builder.LAMBDA_1,
                                      LAMBDA_2=builder.LAMBDA_2,
                                      DT=builder.DT)

            alpha_w = memoised(frequency_simulation, dis, N=N, number_samples=10, NUMBER_IT=2, ignore_cached=False)

            convergence_factor = np.abs(alpha_w[2] / alpha_w[1])
            axes.loglog(axis_freq * dt, convergence_factor, "--")
                
            axes.loglog(axis_freq * dt, np.abs(dis.convergence_rate(axis_freq)), label="convergence rate, $\\alpha$="+str(alpha)+", $\\theta$="+str(theta)+", $\\rho_2/\\rho_1$="+ str(ratio_density)+ ", $h_2/h_1=$"+str(M1/M2))

            #axes.loglog(axis_freq * dt, convergence_factor, "--", label="convergence rate, $\\alpha=C_D||u_2-u_1||$, $\\theta$="+str(theta)+", $\\rho_2/\\rho_1$="+ str(ratio_density)+ ", $h_2/h_1=$"+str(M1/M2))

    #axes.loglog(axis_freq * dt, np.ones_like(convergence_factor), "m--", label="limit of convergence")

    axes.set_xlabel("Frequency variable $\\omega \\delta t$")
    axes.set_ylabel("Convergence factor $\\rho$")
    axes.set_title("Convergence rate with flux prescription $\\phi=\\alpha(\\Delta u)$")
    axes.legend()
    show_or_save("fig_firstBulkAnalysis")


def fig_validatePadeAnalysisFDRR():
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.space.FD_corr import FiniteDifferencesCorr
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.Manfredi import Manfredi
    from cv_factor_pade import rho_Pade_FD_corr0
    from cv_factor_pade import rho_Pade_FD_extra
    from cv_factor_pade import rho_Pade_c
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.LAMBDA_1 = 1.11 # optimal parameters for corr=0, N=3000
    builder.LAMBDA_2 = -0.76
    builder.M1 = 200
    builder.M2 = 200
    builder.D1 = 1.
    builder.D2 = 2.
    builder.R = 0.5
    N = 300
    dt = builder.DT
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)
    print("Courant parabolic number :", builder.D1*dt/h**2)

    time_scheme = Manfredi
        
    discretizations = {}

    #discretizations["FV2"] = (time_scheme, QuadSplinesFV)
    #discretizations["FV4"] = (time_scheme, FourthOrderFV)
    discretizations["FD(corr=0)"] = (time_scheme, FiniteDifferencesNaive)
    # discretizations["FD(extra)"] = (time_scheme, FiniteDifferencesExtra)

    axis_freq = get_discrete_freq(N, dt)
    fig, axes = plt.subplots(1, 1, figsize=[6.4, 4.8])

    dis_cont = builder.build(time_scheme, FiniteDifferencesNaive) # any discretisation would do 
    continuous = dis_cont.analytic_robin_robin_modified(w=axis_freq,
                    order_time=0, order_operators=0,
                    order_equations=0)
    axes.semilogx(axis_freq * dt, continuous, label="$\\rho^{\\rm c, c}$")

    for name in discretizations:
        time_dis, space_dis = discretizations[name]
        dis = builder.build(time_dis, space_dis)
        theorical_convergence_factor = \
                dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=0, order_operators=float('inf'),
                        order_equations=float('inf'))
        axes.semilogx(axis_freq * dt, theorical_convergence_factor,
                label="$\\rho^{\\rm c, "+name + "}$")

        theorical_convergence_factor = \
                dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=4, order_operators=float('inf'),
                        order_equations=float('inf'))
        axes.semilogx(axis_freq * dt, theorical_convergence_factor,
                label="$\\rho^{\\rm modified, "+name + "}$")


    axes.semilogx(axis_freq * dt, rho_Pade_FD_corr0(builder, axis_freq),
            label="$\\rho^{\\rm Pade, FD(corr=0)}$")
    axes.semilogx(axis_freq * dt, rho_Pade_FD_extra(builder, axis_freq),
            label="$\\rho^{\\rm Pade, FD(extra)}$")
    axes.semilogx(axis_freq * dt, rho_Pade_c(builder, axis_freq),
            label="$\\rho^{\\rm Pade, c}$")

    ###########
    # for each discretization, a simulation
    ###########
    for name in discretizations:
        time_dis, space_dis = discretizations[name]
        alpha_w = memoised(Builder.frequency_cv_factor, builder,
                time_dis, space_dis, N=N, number_samples=5, NUMBER_IT=4)
        k = 1
        convergence_factor = np.abs(alpha_w[k+1] / alpha_w[k])
        axes.semilogx(axis_freq * dt, convergence_factor, "--", label=name)
        # k = 2
        # convergence_factor = np.abs(alpha_w[k+1] / alpha_w[k])
        # axes.semilogx(axis_freq * dt, convergence_factor, "--", label=name)
        # k = 3
        # convergence_factor = np.abs(alpha_w[k+1] / alpha_w[k])
        # axes.semilogx(axis_freq * dt, convergence_factor, "--", label=name)

    axes.set_xlabel("Frequency variable $\\omega \\delta t$")
    axes.set_ylabel("Convergence factor $\\rho$")
    axes.set_title("Validation of finite differences discrete analysis")
    axes.legend()
    show_or_save("fig_validatePadeAnalysisFDRR")

def fig_optimized_rho():
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra
    discrete_factor = rho_Pade_c
    setting = Builder()
    setting.M1 = 200
    setting.M2 = 200
    setting.D1 = 1.
    setting.D2 = 2.
    setting.R = 0.5
    setting.DT /= 100
    N = 3000
    axis_freq = get_discrete_freq(N, setting.DT)
    def convergence_factor(lam):
        builder = setting.copy()
        builder.LAMBDA_1 = lam[0]
        builder.LAMBDA_2 = lam[1]
        return np.max(discrete_factor(builder, axis_freq))

    def convergence_factor(lam):
        builder = setting.copy()
        builder.LAMBDA_1 = lam
        builder.LAMBDA_2 = -lam
        return np.max(discrete_factor(builder, axis_freq))

    from scipy.optimize import minimize_scalar, minimize
    optimal_lam = minimize_scalar(fun=convergence_factor)
    print(optimal_lam)
    setting.LAMBDA_1 = optimal_lam.x
    setting.LAMBDA_2 = -optimal_lam.x
    plt.semilogx(axis_freq * setting.DT, discrete_factor(setting, axis_freq),
            label="$\\rho^{\\rm Pade, FD(corr=0)}$")

    show_or_save("fig_optimized_rho")

def fig_compare_discrete_modif():
    from discretizations.time.Manfredi import Manfredi as Pade
    from discretizations.time.backward_euler import BackwardEuler as BE
    from discretizations.space.FD_naive import FiniteDifferencesNaive as FD
    from discretizations.space.FD_extra import FiniteDifferencesExtra as FD
    from discretizations.space.quad_splines_fv import QuadSplinesFV as FV
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra
    fig, axes = plt.subplots(2, 2, figsize=[6.4, 4.4], sharex=False, sharey=True)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.92, wspace=0.1, hspace=0.15)
    COLOR_CONT = '#888888FF'
    COLOR_CONT_FD = '#555555FF'
    COLOR_MODIF = '#000000FF'

    for r, axes in ((0, axes[0,:]), (.1, axes[1,:])):
        setting = Builder()
        setting.R = r

        setting.LAMBDA_1 = 1. # optimal parameters for corr=0, N=3000
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

        full_discrete = rho_Pade_c(setting, w=axis_freq, gamma=gamma_order2) # disccrete in time
        labelg2 = r"P2: $\left|\rho_{\rm RR}^{\rm (\cdot,c)} - \rho_{\rm RR}^{\rm (P2,c)}\right|/\left|\rho_{\rm RR}^{\rm (P2,c)}\right|$" + "\n" + r"$\gamma = z - b (z-1) - \frac{b^2}{2}(z-1)^2$"
        lineg2, = axes[0].semilogx(axis_freq*dt, np.abs(full_discrete - modif_time)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_MODIF, linestyle='solid')
        axes[0].semilogx(axis_freq*dt, np.abs(full_discrete - cont_time)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_CONT, linestyle='solid')

        ######################
        # TIME SCHEME : GAMMA ORDER 1:
        ######################

        full_discrete = rho_Pade_c(setting, w=axis_freq, gamma=gamma_order1) # disccrete in time

        labelg1 = r"P2: $\left|\rho_{\rm RR}^{\rm (\cdot,c)} - \rho_{\rm RR}^{\rm (P2,c)}\right|/\left|\rho_{\rm RR}^{\rm (P2,c)}\right|$" + "\n" + r"$\gamma = z - b (z-1)$"
        lineg1, = axes[0].semilogx(axis_freq*dt, np.abs(full_discrete - modif_time)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_MODIF, linestyle='dashed')
        axes[0].semilogx(axis_freq*dt, np.abs(full_discrete - cont_time)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_CONT, linestyle='dashed')

        ########################
        # TIME SCHEME : Backward Euler
        #########################
        dis = setting.build(BE, space_dis)

        modif_time = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=2, order_equations=0, order_operators=0) # modified in time
        full_discrete = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=float('inf'), order_equations=0, order_operators=0) # discrete in time

        labelbe = r"BE: $\left|\rho_{\rm RR}^{\rm (\cdot,c)} - \rho_{\rm RR}^{\rm (BE,c)}\right|/\left|\rho_{\rm RR}^{\rm (BE,c)}\right|$"
        linebe, = axes[0].semilogx(axis_freq*dt, np.abs(full_discrete - modif_time)/np.abs(full_discrete),
                color=COLOR_MODIF, linestyle=':', linewidth="2.3")
        axes[0].semilogx(axis_freq*dt, np.abs(full_discrete - cont_time)/np.abs(full_discrete),
                color=COLOR_CONT, linestyle=':', linewidth="2.3")

        axes[0].grid()
        axes[0].set_xlim(left=0.9e-2, right=.7)
        #axes[0].set_ylim(top=0.1, bottom=0.) #sharey activated : see axes[1].set_xlim
        Title = r'Relative error of $\rho_{\rm RR}^{\rm (\cdot,c)}$'
        #x_legend= r'$\left| \rho_{\rm RR}^{\rm (\cdot,c)} - \rho_{\rm RR}^{\rm (Discrete,c)}\right|/\left|\rho_{\rm RR}^{\rm (Discrete,c)}\right| $'
        axes[0].set_ylabel(r'$r=' + str(r) + r'\;{\rm s}^{-1}$')
        if r == 0:
            axes[0].legend((lineg2, ), (labelg2, ))
            axes[0].set_title(Title)
            #print(axes[0].ticks)
            axes[0].set_xticklabels([])
        else:
            axes[0].legend((lineg1, linebe), (labelg1, labelbe))
            axes[0].set_xlabel(r'$\omega\Delta t$')

        #########################################################
        # RIGHT CANVA: SPACE COMPARISON
        #########################################################
        time_dis = BE # we don't really care, since everything is continuous in time now

        ######################
        # SPACE SCHEME : FV
        ######################
        dis = setting.build(time_dis, FV)

        cont_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=0, order_operators=0) #continuous in time

        modif_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=2, order_operators=0) # modified in time

        full_discrete = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=float('inf'), order_operators=0)


        axes[1].semilogx(axis_freq*dt, np.abs(full_discrete - modif_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_MODIF, linestyle='solid',
                label=r"FV: $\left|\rho_{\rm RR}^{\rm (c, \cdot)} - \rho_{\rm RR}^{\rm (c,FV)}\right|/\left|\rho_{\rm RR}^{\rm (c,FV)}\right|$")
        axes[1].semilogx(axis_freq*dt, np.abs(full_discrete - cont_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_CONT, linestyle='solid')

        ######################
        # SPACE SCHEME : FD
        ######################
        dis = setting.build(time_dis, FD)

        cont_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=0, order_operators=0) #continuous in time

        modif_space = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=2, order_operators=0) # modified in time

        full_discrete = dis.analytic_robin_robin_modified(w=axis_freq,
                order_time=0, order_equations=float('inf'), order_operators=0)

        axes[1].semilogx(axis_freq*dt, np.abs(full_discrete - modif_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_MODIF, linestyle='dashed',
                label=r"FD: $\left|\rho_{\rm RR}^{\rm (c, \cdot)} - \rho_{\rm RR}^{\rm (c,FD)}\right|/\left|\rho_{\rm RR}^{\rm (c,FD)}\right|$")
        axes[1].semilogx(axis_freq*dt, np.abs(full_discrete - cont_space)/np.abs(full_discrete), linewidth='2.',
                color=COLOR_CONT_FD, linestyle='dashed')

        axes[1].grid()
        axes[1].set_xlim(left=2e-2, right=3)
        axes[1].set_ylim(top=0.03, bottom=0.)
        Title = r'Relative error of $\rho_{\rm RR}^{\rm (c, \cdot)}$'
        #x_legend= r'$\left| \rho_{\rm RR}^{\rm (c, \cdot)} - \rho_{\rm RR}^{\rm (c, Discrete)}\right|/\left|\rho_{\rm RR}^{\rm (c, Discrete)}\right| $'
        if r == 0:
            axes[1].legend()
            axes[1].set_title(Title)
            axes[1].set_xticklabels([])
        else:
            axes[1].set_xlabel(r'$\omega\Delta t$')

    show_or_save("fig_compare_discrete_modif")

def fig_optimized_rho_BE_FV():
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.RK2 import RK2
    from discretizations.space.fourth_order_fv import FourthOrderFV
    time_dis = BackwardEuler
    space_dis = FourthOrderFV
    setting = Builder()
    setting.R = .0
    #setting.DT /= 10
    N = 3000
    axis_freq = get_discrete_freq(N, setting.DT)
    def convergence_factor(lam):
        builder = setting.copy()
        builder.LAMBDA_1 = lam[0]
        builder.LAMBDA_2 = lam[1]
        dis = builder.build(time_dis, space_dis)
        return dis.analytic_robin_robin_modified(w=axis_freq,
            order_time=float('inf'), order_equations=float('inf'), order_operators=float('inf'))

    def to_minimize(lam):
        return np.max(np.abs(convergence_factor(lam)))

    from scipy.optimize import minimize_scalar, minimize
    optimal_lam = minimize(fun=to_minimize, x0=(.5, -.5))
    print(optimal_lam)
    plt.semilogx(axis_freq * setting.DT, convergence_factor(optimal_lam.x),
            label="$\\rho^{\\rm Pade, FD(corr=0)}$")

    show_or_save("fig_optimized_rho_BE_FV")

def fig_impact_DT_pade_DN():
    from cv_factor_pade import rho_Pade_FD_corr0, rho_Pade_c, rho_Pade_FD_extra
    setting = Builder()
    setting.R = .0
    setting.D2 = 2.
    N = 300
    axis_freq = get_discrete_freq(N, setting.DT)

    plt.semilogx(axis_freq * setting.DT, rho_Pade_c(setting, axis_freq),
            label="$\\rho^{\\rm Pade, FD(corr=0)}$, dt1")
    setting.R += .1
    axis_freq = get_discrete_freq(N, setting.DT)
    plt.semilogx(axis_freq * setting.DT, rho_Pade_c(setting, axis_freq),
            "--", label="$\\rho^{\\rm Pade, FD(corr=0)}$, dt2")
    setting.R += 1.
    axis_freq = get_discrete_freq(N, setting.DT)
    plt.semilogx(axis_freq * setting.DT, rho_Pade_c(setting, axis_freq),
            "k-.", label="$\\rho^{\\rm Pade, FD(corr=0)}$, dt3")

    show_or_save("fig_optimized_rho")


def fig_compareSettingsDirichletNeumann():
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.space.FD_corr import FiniteDifferencesCorr
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.Manfredi import Manfredi
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.LAMBDA_1 = 1e9  # extremely high lambda is a Dirichlet condition
    builder.LAMBDA_2 = 0. # lambda=0 is a Neumann condition
    builder.D1 = 1.
    builder.D2 = 2.
    builder.R = 0.4
    dt = builder.DT
    assert builder.R * builder.DT < 1
        


    discretizations = {}
    time_scheme = Manfredi

    discretizations["FV2"] = (time_scheme, QuadSplinesFV)
    discretizations["FV4"] = (time_scheme, FourthOrderFV)
    discretizations["FD, extra"] = (time_scheme, FiniteDifferencesExtra)
    discretizations["FD, corr=0"] = (time_scheme, FiniteDifferencesNaive)
    #discretizations["FD, corr=1"] = (time_scheme, FiniteDifferencesCorr)

    convergence_factors = {}
    theorical_convergence_factors = {}

    N = 300
    axis_freq = get_discrete_freq(N, builder.DT)

    kwargs_label_simu = {'label':"Validation by simulation"}
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    ###########
    # for each discretization, a simulation
    ###########
    for name in discretizations:
        time_dis, space_dis = discretizations[name]
        alpha_w = memoised(Builder.frequency_cv_factor, builder, time_dis, space_dis, N=N, number_samples=5)
        k = 1
        convergence_factors[name] = alpha_w[k+1] / alpha_w[k]

        dis = builder.build(time_dis, space_dis)
        theorical_convergence_factors[name] = \
                dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=0, order_operators=float('inf'),
                        order_equations=float('inf'))
        # continuous = dis.analytic_robin_robin_modified(w=axis_freq,
        #                 order_time=0, order_operators=float('inf'),
        #                 order_equations=float('inf'))
        # plt.plot(axis_freq * dt, continuous, "--", label="Continuous Theorical " + name)
        #axes[0].semilogx(axis_freq * dt, convergence_factors[name], "k--", **kwargs_label_simu)
        axes[0].semilogx(axis_freq * dt, convergence_factors[name], label=name)
        if kwargs_label_simu: # We only want the legend to be present once
            kwargs_label_simu = {}
        #axes[0].semilogx(axis_freq * dt, theorical_convergence_factors[name], label=name+ " theorical")
    w, rho_theoric = wAndRhoPadeRR(builder)
    axes[0].semilogx(w*builder.DT, rho_theoric, "k--", label="theoric")

    axes[0].set_xlabel("Frequency variable $\\omega \\delta t$")
    axes[0].set_ylabel("Convergence factor $\\rho$")
    axes[0].set_title("Various space discretizations with " + time_scheme.__name__)

    axes[1].set_xlabel("Frequency variable $\\omega \\delta t$")
    axes[1].set_ylabel("Convergence factor $\\rho$")
    axes[1].set_title("Various time discretizations with Finite Differences, Corr=0")

    space_scheme = FiniteDifferencesNaive
    discretizations = {}

    discretizations["BackwardEuler"] = (BackwardEuler, space_scheme)
    discretizations["ThetaMethod"] = (ThetaMethod, space_scheme)
    # discretizations["RK2"] = (RK2, space_scheme)
    # discretizations["RK4"] = (RK4, space_scheme)
    discretizations["Manfredi"] = (Manfredi, space_scheme)

    kwargs_label_simu = {'label':"Validation by simulation"}

    for name in discretizations:
        time_dis, space_dis = discretizations[name]
        alpha_w = memoised(Builder.frequency_cv_factor, builder, time_dis, space_dis, N=N, number_samples=5)
        k = 1
        convergence_factors[name] = alpha_w[k+1] / alpha_w[k]

        dis = builder.build(time_dis, space_dis)
        theorical_convergence_factors[name] = \
                dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=0, order_operators=float('inf'),
                        order_equations=float('inf'))
        # continuous = dis.analytic_robin_robin_modified(w=axis_freq,
        #                 order_time=0, order_operators=float('inf'),
        #                 order_equations=float('inf'))
        # plt.plot(axis_freq * dt, continuous, "--", label="Continuous Theorical " + name)
        axes[1].semilogx(axis_freq * dt, convergence_factors[name], label=name)

    axes[0].legend()
    axes[1].legend()
    show_or_save("fig_compareSettingsDirichletNeumann")

def fig_rootsManfrediFD():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=[9.6, 2.])
    plt.subplots_adjust(left=.07, bottom=.28, right=.97, top=.85)
    builder = Builder()

    ###########################
    # equation: (\Gamma_{a,j} = a*dt*nu/h^2, \Gamma_{b,j} = b*dt*nu/h^2)
    #        (z-1+r\Delta t + z r^2 \Delta t b)\lambda_i^2 + 
    #        \left(\Gamma_a - 2z\Gamma_b(1+r\Delta t b)\right) \lambda \left(\lambda-1\right)^2 + 
    #        2\Gamma_b^2 \left(\lambda-1\right)^4 = 0
    # rewrite it for wolframAlpha: f* x^2 + g*x(x-1)^2 + (x-1)^4 = 0
    # where x = \lambda
    # where f = (z-1+r\Delta t + z*r^2 \Delta t b) / (2\Gamma_b^2)
    # and g = (\Gamma_a - 2z\Gamma_b) / (2\Gamma_b^2)
    ##########################"
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.R
    nu_1 = builder.D1
    nu_2 = builder.D2
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)

    def get_z(w):
        return np.exp(-1j*w*dt)

    def Gamma(ab, nu):
        return ab*dt*nu/h**2

    def get_f_g(w, nu):
        z = get_z(w)
        Gamma_a, Gamma_b = Gamma(a, nu), Gamma(b, nu)
        return (z - 1 + r*dt + z*r**2*dt*b) / (2*Gamma_b**2), \
                (Gamma_a - 2*z*Gamma_b*(1 + r*dt*b)) / (2*Gamma_b**2)

    def square_root_interior(f, g):
        return np.sqrt(-(4*(g-4)*(f-2*g+6) - (g-4)**3 - 8*(g-4))/(2*np.sqrt(g**2 - 4*f)) \
                - f + (g-4)**2/2 + 2*g - 8)/2

    def lambda_pp(w, nu):
        f, g = get_f_g(w, nu)
        return 1 - g/4 + 1j*np.sqrt(4*f - g**2)/4 + square_root_interior(f, g)

    def lambda_pm(w, nu):
        f, g = get_f_g(w, nu)
        return 1 - g/4 + 1j* np.sqrt(4*f - g**2)/4 - square_root_interior(f, g)

    def lambda_mp(w, nu):
        f, g = get_f_g(w, nu)
        return 1 - g/4 - 1j* (np.sqrt(4*f - g**2)/4 + square_root_interior(f, g))

    def lambda_mm(w, nu):
        f, g = get_f_g(w, nu)
        return 1 - g/4 - 1j* (np.sqrt(4*f - g**2)/4 - square_root_interior(f, g))

    N = 30000
    w = get_discrete_freq(N, dt)

    sigma_1 = np.log(lambda_pm(w, nu_1)) / h
    sigma_2 = np.log(lambda_mp(w, nu_1)) / h
    sigma_3 = np.log(lambda_pp(w, nu_1)) / h
    sigma_4 = np.log(lambda_mm(w, nu_1)) / h

    axes[0].semilogx(w, np.real(sigma_1), label="$\\sigma_1$")

    axes[0].semilogx(w, np.real(sigma_2), label="$\\sigma_2$")
    axes[0].semilogx(w, np.real(sigma_3), label="$\\sigma_3$")
    axes[0].semilogx(w, np.real(sigma_4), label="$\\sigma_4$")

    axes[0].semilogx(w, np.abs(np.real(np.sqrt((r+1j*w)/nu_1))), "k--", label="$\\sigma_j$ continuous")
    axes[0].semilogx(w, np.abs(np.real(-np.sqrt((r+1j*w)/nu_1))), "k--")
    axes[0].set_xlabel("$\\Delta t\\omega$")
    axes[0].set_ylabel("$\\mathfrak{R}(\\sigma)$")
    axes[0].set_title("Real part $\\mathfrak{R}(\\sigma)$")
    axes[0].grid()

    axes[1].semilogx(w, np.imag(sigma_1), label="$\\sigma_1$")

    axes[1].semilogx(w, np.imag(sigma_2), label="$\\sigma_2$")
    axes[1].semilogx(w, np.imag(sigma_3), label="$\\sigma_3$")
    axes[1].semilogx(w, np.imag(sigma_4), label="$\\sigma_4$")

    axes[1].semilogx(w, np.imag(np.sqrt((r+1j*w)/nu_1)), "k--", label="$\\sigma_j$ continuous")
    axes[1].semilogx(w, np.imag(-np.sqrt((r+1j*w)/nu_1)), "k--")
    axes[1].set_xlabel("$\\Delta t\\omega$")
    axes[1].set_ylabel("$Im(\\sigma)$")
    axes[1].set_title("Imaginary part $Im(\\sigma)$")
    axes[1].grid()

    plt.legend()
    show_or_save("fig_rootsManfrediFD")

def wAndRhoPadeRR(builder):
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

def fig_rhoDNPade():
    import matplotlib.pyplot as plt
    w, varrho = wAndRhoPadeRR()
    plt.semilogx(w*DEFAULT.DT, np.abs(varrho), label="$\\rho_{DN}^{Pade, c}$")
    plt.title("Convergence rate of Pade scheme") 
    plt.grid()
    plt.legend()
    show_or_save("fig_gammaTilde")

def fig_gammaTilde():
    import matplotlib.pyplot as plt
    dt=1.
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    r=.0
    assert r == 0.
    def mu_plus(w):
        z = np.exp(-1j*w*dt)
        s = (z - 1)/z
        return z*(1/np.sqrt(2) * (1-s) + \
                1j*(-b/a**2)*np.sqrt((1+a**2*s**2 - 2*dt*a**2*r/z - (a**2+1)*s)/(-1)))
    def mu_minus(w):
        z = np.exp(-1j*w*dt)
        s = (z - 1)/z
        return z*(1/np.sqrt(2) * (1-s) - \
                1j*(-b/a**2)*np.sqrt((1+a**2*s**2 - 2*dt*a**2*r/z - (a**2+1)*s)/(-1)))

    def gamma(w):
        z = np.exp(-1j*w*dt)
        return z - b*(z-1) - b/2 * (z-1)**2
    w = np.linspace(0,pi, 1000)
    plt.plot(w, np.real((mu_minus(w) - gamma(w))/(mu_plus(w) - mu_minus(w))), label="Real part of $\\tilde{\\gamma}$")
    plt.plot(w, np.imag((mu_minus(w) - gamma(w))/(mu_plus(w) - mu_minus(w))), label="Imaginary part of $\\tilde{\\gamma}$")
    # plt.plot(w, np.abs(1-(mu_minus(w) - gamma(w))/(mu_plus(w) - mu_minus(w))), label="modulus of gamma")
    plt.title("Value of $\\tilde{\\gamma}$")
    plt.grid()
    plt.legend()
    show_or_save("fig_gammaTilde")

def fig_rootsManfredi():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=[9.6, 2.])
    plt.subplots_adjust(left=.07, bottom=.28, right=.97, top=.85)


    dt=1.
    a = 1+np.sqrt(2)
    r=.5
    nu_1 = 1.
    nu_2 = 2.
    #assert r == 0.
    def get_z_s(w):
        z = np.exp(-1j*w*dt)
        return z, (z - 1)/(z*dt)

    def square_root_interior(w):
        z, s = get_z_s(w)
        return 1j*np.sqrt(-1*(1+(a*dt*s)**2 - (a**2+1)*dt*s))

    def sigma_plus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s + a**2*dt*r + square_root_interior(w))/(a*np.sqrt(dt*nu))

    def sigma_minus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s + a**2*dt*r - square_root_interior(w))/(a*np.sqrt(dt*nu))

    w = np.exp(np.linspace(-8, np.log(pi), 1000))[:-1]
    ref =(np.real(sigma_minus(w, nu_1)))

    axes[0].semilogx(w, (np.real(sigma_minus(w, nu_1))), label="$\\sigma_1$")

    axes[0].semilogx(w, np.abs(np.real(-sigma_minus(w, nu_2))), label="$\\sigma_2$")
    axes[0].semilogx(w, (np.real(sigma_plus(w, nu_1))), label="$\\sigma_3$")
    axes[0].semilogx(w, np.abs(np.real(-sigma_plus(w, nu_2))), label="$\\sigma_4$")

    axes[0].loglog(w, np.abs(np.real(np.sqrt((r+1j*w)/nu_1))), "k--", label="$\\sigma_j$ continuous")
    axes[0].semilogx(w, np.abs(np.real(-np.sqrt((r+1j*w)/nu_2))), "k--")

    #axes[0].loglog(w, np.abs(np.real(np.sqrt((r+1j*w + (4+3*np.sqrt(2))* (w*1j)**3 * dt**2/6)/nu_1) - ref)), label="$\\sigma_j$ modified")
    #axes[0].semilogx(w, np.abs(np.real(-np.sqrt((r+1j*w+ (4+3*np.sqrt(2))* (w*1j)**3 * dt**2/6)/nu_2))), "k--")



    axes[0].set_xlabel("$\\Delta t\\omega$")
    axes[0].set_ylabel("$\\mathfrak{R}(\\sigma)$")
    axes[0].set_title("Real part $|\\mathfrak{R}(\\sigma)|$")
    axes[0].grid()

    axes[1].loglog(w, np.abs(np.imag(sigma_minus(w, nu_1))), label="$\\sigma_1$")

    axes[1].loglog(w, np.abs(np.imag(-sigma_minus(w, nu_2))), label="$\\sigma_2$")
    axes[1].loglog(w, np.abs(np.imag(sigma_plus(w, nu_1))), label="$\\sigma_3$")
    axes[1].loglog(w, np.abs(np.imag(-sigma_plus(w, nu_2))), label="$\\sigma_4$")

    axes[1].loglog(w, np.abs(np.imag(np.sqrt((r+1j*w)/nu_1))), "k--", label="$\\sigma_j$ continuous")
    axes[1].loglog(w, np.abs(np.imag(-np.sqrt((r+1j*w)/nu_2))), "k--")
    axes[1].set_xlabel("$\\Delta t\\omega$")
    axes[1].set_ylabel("$Im(\\sigma)$")
    axes[1].set_title("Imaginary part $|Im(\\sigma)|$")
    axes[1].grid()

    plt.legend()
    show_or_save("fig_rootsManfredi")

def old_compare_rho_discrete_semidiscrete(axes, builder, N=3000):
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.R
    nu_1 = builder.D1
    nu_2 = builder.D2
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2

    def get_z(w):
        return np.exp(-1j*w*dt)

    def gamma(w):
        z, _ = get_z_s(w)
        return z - b*(z-1) - b/2 * (z-1)**2

    w = get_discrete_freq(N, dt)

    ##################################
    # CONTINUOUS CASE, discrete in time ofc
    ##################################

    def get_z_s(w):
        z = get_z(w)
        return z, (z - 1)/(z*dt)

    def square_root_interior(w):
        z, s = get_z_s(w)
        return np.sqrt(1 - dt*s) * np.sqrt(1 - a**2*dt*s)

    def sigma_plus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s +a**2*dt*r + square_root_interior(w))/(a*np.sqrt(dt*nu))

    def sigma_minus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s +a**2*dt*r - square_root_interior(w))/(a*np.sqrt(dt*nu))

    sigma_1 = sigma_minus(w, nu_1)
    sigma_2 = - sigma_minus(w, nu_2)
    sigma_3 = sigma_plus(w, nu_1)
    sigma_4 = -sigma_plus(w, nu_2)

    ##################################
    # DISCRETE CASE, discrete in time ofc
    ##################################

    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)

    def sqrt_g2_4f(w, nu):
        """
            computes the value sqrt(g^2 - 4f).
            This value must be carefully computed because
            it is the square root of a complex:
            a factorization of (g^2-4f) is needed.
        """
        z = get_z(w)
        # f is (q + d / z)/Gamma_b^2
        # g is (v / z - c)/Gamma_b
        q = (1 + r * dt * b)**2
        d = - 1 - a*r*dt
        v = a / b
        c = 2*(1+r*dt*b)
        Gamma_b = Gamma(b, nu)
        # now we have g^2-4f = (pol_a /z^2 + pol_b / z + pol_c)/Gamma_b^2
        pol_a = v**2
        pol_b = - (4*d + 2*c*v)
        pol_c = c**2 - 4*q

        first_term = np.sqrt((1/z - (-pol_b + np.sqrt(pol_b**2 - 4*pol_a*pol_c))/(2*pol_a)))
        second_term = np.sqrt((1/z - (-pol_b - np.sqrt(pol_b**2 - 4*pol_a*pol_c))/(2*pol_a)))

        return np.sqrt(pol_a) * first_term * second_term / Gamma_b

    def Gamma(ab, nu):
        return ab*dt*nu/h**2

    def get_f_g(w, nu):
        z = get_z(w)
        Gamma_a, Gamma_b = Gamma(a, nu), Gamma(b, nu)
        return ((1+b*r*dt)**2 - (1+a*r*dt)/z) / (Gamma_b**2), \
                (Gamma_a - 2*z*Gamma_b*(1 + r*dt*b)) / (z*Gamma_b**2)

    def lambda_pp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + sqrt_g2_4f(w, nu)/4 + np.sqrt((-(g-4)*(sqrt_g2_4f(w, nu) -g)/2 - f))/2

    def lambda_pm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + sqrt_g2_4f(w, nu)/4 - np.sqrt((-(g-4)*(sqrt_g2_4f(w, nu) -g)/2 - f))/2

    def lambda_mp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (sqrt_g2_4f(w, nu)/4 + np.sqrt(((g-4)*(sqrt_g2_4f(w, nu) +g)/2 - f))/2)

    def lambda_mm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (sqrt_g2_4f(w, nu)/4 - np.sqrt(((g-4)*(sqrt_g2_4f(w, nu) +g)/2 - f))/2)

    lambda_1 = lambda_mp(w, nu_1) # bon en fait normalement on pourrait utiliser pp et mm
    lambda_2 = lambda_mp(w, nu_2) # mais faudrait utiliser partout lambda_1^{-1} au lieu
    lambda_3 = lambda_pm(w, nu_1) # de lambda_1. Ca vaut pas vraiment le coup
    lambda_4 = lambda_pm(w, nu_2)

    sigma_1FD = np.log(lambda_1) / h
    sigma_2FD = np.log(lambda_2) / h
    sigma_3FD = np.log(lambda_3) / h
    sigma_4FD = np.log(lambda_4) / h

    chi_1 = h**2 * (r+1j*w)/nu_1
    chi_2 = h**2 * (r+1j*w)/nu_2
    lambda1_c = 1+chi_1/2 - np.sqrt(chi_1*(chi_1+4))/2 # Je comprends pas pourquoi c'est pas un +
    lambda2_c = 1+chi_2/2 - np.sqrt(chi_2*(chi_2+4))/2

    """
    axes.semilogx(w*dt, -sigma_1FD, label="$\\sigma_1$ FD")
    axes.semilogx(w*dt, sigma_2FD, label="$\\sigma_2$ FD")
    axes.semilogx(w*dt, sigma_1, "--", label="$\\sigma_1$ sd time")

    axes.semilogx(w*dt, sigma_2, "--", label="$\\sigma_2$ sd time")
    axes.semilogx(w*dt, -np.log(lambda1_c)/h, label="$\\sigma_1$ sd space")
    axes.semilogx(w*dt, np.sqrt((1j*w + r)/nu_1), label="$\\sigma_1$ continuous")
    axes.semilogx(w*dt, np.log(lambda2_c)/h, label="$\\sigma_2$ time continuous")

    axes.semilogx(w*dt, sigma_1FD, label="$\\sigma_1$ FD")
    axes.semilogx(w*dt, sigma_2FD, label="$\\sigma_2$ FD")

    axes.semilogx(w*dt, sigma_3, "--", label="$\\sigma_3$ continuous")
    axes.semilogx(w*dt, sigma_4, "--", label="$\\sigma_4$ continuous")
    """
    #axes.semilogx(w*dt, sigma_1FD, label="$\\sigma_1$ FD")
    #axes.semilogx(w*dt, sigma_2FD, label="$\\sigma_2$ FD")
    # axes.semilogx(w*dt, sigma_3FD, label="$\\sigma_3$ FD")
    # axes.semilogx(w*dt, sigma_4FD, label="$\\sigma_4$ FD")
    #axes.semilogx(w*dt, sigma_1, "k--", label="$\\sigma_1$ continuous")
    #axes.semilogx(w*dt, sigma_2, "k--", label="$\\sigma_1$ continuous")
    # axes.semilogx(w*dt, sigma_3, "k--", label="$\\sigma_3$ continuous")
    # axes.semilogx(w*dt, sigma_4, "k--", label="$\\sigma_3$ continuous")

    z = get_z(w)
    mu_1 = z*(1 + r*dt*b - b*dt*nu_1*sigma_1**2)
    mu_2 = z*(1 + r*dt*b - b*dt*nu_2*sigma_2**2)
    #z = get_z(-w)
    mu_3 = z*(1 + r*dt*b - b*dt*nu_1*sigma_3**2)
    mu_4 = z*(1 + r*dt*b - b*dt*nu_2*sigma_4**2)

    def mu_FD(w, nu_i, lambda_i):
        z = get_z(w)
        return z*(1 + r*dt*b - Gamma(b, nu_i)*(lambda_i - 2 + 1/lambda_i))

    # Comparing mu in continuous and discrete cases:
    # axes.plot(w*dt, mu_1, label="mu_1")
    # #axes.plot(w*dt, mu_2, "k--", label="mu_2")
    # axes.plot(w*dt, mu_3, label="mu_3")
    # #axes.plot(w*dt, mu_4, "k--", label="mu_4")

    # axes.plot(w*dt, mu_FD(w, nu_1, lambda_1), "k--", label="mu_1 FD")
    # axes.plot(w*dt, mu_FD(w, nu_2, lambda_2), "k-.", label="mu_2 FD")

    # axes.plot(w*dt, mu_FD(w, nu_1, lambda_3), label="mu_3 FD")
    # axes.plot(w*dt, mu_FD(w, nu_2, lambda_4), label="mu_4 FD")

    mu_1FD = mu_FD(w, nu_1, lambda_1)
    mu_2FD = mu_FD(w, nu_2, lambda_2)
    mu_3FD = mu_FD(w, nu_1, lambda_3)
    mu_4FD = mu_FD(w, nu_2, lambda_4)

    gamma_t1 = (mu_1FD - gamma(w))/(mu_1FD - mu_3FD)
    gamma_t2 = (mu_2FD - gamma(w))/(mu_2FD - mu_4FD)
    gamma_t = (mu_1 - gamma(w))/(mu_1 - mu_3)

    # comparing \\Tilde{gamma} to gamma_t1, gamma_t2
    # axes.loglog(w*dt, np.abs(np.imag(gamma_t1)), "k--", label="gammat1")
    # axes.loglog(w*dt, np.abs(np.imag(gamma_t2)), "k-.", label="gammat2")
    # axes.loglog(w*dt, np.abs(np.imag((mu_1 - gamma(w))/(mu_1 - mu_3))), label="gamma")
    # axes.loglog(w*dt, np.abs(np.real(gamma_t1)), "k--", label="gammat1r")
    # axes.loglog(w*dt, np.abs(np.real(gamma_t2)), "k-.", label="gammat2r")
    # axes.loglog(w*dt, (np.real((mu_1 - gamma(w))/(mu_1 - mu_3))), label="gammar")

    eta_22 = nu_2 * sigma_2
    eta_24 = nu_2 * sigma_4
    eta_11 = nu_1 * sigma_1
    eta_13 = nu_1 * sigma_3

    # DN: varrho_cont = ((1 - gamma_t) * eta_11 + gamma_t * eta_13) * ((1 - gamma_t) / eta_22 + gamma_t / eta_24)
    varrho_cont = ((L1 + eta_22)/(L2 + eta_22) * (1-gamma_t) + \
             (L1 + eta_24)/(L2 + eta_24) * (gamma_t)) * \
             ((L2 + eta_11)/(L1 + eta_11) * (1-gamma_t) + \
             (L2 + eta_13)/(L1 + eta_13) * (gamma_t))
    axes.semilogx(w*dt, np.abs((varrho_cont)), label="$\\rho^{\\rm Pade, c}$")

    # naive interface:
    eta_22 = nu_2 * (lambda_2-1)/h
    eta_24 = nu_2 * (lambda_4-1)/h
    eta_11 = nu_1 * (1-lambda_1)/h
    eta_13 = nu_1 * (1-lambda_3)/h

    #DN :
    varrho = ((1 - gamma_t1) * eta_11 + gamma_t1 * eta_13) * ((1 - gamma_t2) / eta_22 + gamma_t2 / eta_24)
    # RR:
    varrho = ((L1 + eta_22)/(L2 + eta_22) * (1-gamma_t2) + \
             (L1 + eta_24)/(L2 + eta_24) * (gamma_t2)) * \
             ((L2 + eta_11)/(L1 + eta_11) * (1-gamma_t1) + \
             (L2 + eta_13)/(L1 + eta_13) * (gamma_t1))

    #DN: axes.semilogx(w*dt, np.abs(nu_1/nu_2*(lambda1_c - 1)/(lambda2_c - 1)), "--", label="rho_sd_space")
    axes.semilogx(w*dt, np.abs((varrho)), label="$\\rho^{\\rm Pade, FD(corr=0)}$")


    # extrapolation:
    eta_11 = -nu_1/h * (lambda_1 - 1) * (3/2 - lambda_1/2)
    eta_13 = -nu_1/h * (lambda_3 - 1) * (3/2 - lambda_3/2)
    eta_22 = nu_2/h * (lambda_2 - 1) * (3/2 - lambda_2/2)
    eta_24 = nu_2/h * (lambda_4 - 1) * (3/2 - lambda_4/2)
    # DN: varrho = ((1 - gamma_t1) * eta_11 + gamma_t1 * eta_13) * ((1 - gamma_t2) / eta_22 + gamma_t2 / eta_24)
    # RR:
    varrho = ((L1 + eta_22)/(L2 + eta_22) * (1-gamma_t2) + \
             (L1 + eta_24)/(L2 + eta_24) * (gamma_t2)) * \
             ((L2 + eta_11)/(L1 + eta_11) * (1-gamma_t1) + \
             (L2 + eta_13)/(L1 + eta_13) * (gamma_t1))

    axes.semilogx(w*dt, np.abs((varrho)), label="$\\rho^{\\rm Pade, FD(extra)}$")

    axes.set_xlim(left=0)
    axes.legend()


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

    def frequency_cv_factor(self, time_discretization, space_discretization, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        return frequency_simulation(discretization, **kwargs)

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
