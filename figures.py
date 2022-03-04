#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import numpy as np
from memoisation import memoised
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
mpl.rcParams["figure.figsize"] = (8.4, 2.8)
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = ':'
mpl.rcParams["grid.alpha"] = '0.5'
mpl.rcParams["grid.linewidth"] = '0.5'
col = {}
col["blue"] = "#332288"
col["green"] = "#117733"
col["cyan"] = "#77DDCC"
col["yellow"] = "#DDCC77"
col["orange"] = "#D29977"
col["red"] = "#CC6677"
col["purple"] = "#882255"
col["grey"] = "#BBBBBB"
col_discrete=col["red"]
col_sdspace = col["green"]
col_sdtime = col["cyan"]
col_cont = col["yellow"]
col_cont_discop = col["orange"]
col_combined=col["blue"]
col_modified=col["purple"]
col_numeric = col["grey"]
symb_cont = "o"
symb_sdspace = "^"
symb_sdtime = "p"
symb_combined = "s"
symb_modified = "o"
symb_discrete = "P"
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from simulator import frequency_simulation, schwarz_simulator
from cv_factor_onestep import rho_c_c, rho_c_FV, rho_c_FD, DNWR_c_c, \
        DNWR_s_c, DNWR_c_FD
from cv_factor_onestep import rho_s_c
from ocean_models.ocean_Pade_FD import OceanPadeFD
from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD
from cv_factor_pade import rho_Pade_c, rho_Pade_FV, rho_Pade_FD, \
        DNWR_Pade_c, DNWR_Pade_FD

REAL_FIG = True
k_c=1

def rho_robin(builder, func, w, p1, p2, **kwargs):
    """
        Returns the cv factor defined with func where
        p1, p2 override the robin parameters of builder.
    """
    setting2 = builder.copy()
    setting2.LAMBDA_1 = p1
    setting2.LAMBDA_2 = p2
    return np.max(np.abs(func(setting2, w, **kwargs)))

def optimal_robin_parameter(builder, func, axis_freq, **kwargs):
    x0_eye1 = (1., -0.05)
    x0_eye2 = (0.05, -1.)
    def to_optimize(x):
        return rho_robin(builder, func, axis_freq, x[0], x[1],
                **kwargs)
    optimal_lam_eye1 = minimize(method='Nelder-Mead',
            fun=to_optimize, x0=x0_eye1)
    optimal_lam_eye2 = minimize(method='Nelder-Mead',
            fun=to_optimize, x0=x0_eye2)
    if optimal_lam_eye1.fun < optimal_lam_eye2.fun:
        return optimal_lam_eye1
    return optimal_lam_eye2

def fig_introDiscreteAnalysis():
    setting = Builder()
    N = 1000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(bottom=0.15, right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[2])
    lw_important = 2.1
    ax = axes[1]

    setting.LAMBDA_1=1e10 # >=0
    setting.LAMBDA_2=-0. # <= 0
    theta = optimal_DNWR_parameter(setting, DNWR_c_c, axis_freq)
    ax.semilogx(axis_freq, np.abs(DNWR_c_c(setting, axis_freq, theta=theta)), lw=lw_important, color=col_cont)
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD, K_c=k_c)
    if REAL_FIG:
        alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=1, NUMBER_IT=1,
                laplace_real_part=0, T=N*setting.DT, init="white", relaxation=theta, ignore_cached=False)
        ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]), color=col_numeric)
    ax.semilogx(axis_freq, np.abs(DNWR_Pade_FD(setting, axis_freq, theta=theta, k_c=k_c)), "--", lw=lw_important, color=col_discrete)
    ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    ax = axes[0]
    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_c, axis_freq, overlap_L=overlap_M*h).x
    ax.semilogx(axis_freq, np.abs(rho_c_c(setting, axis_freq, overlap_L=0.)), lw=lw_important, label="Continuous convergence rate", color=col_cont)
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD, K_c=k_c)
    if REAL_FIG:
        alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=1, NUMBER_IT=1,
                laplace_real_part=0, T=N*setting.DT, init="white")
        ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]), label="Numerical simulation",color=col_numeric)
    ax.semilogx(axis_freq, np.abs(rho_Pade_FD(setting, axis_freq, overlap_M=0, k_c=k_c)), "--", lw=lw_important, label="Discrete convergence rate",color=col_discrete)
    ax.set_title(r"$RR, "+ ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.7, 0.6))
    show_or_save("fig_introDiscreteAnalysis")

def combined_Pade(setting, axis_freq, overlap_M=0, k_c=k_c):
    """
        Returns the combined cv factor.
        Setting.h must be provided.
    """
    combined = - rho_c_c(setting, axis_freq,
            overlap_L=overlap_M*setting.h) \
                + rho_Pade_c(setting, axis_freq,
                        overlap_L=overlap_M*setting.h) \
                + rho_c_FD(setting, axis_freq,
                        overlap_M=overlap_M, k_c=k_c)
    return np.abs(combined)

def combined_Pade_DNWR(setting, axis_freq, theta, k_c=k_c):
    builder = setting.copy()
    combined = - DNWR_c_c(builder, axis_freq, theta=theta) \
                + DNWR_Pade_c(builder, axis_freq, theta=theta) \
                + DNWR_c_FD(builder, axis_freq, theta=theta, k_c=k_c)
    return np.abs(combined)

def modified_FD_DNWR(builder, axis_freq, theta, k_c=k_c):
    Gamma_1 = builder.DT * builder.D1 / builder.h**2
    Gamma_2 = builder.DT * builder.D2 / builder.h**2
    # Finite differences:
    d1 = 1/12
    d2 = 1/360
    s_c = 1j*axis_freq
    s_modified1 = s_c - d1 * builder.DT/Gamma_1 * (s_c + builder.R)**2
    s_modified2 = s_c - d1 * builder.DT/Gamma_2 * (s_c + builder.R)**2
    return np.abs(DNWR_s_c(builder, s_modified1, s_modified2,
        w=axis_freq, theta=theta, continuous_interface_op=False,
        k_c=k_c))

def modified_FD(builder, axis_freq, overlap_L=0, k_c=k_c):
    Gamma_1 = builder.DT * builder.D1 / builder.h**2
    Gamma_2 = builder.DT * builder.D2 / builder.h**2
    # Finite differences:
    d1 = 1/12
    d2 = 1/360
    s_c = 1j*axis_freq
    s_modified1 = s_c - d1 * builder.DT/Gamma_1 * (s_c + builder.R)**2
    s_modified2 = s_c - d1 * builder.DT/Gamma_2 * (s_c + builder.R)**2
    return np.abs(rho_s_c(builder, s_modified1, s_modified2,
        w=axis_freq, overlap_L=overlap_L,
        continuous_interface_op=False, k_c=k_c))

def relative_acceleration_combined(builder, N):
    """
        with the parameters contained in builder,
        compute the relative difference between
        the max of the discrete convergence rates,
        obtained with the continuous analysis on one hand
        and with the combined analysis on the other hand.
    """
    axis_freq = get_discrete_freq(N, builder.DT)[int(N//2)+1:]
    #optimization routine:

    # 1.a optimization in the continuous case
    cont = optimal_robin_parameter(builder, rho_c_c,
            axis_freq, k_c=k_c)
    # 1.b optimization in the combined case
    combined = optimal_robin_parameter(builder, combined_Pade,
            axis_freq, k_c=k_c)
    # 2. computation of the discrete rate associated
    cont_rho = np.max(rho_robin(builder, rho_Pade_FD, axis_freq,
            *cont.x, k_c=k_c))
    combined_rho = np.max(rho_robin(builder, rho_Pade_FD, axis_freq,
            *combined.x, k_c=k_c))
    # 3. relative difference
    return (cont_rho - combined_rho) / cont_rho

def relative_acceleration_combined_DNWR(builder, N):
    """
        with the parameters contained in builder,
        compute the relative difference between
        the max of the discrete convergence rates,
        obtained with the continuous analysis on one hand
        and with the combined analysis on the other hand.
    """
    axis_freq = get_discrete_freq(N, builder.DT)[int(N//2)+1:]
    #optimization routine:

    # 1.a optimization in the continuous case
    cont_theta = optimal_DNWR_parameter(builder, DNWR_c_c, axis_freq)
    # 1.b optimization in the combined case
    combined_theta = optimal_DNWR_parameter(builder, combined_Pade_DNWR,
            axis_freq, k_c=k_c)
    # 2. computation of the discrete rate associated
    cont_rho = np.max(np.abs(DNWR_Pade_FD(builder, axis_freq,
            cont_theta, k_c=k_c)))
    combined_rho = np.max(np.abs(DNWR_Pade_FD(builder, axis_freq,
            combined_theta, k_c=k_c)))
    # 3. relative difference
    return (cont_rho - combined_rho) / cont_rho

def relative_acceleration_modified(builder, N):
    """
        with the parameters contained in builder,
        compute the relative difference between
        the max of the discrete convergence rates,
        obtained with the continuous analysis on one hand
        and with the combined analysis on the other hand.
    """
    axis_freq = get_discrete_freq(N, builder.DT)[int(N//2)+1:]
    #optimization routine:

    # 1.a optimization in the continuous case (disc. op.)
    cont = optimal_robin_parameter(builder, rho_c_c,
            axis_freq, continuous_interface_op=False, k_c=k_c)
    # 1.b optimization in the modified case
    modified = optimal_robin_parameter(builder, modified_FD,
            axis_freq, k_c=k_c)
    # 2. computation of the discrete rate associated
    cont_rho = np.max(rho_robin(builder, rho_Pade_FD, axis_freq,
            *cont.x, k_c=k_c))
    modified_rho = np.max(rho_robin(builder, rho_Pade_FD, axis_freq,
            *modified.x, k_c=k_c))
    # 3. relative difference
    return (cont_rho - modified_rho) / cont_rho

def relative_acceleration_modified_DNWR(builder, N):
    """
        with the parameters contained in builder,
        compute the relative difference between
        the max of the discrete convergence rates,
        obtained with the continuous analysis on one hand
        and with the combined analysis on the other hand.
    """
    axis_freq = get_discrete_freq(N, builder.DT)[int(N//2)+1:]
    #optimization routine:

    # 1.a optimization in the continuous case (disc. op.)
    cont_theta = optimal_DNWR_parameter(builder, DNWR_c_c,
            axis_freq, continuous_interface_op=False, k_c=k_c)
    # 1.b optimization in the modified case
    modified_theta = optimal_DNWR_parameter(builder, modified_FD_DNWR,
            axis_freq, k_c=k_c)
    # 2. computation of the discrete rate associated
    cont_rho = np.max(np.abs(DNWR_Pade_FD(builder, axis_freq,
            cont_theta, k_c=k_c)))
    modified_rho = np.max(np.abs(DNWR_Pade_FD(builder, axis_freq,
            modified_theta, k_c=k_c)))
    # 3. relative difference
    return (cont_rho - modified_rho) / cont_rho

def fig_robustness_r():
    N = 1000
    builder = Builder()
    COMPUTE_AGAIN = False
    def function_to_plot(r, func):
        setting = builder.copy()
        setting.R = r
        return func(setting, N)

    resolution = 80
    rmin, rmax = 1e-6, 1.
    allr = np.geomspace(rmin, rmax, resolution)

    name_fileRR = "cache_npy/direct_combined_rRR.npy"
    name_fileDNWR = "cache_npy/direct_combined_rDNWR.npy"
    name_fileRR_modif = "cache_npy/direct_modified_rRR.npy"
    name_fileDNWR_modif = "cache_npy/direct_modified_rDNWR.npy"
    if COMPUTE_AGAIN:
        Z_combinedRR = [function_to_plot(r,
            relative_acceleration_combined) for r in allr]
        Z_combinedDNWR = [function_to_plot(r,
            relative_acceleration_combined_DNWR) for r in allr]
        Z_modifiedRR = [function_to_plot(r,
            relative_acceleration_modified) for r in allr]
        Z_modifiedDNWR = [function_to_plot(r,
            relative_acceleration_modified_DNWR) for r in allr]
        np.save(name_fileRR, Z_combinedRR)
        np.save(name_fileDNWR, Z_combinedDNWR)
        np.save(name_fileRR_modif, Z_modifiedRR)
        np.save(name_fileDNWR_modif, Z_modifiedDNWR)
    else:
        Z_combinedRR = np.load(name_fileRR)
        Z_combinedDNWR = np.load(name_fileDNWR)
        Z_modifiedDNWR = np.load(name_fileDNWR_modif)
        Z_modifiedRR = np.load(name_fileRR_modif)

    fig, ax = plt.subplots(1, 1, figsize=(6., 2.))
    fig.subplots_adjust(bottom=0.23)
    ax.loglog(allr, Z_combinedRR, label="Combined, RR",
            color=col_combined)
    ax.loglog(allr, Z_combinedDNWR, "--", label="Combined, DNWR",
            color=col_combined)
    ax.loglog(allr, Z_modifiedRR, label="Modified, RR",
            color=col_modified)
    ax.loglog(allr, Z_modifiedDNWR, "--", label="Modified, DNWR",
            color=col_modified)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"Relative acceleration")
    ax.legend()
    show_or_save("fig_robustness_r")

def fig_robustness():
    N = 1000
    builder = Builder()
    COMPUTE_AGAIN = False

    def function_to_plot_combined(nu1, nu2):
        setting = builder.copy()
        setting.D1, setting.D2 = nu1, nu2
        return relative_acceleration_combined(setting, N)

    resolution = 50
    nu1min, nu1max, nu2min, nu2max = .1, 4., .1, 4.
    allnu1 = np.linspace(nu1min, nu1max, resolution)
    allnu2 = np.linspace(nu2min, nu2max, resolution)
    X, Y = np.meshgrid(allnu1, allnu2)

    name_file = "cache_npy/direct_combined_nu.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_combined)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)

    fig, axes = plt.subplots(2, 4, figsize=(8, 5))
    fig.subplots_adjust(hspace=0.5)
    levels = np.linspace(np.min(Z), np.max(Z), 9)
    levels = [0., 0.1, 0.12, 0.13, 0.14]
    CS = axes[0, 0].contour(X, Y, Z, levels=levels)
    axes[0, 0].clabel(CS, inline=True, fontsize=10)
    axes[0, 0].set_title(r"Combined, RR")
    axes[0, 0].set_xlabel(r"$\nu_1$")
    axes[0, 0].set_ylabel(r"$\nu_2$")

    def function_to_plot_modified(nu1, nu2):
        setting = builder.copy()
        setting.D1, setting.D2 = nu1, nu2
        return relative_acceleration_modified(setting, N)

    name_file = "cache_npy/direct_modified_nu.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_modified)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)
    levels = [0., 0.01, 0.02]
    CS = axes[0, 2].contour(X, Y, Z, levels=levels)
    axes[0, 2].clabel(CS, inline=True, fontsize=10)
    axes[0, 2].set_title(r"Modified, RR")
    axes[0, 2].set_xlabel(r"$\nu_1$")

    def function_to_plot_combined(h, dt):
        setting = builder.copy()
        setting.h, setting.DT = h, dt
        setting.M1 = setting.M2 = 1001
        setting.SIZE_DOMAIN_1 = setting.h*(setting.M1-1)
        setting.SIZE_DOMAIN_2 = setting.h*(setting.M2-1)
        return relative_acceleration_combined(setting, N)

    hmin, hmax, dtmin, dtmax = .1, 10., .1, 50.
    allh = np.linspace(hmin, hmax, resolution)
    alldt = np.linspace(dtmin, dtmax, resolution)
    X, Y = np.meshgrid(allh, alldt)

    name_file = "cache_npy/direct_combined_hdt.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_combined)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)
    levels = [-0.5, -0.3, -0.1, 0.1, 0.3]
    CS = axes[1, 0].contour(X, Y, Z, levels=levels)
    axes[1, 0].clabel(CS, inline=True, fontsize=10)
    axes[1, 0].set_title(r"Combined, RR")
    axes[1, 0].set_xlabel(r"$h$")
    axes[1, 0].set_ylabel(r"$\Delta t$")

    def function_to_plot_modified(h, dt):
        setting = builder.copy()
        setting.h, setting.DT = h, dt
        setting.M1 = setting.M2 = 1001
        setting.SIZE_DOMAIN_1 = setting.h*(setting.M1-1)
        setting.SIZE_DOMAIN_2 = setting.h*(setting.M2-1)
        return relative_acceleration_modified(setting, N)
    name_file = "cache_npy/direct_modified_hdt.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_modified)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)

    levels = np.linspace(np.min(Z), np.max(Z), 9)
    levels = [-0.01, 0., 0.01, 0.02]
    CS = axes[1, 2].contour(X, Y, Z, levels=levels)
    axes[1, 2].clabel(CS, inline=True, fontsize=10)
    axes[1, 2].set_title(r"Modified, RR")
    axes[1, 2].set_xlabel(r"$h$")

    def function_to_plot_combined(nu1, nu2):
        setting = builder.copy()
        setting.D1, setting.D2 = nu1, nu2
        return relative_acceleration_combined_DNWR(setting, N)

    resolution = 50
    nu1min, nu1max, nu2min, nu2max = .1, 4., .1, 4.
    allnu1 = np.linspace(nu1min, nu1max, resolution)
    allnu2 = np.linspace(nu2min, nu2max, resolution)
    X, Y = np.meshgrid(allnu1, allnu2)

    name_file = "cache_npy/direct_combined_nu_DNWR.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_combined)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)

    levels = np.linspace(np.min(Z), np.max(Z), 9)
    levels = [0., 0.1, 0.2, 0.3]
    CS = axes[0, 1].contour(X, Y, Z, levels=levels)
    axes[0, 1].clabel(CS, inline=True, fontsize=10)
    axes[0, 1].set_title(r"Combined, DNWR")
    axes[0, 1].set_xlabel(r"$\nu_1$")

    def function_to_plot_modified(nu1, nu2):
        setting = builder.copy()
        setting.D1, setting.D2 = nu1, nu2
        return relative_acceleration_modified_DNWR(setting, N)

    name_file = "cache_npy/direct_modified_nu_DNWR.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_modified)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)
    levels = [-0.02, -0.01, 0., 0.01, 0.02]
    CS = axes[0, 3].contour(X, Y, Z, levels=levels)
    axes[0, 3].clabel(CS, inline=True, fontsize=10)
    axes[0, 3].set_title(r"Modified, DNWR")
    axes[0, 3].set_xlabel(r"$\nu_1$")

    def function_to_plot_combined(h, dt):
        setting = builder.copy()
        setting.h, setting.DT = h, dt
        setting.M1 = setting.M2 = 1001
        setting.SIZE_DOMAIN_1 = setting.h*(setting.M1-1)
        setting.SIZE_DOMAIN_2 = setting.h*(setting.M2-1)
        return relative_acceleration_combined_DNWR(setting, N)

    hmin, hmax, dtmin, dtmax = .1, 10., .1, 50.
    allh = np.linspace(hmin, hmax, resolution)
    alldt = np.linspace(dtmin, dtmax, resolution)
    X, Y = np.meshgrid(allh, alldt)

    name_file = "cache_npy/direct_combined_hdt_DNWR.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_combined)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)
    levels = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
    CS = axes[1, 1].contour(X, Y, Z, levels=levels)
    axes[1, 1].clabel(CS, inline=True, fontsize=10)
    axes[1, 1].set_title(r"Combined, DNWR")
    axes[1, 1].set_xlabel(r"$h$")

    def function_to_plot_modified(h, dt):
        setting = builder.copy()
        setting.h, setting.DT = h, dt
        setting.M1 = setting.M2 = 1001
        setting.SIZE_DOMAIN_1 = setting.h*(setting.M1-1)
        setting.SIZE_DOMAIN_2 = setting.h*(setting.M2-1)
        return relative_acceleration_modified_DNWR(setting, N)
    name_file = "cache_npy/direct_modified_hdt_DNWR.npy"
    if COMPUTE_AGAIN:
        vfunc = np.vectorize(function_to_plot_modified)
        Z = vfunc(X, Y)
        np.save(name_file, Z)
    else:
        Z = np.load(name_file)

    levels = np.linspace(np.min(Z), np.max(Z), 9)
    levels = [-0.01, 0., 0.01, 0.02]
    CS = axes[1, 3].contour(X, Y, Z, levels=levels)
    axes[1, 3].clabel(CS, inline=True, fontsize=10)
    axes[1, 3].set_title(r"Modified, DNWR")
    axes[1, 3].set_xlabel(r"$h$")

    show_or_save('fig_robustness')

def fig_L2normsRR():
    #######
    # initialization
    #######
    setting = Builder()
    N = 1000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    #######
    # Optimization
    #######
    def callrho(fun, p1p2, **kwargs):
        return rho_robin(setting, fun, axis_freq, p1p2[0], p1p2[1], **kwargs)

    cont = optimal_robin_parameter(setting, rho_c_c, axis_freq)
    discrete = optimal_robin_parameter(setting, rho_Pade_FD,
            axis_freq, k_c=k_c)
    combined = optimal_robin_parameter(setting, combined_Pade,
            axis_freq, k_c=k_c)

    ######
    # validation
    ######
    NUMBER_IT=5
    errors_cont = memoised(L2_evolutionPadeFD, tuple(cont.x),
        NUMBER_IT, setting, N)
    errors_combined = memoised(L2_evolutionPadeFD, tuple(combined.x),
        NUMBER_IT, setting, N)
    errors_discrete = memoised(L2_evolutionPadeFD, tuple(discrete.x),
        NUMBER_IT, setting, N)

    ########
    # Plotting
    ########
    k_scatter = np.arange(0, NUMBER_IT+2)
    k = k_scatter[1:]
    size_symb = 80
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 2.8))
    fig.subplots_adjust(left=.15, bottom=.15, right=.963, top=.963)

    lab_cont=ax.scatter(k_scatter, errors_cont,
            marker=symb_cont, alpha=1., s=size_symb,
            c=col_cont, label="Continuous")

    lab_discrete=ax.scatter(k_scatter, errors_discrete,
            marker=symb_discrete, alpha=1., s=size_symb,
            c=col_discrete, label="Discrete")

    lab_combined=ax.scatter(k_scatter, errors_combined,
            marker=symb_combined,
            alpha=1., s=size_symb, edgecolors=col_combined,
            facecolors="none", linewidth=2., label="Combined")

    # upper bound:
    lab_L2cont=ax.semilogy(k, errors_cont[1]* cont.fun**(k-1),
            "--", color=col_cont, label=r"$\propto \max(\rho^{(c,c)})^k$")
    lab_L2discrete=ax.semilogy(k, errors_discrete[1]* discrete.fun**(k-1),
            "--", color=col_discrete, label=r"$\propto \max(\rho^{(DIRK, FD)})^k$")
    lab_L2combined=ax.semilogy(k, errors_combined[1]* combined.fun**(k-1),
            "--", color=col_combined, label=r"$\propto \max(\rho^{(DIRK, FD)}_{\rm combined})^k$")

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$||e^k||_2$")
    legend = [lab_cont, lab_discrete, lab_combined,
            lab_L2cont[0], lab_L2discrete[0], lab_L2combined[0]]
    ax.legend([lab for lab in legend],
            [lab.get_label() for lab in legend],)
    show_or_save("fig_L2normsRR")

def L2_evolutionPadeFD(robin_param, NUMBER_IT, setting, N):
    builder = setting.copy()
    builder.LAMBDA_1, builder.LAMBDA_2 = robin_param
    ocean, atmosphere = builder.build(OceanPadeFD, AtmospherePadeFD)
    errors = schwarz_simulator(atmosphere, ocean, T=N*builder.DT,
            NUMBER_IT=NUMBER_IT)
    return np.linalg.norm(errors, axis=-1)

def fig_RobinTwoSided():
    size_symb = 80

    setting = Builder()
    N = 1000
    overlap_M = 0
    res_x = 60

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    fig, axes = plt.subplots(1, 2)
    fig.subplots_adjust(right=0.97, hspace=0.65, wspace=0.28)

    dt = setting.DT

    D1, D2 = setting.D1, setting.D2
    # Finite differences:
    s_c = 1j*axis_freq # BE_s(dt, axis_freq)

    from cv_factor_onestep import rho_c_FD, rho_c_c
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD

    ax = axes[0]
    p1min, p1max = 0.03, 1.5
    p2min, p2max = -.4, -.01
    p1_range = np.arange(p1min, p1max, (p1max - p1min)/res_x)
    p2_range = np.arange(p2min, p2max, (p2max - p2min)/res_x)
    Z = [[rho_robin(setting, rho_Pade_FD, axis_freq, p1, p2, overlap_M=0, k_c=k_c)
            for p2 in p2_range] for p1 in p1_range]
    p1_arr = [[p1 for p2 in p2_range] for p1 in p1_range]
    p2_arr = [[p2 for p2 in p2_range] for p1 in p1_range]
    fig.colorbar(ax.contour(p1_arr, p2_arr, Z, zorder=-1), ax=ax)

    def callrho(fun, p1p2, **kwargs):
        return rho_robin(setting, fun, axis_freq, p1p2[0], p1p2[1], **kwargs)
    eye1 = (1., -0.05)
    eye2 = (0.05, -1.)

    cont = optimal_robin_parameter(setting, rho_c_c, axis_freq)
    s_d_space = optimal_robin_parameter(setting, rho_c_FD, axis_freq,
            k_c=k_c)
    s_d_time = optimal_robin_parameter(setting, rho_Pade_c, axis_freq)
    discrete = optimal_robin_parameter(setting, rho_Pade_FD, axis_freq,
            k_c=k_c)
    combined = optimal_robin_parameter(setting, combined_Pade, axis_freq,
            k_c=k_c)
    ax.scatter(*cont.x, marker=symb_cont, alpha=1., s=size_symb, c=col_cont,
            label="Continuous: {:.3f}".format(callrho(rho_Pade_FD,
                cont.x, overlap_M=overlap_M, k_c=k_c)))
    ax.scatter(*s_d_space.x, marker=symb_sdspace, alpha=1., s=size_symb, c=col_sdspace,
            label="S-d space: {:.3f}".format(callrho(rho_Pade_FD,
                s_d_space.x, overlap_M=overlap_M, k_c=k_c)))
    ax.scatter(*s_d_time.x, marker=symb_sdtime, alpha=1., s=size_symb, c=col_sdtime,
            label="S-d time: {:.3f}".format(callrho(rho_Pade_FD,
                s_d_time.x, overlap_M=overlap_M, k_c=k_c)))
    ax.scatter(*discrete.x, marker=symb_discrete, alpha=1., s=size_symb, c=col_discrete,
            label="Discrete: {:.3f}".format(discrete.fun))
    ax.scatter(*combined.x, marker=symb_combined, alpha=1., s=size_symb,
            edgecolors=col_combined, facecolors="none", linewidth=2.,
            label="Combined: {:.3f}".format(callrho(rho_Pade_FD,
                combined.x, k_c=k_c, overlap_M=overlap_M)))
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.legend()
    ax.set_title("Combined")

    def maxrho(func, p, **kwargs):
        builder = setting.copy()
        builder.LAMBDA_1, builder.LAMBDA_2 = p, -p
        return np.max(np.abs(func(builder, **kwargs)))

    ax = axes[1]
    p1min, p1max = 1.2, 1.6
    p2min, p2max = -.1, -.0
    p1_range = np.arange(p1min, p1max, (p1max - p1min)/res_x)
    p2_range = np.arange(p2min, p2max, (p2max - p2min)/res_x)
    Z = [[rho_robin(setting, rho_c_FD, axis_freq, p1, p2, overlap_M=0, k_c=k_c)
            for p2 in p2_range] for p1 in p1_range]
    p1_arr = [[p1 for p2 in p2_range] for p1 in p1_range]
    p2_arr = [[p2 for p2 in p2_range] for p1 in p1_range]
    fig.colorbar(ax.contour(p1_arr, p2_arr, Z, zorder=-1), ax=ax)

    cont = optimal_robin_parameter(setting, rho_c_c, axis_freq,
            continuous_interface_op=False, k_c=k_c)
    s_d_space = optimal_robin_parameter(setting, rho_c_FD, axis_freq, k_c=k_c)
    modified = optimal_robin_parameter(setting, modified_FD, axis_freq,
            overlap_L=0, k_c=k_c)
    ax.scatter(*cont.x, marker=symb_cont, alpha=1., s=size_symb, c=col_cont_discop,
            label="Continuous (disc. op.): {:.3f}".format(callrho(rho_c_FD,
                cont.x, overlap_M=overlap_M, k_c=k_c)))
    ax.scatter(*s_d_space.x, marker=symb_sdspace, alpha=1., s=size_symb, c=col_sdspace,
            label="S-d space: {:.3f}".format(s_d_space.fun))
    ax.scatter(*modified.x, marker=symb_modified, alpha=1., s=size_symb, edgecolors=col_modified, facecolors="none", linewidth=2.,
            label="Modified: {:.3f}".format(callrho(rho_c_FD,
                modified.x, overlap_M=overlap_M, k_c=k_c)))
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.legend()
    ax.set_title("Modified")

    show_or_save("fig_RobinTwoSided")

def fig_dependency_maxrho_combined():
    setting = Builder()
    N = 1000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.15, right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[2])
    lw_important = 2.1

    from cv_factor_onestep import rho_c_FD, rho_c_c, DNWR_c_c, DNWR_c_FD
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD, DNWR_Pade_c, DNWR_Pade_FD

    def combined_Pade_DNWR(setting, axis_freq, theta, k_c):
        builder = setting.copy()
        combined = - DNWR_c_c(builder, axis_freq, theta=theta) \
                    + DNWR_Pade_c(builder, axis_freq, theta=theta) \
                    + DNWR_c_FD(builder, axis_freq, theta=theta, k_c=k_c)
        return np.abs(combined)

    ax = axes[1]
    all_theta = np.linspace(0.5,0.72,300)

    semidiscrete_space = [np.max(np.abs(DNWR_c_FD(setting, axis_freq, theta, k_c=k_c))) for theta in all_theta]
    semidiscrete_time = [np.max(np.abs(DNWR_Pade_c(setting, axis_freq, theta))) for theta in all_theta]
    discrete= [np.max(np.abs(DNWR_Pade_FD(setting, axis_freq, theta, k_c=k_c))) for theta in all_theta]
    continuous = [np.max(np.abs(DNWR_c_c(setting, axis_freq, theta))) for theta in all_theta]
    combined = [np.max(np.abs(combined_Pade_DNWR(setting, axis_freq, theta, k_c=k_c))) for theta in all_theta]
    ax.plot(all_theta, continuous, lw=lw_important, color=col_cont)
    ax.plot(all_theta, discrete, lw=lw_important, color=col_discrete)
    ax.plot(all_theta, combined, "--", lw=lw_important, color=col_combined)
    ax.plot(all_theta, semidiscrete_space, "--", color=col_sdspace)
    ax.plot(all_theta, semidiscrete_time, "--", color=col_sdtime)
    minima_indices = [np.argmin(continuous), np.argmin(discrete),
            np.argmin(combined), np.argmin(semidiscrete_space), np.argmin(semidiscrete_time)]
    col_minimas = [col_cont, col_discrete, col_combined, col_sdspace, col_sdtime]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=all_theta[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # xmin, xmax = ax.get_xlim()[0], all_theta[minima_indices]
    # ax.hlines(y=np.array(discrete)[minima_indices], xmin=xmin, xmax=xmax,
    #         colors=col_minimas, linestyle='dashed')
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\max_\omega (\rho)$")
    ax.set_title("DNWR")
    print("thetas: continuous: {}, discrete :{}, combined:{}, s-d space:{}, s-d time:{}".format(*all_theta[minima_indices]))
    print("rho_DNWR: continuous: {}, discrete :{}, combined:{}, s-d space:{}, s-d time:{}".format(*np.array(discrete)[minima_indices]))

    from cv_factor_onestep import rho_c_FD, rho_c_c, rho_s_c
    def maxrho(func, p, **kwargs):
        builder = setting.copy()
        builder.LAMBDA_1, builder.LAMBDA_2 = p, -p
        return np.max(np.abs(func(builder, **kwargs)))

    ax = axes[ 0]
    all_p1 = np.linspace(0.08,.34,300)
    discrete = [maxrho(rho_Pade_FD, p, w=axis_freq, overlap_M=overlap_M, k_c=k_c) for p in all_p1]
    semidiscrete_time = [maxrho(rho_Pade_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    semidiscrete_space = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M, k_c=k_c) for p in all_p1]
    continuous = [maxrho(rho_c_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    combined = [maxrho(combined_Pade, p, overlap_M=overlap_M,
        k_c=k_c, axis_freq=axis_freq) for p in all_p1]

    ax.plot(all_p1, continuous, label="Continuous", lw=lw_important, color=col_cont)
    ax.plot(all_p1, discrete, lw=lw_important, label="Discrete", color=col_discrete)
    ax.plot(all_p1, combined, "--", lw=lw_important, label="Combined", color=col_combined)
    ax.plot(all_p1, semidiscrete_space, "--", label="S-d space", color=col_sdspace)
    ax.plot(all_p1, semidiscrete_time, "--", label="S-d time", color=col_sdtime)
    minima_indices = [np.argmin(continuous), np.argmin(discrete),
            np.argmin(combined), np.argmin(semidiscrete_space), np.argmin(semidiscrete_time)]
    col_minimas = [col_cont, col_discrete, col_combined, col_sdspace, col_sdtime]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=all_p1[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # xmin, xmax = ax.get_xlim()[0], all_p1[minima_indices]
    # ax.hlines(y=np.array(discrete)[minima_indices], xmin=xmin, xmax=xmax,
    #         colors=col_minimas, linestyle='dashed')
    ax.set_xlabel(r"$p_1 = -p_2$")
    ax.set_ylabel(r"$\max_\omega (\rho)$")
    ax.set_title(r"$RR$")

    print("p1: continuous: {}, discrete :{}, combined:{}, s-d space:{}, s-d time:{}".format(*all_p1[minima_indices]))
    print("rho_RR: continuous: {}, discrete :{}, combined:{}, s-d space:{}, s-d time:{}".format(*np.array(discrete)[minima_indices]))

    # ax = axes[1, 0]
    # overlap_M = 1
    # setting.D1 = setting.D2

    # discrete = [maxrho(rho_Pade_FD, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    # semidiscrete_time = [maxrho(rho_Pade_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    # semidiscrete_space = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    # continuous = [maxrho(rho_c_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    # combined = [maxrho(combined_Pade, p, overlap_M=overlap_M) for p in all_p1]
    # ax.plot(all_p1, continuous, label="Continuous", lw=lw_important, color=col_cont)
    # ax.plot(all_p1, discrete, label="Discrete", lw=lw_important, color=col_discrete)
    # ax.plot(all_p1, combined, "--", label="Combined", lw=lw_important, color=col_combined)
    # ax.plot(all_p1, semidiscrete_space, "--", label="S-d space", color=col_sdspace)
    # ax.plot(all_p1, semidiscrete_time, "--", label="S-d time", color=col_sdtime)
    # minima_indices = [np.argmin(continuous), np.argmin(discrete),
    #         np.argmin(combined), np.argmin(semidiscrete_space), np.argmin(semidiscrete_time)]
    # col_minimas = [col_cont, col_discrete, col_combined, col_sdspace, col_sdtime]
    # ymin, ymax = ax.get_ylim()
    # ax.vlines(x=all_p1[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # # xmin, xmax = ax.get_xlim()[0], all_p1[minima_indices]
    # # ax.hlines(y=np.array(discrete)[minima_indices], xmin=xmin, xmax=xmax,
    # #         colors=col_minimas, linestyle='dashed')
    # ax.set_xlabel(r"$p_1 = -p_2$")
    # ax.set_ylabel(r"$\max_\omega (\rho)$")
    # ax.set_title(r"$RR^{M=1}$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.7, 0.6))
    show_or_save("fig_dependency_maxrho_combined")

def fig_dependency_maxrho_modified():
    setting = Builder()
    N = 1000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.97, hspace=0.65, wspace=0.41)
    fig.delaxes(ax= axes[2])
    lw_important = 2.1

    dt = setting.DT

    D1, D2 = setting.D1, setting.D2
    Gamma_1 = dt * D1 / h**2
    Gamma_2 = dt * D2 / h**2
    # Finite differences:
    d1 = 1/12
    d2 = 1/360
    s_c = 1j*axis_freq # BE_s(dt, axis_freq)
    s_modified1 = s_c - d1 * dt/Gamma_1 * (s_c + setting.R)**2
    s_modified2 = s_c - d1 * dt/Gamma_2 * (s_c + setting.R)**2

    from cv_factor_onestep import DNWR_c_c, DNWR_s_c, DNWR_c_FD
    ax = axes[1]
    all_theta = np.linspace(0.54,0.58,300)
    discrete = np.array([np.max(np.abs(DNWR_c_FD(setting, axis_freq, theta=theta, k_c=k_c)))
        for theta in all_theta])
    continuous = [np.max(np.abs(DNWR_s_c(setting, s_c, s_c, w=axis_freq,
        theta=theta, continuous_interface_op=False, k_c=k_c))) for theta in all_theta]
    modified_in_space = [np.max(np.abs(DNWR_s_c(setting, s_modified1, s_modified2, w=axis_freq,
        theta=theta, continuous_interface_op=False, k_c=k_c))) for theta in all_theta]
    ax.plot(all_theta, continuous, lw=lw_important, color=col_cont_discop)
    ax.plot(all_theta, discrete, lw=lw_important, color=col_sdspace)
    ax.plot(all_theta, modified_in_space, "--", lw=lw_important, color=col_modified)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\max_\omega (\rho)$")
    minima_indices = [np.argmin(discrete), np.argmin(continuous), np.argmin(modified_in_space)]
    col_minimas = [col_sdspace, col_cont_discop, col_modified]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=all_theta[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # xmin, xmax = ax.get_xlim()[0], all_theta[minima_indices]
    # ax.hlines(y=discrete[minima_indices], xmin=xmin, xmax=xmax,
    #         colors=col_minimas, linestyle='dashed')
    ax.set_title("DNWR")
    print("thetas: s-d space: {}, continuous :{}, modified:{}".format(*all_theta[minima_indices]))
    print("rho_DNWR: s-d space: {}, continuous :{}, modified:{}".format(*np.array(discrete)[minima_indices]))

    from cv_factor_onestep import rho_c_FD, rho_c_c, rho_s_c
    ax = axes[0]
    all_p1 = np.linspace(0.3,.32,300)
    def maxrho(func, p, **kwargs):
        builder = setting.copy()
        builder.LAMBDA_1, builder.LAMBDA_2 = p, -p
        return np.max(np.abs(func(builder, **kwargs)))

    discrete = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M, k_c=k_c) for p in all_p1]
    continuous = [maxrho(rho_s_c, p, s_1=s_c, s_2=s_c, w=axis_freq,
        overlap_L=overlap_M*h, continuous_interface_op=False, k_c=k_c) for p in all_p1]
    modified_in_space = [maxrho(rho_s_c, p, s_1=s_modified1, s_2=s_modified2, w=axis_freq,
        overlap_L=overlap_M*h, continuous_interface_op=False, k_c=k_c) for p in all_p1]
    ax.plot(all_p1, continuous, lw=lw_important, label="Continuous (disc. op.)", color=col_cont_discop)
    ax.plot(all_p1, discrete, lw=lw_important, label="S-d space",color=col_sdspace)
    ax.plot(all_p1, modified_in_space, "--", lw=lw_important, label="Modified",color=col_modified)
    minima_indices = [np.argmin(discrete), np.argmin(continuous), np.argmin(modified_in_space)]
    col_minimas = [col_sdspace, col_cont_discop, col_modified]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=all_p1[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # xmin, xmax = ax.get_xlim()[0], all_p1[minima_indices]
    # ax.hlines(y=np.array(discrete)[minima_indices], xmin=xmin, xmax=xmax,
    #         colors=col_minimas, linestyle='dashed')
    ax.set_xlabel(r"$p_1 = -p_2$")
    ax.set_ylabel(r"$\max_\omega (\rho)$")
    ax.set_title(r"$RR$")
    print("p1: s-d space: {}, continuous :{}, modified:{}".format(*all_p1[minima_indices]))
    print("rho_RR: s-d space: {}, continuous :{}, modified:{}".format(*np.array(discrete)[minima_indices]))

    # ax = axes[1, 0]
    # all_p1 = np.linspace(0.09,.1,300)
    # overlap_M = 1
    # setting.D1 = setting.D2

    # discrete = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    # continuous = [maxrho(rho_s_c, p, s_1=s_c, s_2=s_c,
    #     overlap_L=overlap_M*h, continuous_interface_op=False) for p in all_p1]
    # modified_in_space = [maxrho(rho_s_c, p, s_1=s_modified1, s_2=s_modified2,
    #     overlap_L=overlap_M*h, continuous_interface_op=False) for p in all_p1]
    # ax.plot(all_p1, continuous, label="Continuous (disc. op.)", lw=lw_important, color=col_cont_discop)
    # ax.plot(all_p1, discrete, label="S-d space", lw=lw_important, color=col_sdspace)
    # ax.plot(all_p1, modified_in_space, "--", label="Modified", lw=lw_important, color=col_modified)
    # minima_indices = [np.argmin(discrete), np.argmin(continuous), np.argmin(modified_in_space)]
    # col_minimas = [col_sdspace, col_cont_discop, col_modified]
    # ymin, ymax = ax.get_ylim()
    # ax.vlines(x=all_p1[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # # xmin, xmax = ax.get_xlim()[0], all_p1[minima_indices]
    # # ax.hlines(y=np.array(discrete)[minima_indices], xmin=xmin, xmax=xmax,
    # #         colors=col_minimas, linestyle='dashed')
    # ax.set_xlabel(r"$p_1 = -p_2$")
    # ax.set_ylabel(r"$\max_\omega (\rho)$")
    # ax.set_title(r"$RR^{M=1}$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.7, 0.6))
    show_or_save("fig_dependency_maxrho_modified")

def optimal_DNWR_parameter(builder, func, w, **kwargs):
    from scipy.optimize import minimize_scalar
    def to_optimize(x0):
        return np.max(np.abs(func(builder, w, x0, **kwargs)))
    optimal_lam = minimize_scalar(to_optimize)
    return optimal_lam.x

def fig_modif_time():
    from cv_factor_onestep import rho_s_c, rho_c_c
    from cv_factor_pade import rho_Pade_c
    setting = Builder()
    N = 1000
    overlap_M = 0
    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_Pade_c, axis_freq, overlap_L=overlap_M*h).x

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.15, right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[2])
    ax = axes[0]

    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    dt = setting.DT

    s_c = 1j*axis_freq # BE_s(dt, axis_freq)
    s_modified1 = s_c - (4 + 3*np.sqrt(2)) * dt**2/6 * 1j * axis_freq**3
    s_modified2 = s_c - (4 + 3*np.sqrt(2)) * dt**2/6 * 1j * axis_freq**3

    discrete = np.abs(rho_Pade_c(setting, axis_freq, overlap_L=overlap_M*h))
    continuous = np.abs(rho_c_c(setting, axis_freq, overlap_L=overlap_M*h))
    modified_in_time = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h))
    ax.semilogx(axis_freq, continuous, color=col_cont)
    ax.semilogx(axis_freq, discrete, color=col_sdtime)
    ax.semilogx(axis_freq, modified_in_time, "--", color=col_modified)
    ax.set_title(r"$RR, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    ax = axes[1]
    from cv_factor_pade import DNWR_Pade_c
    from cv_factor_onestep import DNWR_c_c, DNWR_s_c
    theta = optimal_DNWR_parameter(setting, DNWR_Pade_c, axis_freq)

    discrete = np.abs(DNWR_Pade_c(setting, axis_freq, theta=theta))
    continuous = np.abs(DNWR_c_c(setting, axis_freq, theta=theta))
    modified_in_time = np.abs(DNWR_s_c(setting, s_modified1, s_modified2, theta=theta))
    ax.semilogx(axis_freq, continuous, label="Continuous", color=col_cont)
    ax.semilogx(axis_freq, discrete, label="Semi-Discrete in time", color=col_sdtime)
    ax.semilogx(axis_freq, modified_in_time, "--", label="Modified in time", color=col_modified)
    ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    # ax = axes[1,0]
    # overlap_M = 1
    # setting.D1 = setting.D2

    # setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
    #         rho_Pade_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)

    # discrete = np.abs(rho_Pade_c(setting, axis_freq, overlap_L=overlap_M*h))
    # continuous = np.abs(rho_c_c(setting, axis_freq, overlap_L=overlap_M*h))
    # modified_in_time = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h))
    # ax.semilogx(axis_freq, continuous, color=col_cont)
    # ax.semilogx(axis_freq, discrete, color=col_sdtime)
    # ax.semilogx(axis_freq, modified_in_time, "--", color=col_modified)
    # ax.set_title(r"$RR^{M=1}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    # ax.set_xlabel(r"$\omega$")
    # ax.set_ylabel(r"$\rho$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.7, 0.6))
    show_or_save("fig_modif_time")


def fig_modif_space():
    from cv_factor_onestep import rho_c_FD, rho_s_c, rho_c_c
    setting = Builder()
    #setting.M1 = setting.M2 = 21 # warning, we change the parameter to highlight the differences
    N = 1000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_FD, axis_freq, overlap_M=overlap_M, k_c=k_c).x

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.15, right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[2])
    ax = axes[0]

    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    dt = setting.DT
    D1, D2 = setting.D1, setting.D2
    Gamma_1 = dt * D1 / h**2
    Gamma_2 = dt * D2 / h**2

    # Finite differences:
    d1 = 1/12
    s_c = 1j*axis_freq # BE_s(dt, axis_freq)
    s_modified1 = s_c - d1 * dt/Gamma_1 * (s_c + setting.R)**2
    s_modified2 = s_c - d1 * dt/Gamma_2 * (s_c + setting.R)**2

    cont = np.abs(rho_c_c(setting, w=axis_freq, overlap_L=overlap_M*h, continuous_interface_op=True))
    modified_in_space = np.abs(rho_s_c(setting, w=axis_freq, s_1=s_modified1, s_2=s_modified2, overlap_L=overlap_M*h, continuous_interface_op=False, k_c=k_c))
    ax.semilogx(axis_freq, cont, color=col_cont)
    ax.semilogx(axis_freq, np.abs(rho_c_FD(setting, axis_freq, overlap_M=overlap_M, k_c=k_c)), color=col_sdspace)
    ax.semilogx(axis_freq, modified_in_space, "--", color=col_modified)
    ax.semilogx(axis_freq, np.abs(rho_s_c(setting, w=axis_freq, s_1=1j*axis_freq, s_2=1j*axis_freq,
        overlap_L=overlap_M*h, continuous_interface_op=False, k_c=k_c)),
        color=col_cont_discop)

    ax.set_title(r"$RR, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    #ax.set_ylim(bottom=0.94, top=0.982)
    ax.set_ylabel(r"$\rho$")

    from cv_factor_onestep import DNWR_s_c, DNWR_c_FD, DNWR_c_c
    ax = axes[1]
    overlap_M = 0

    theta = optimal_DNWR_parameter(setting, DNWR_c_FD, axis_freq, k_c=k_c)

    ax.semilogx(axis_freq, np.abs(DNWR_c_c(setting, w=axis_freq, theta=theta)), label="Continuous", color=col_cont)
    ax.semilogx(axis_freq, np.abs(DNWR_s_c(setting, w=axis_freq, s_1=1j*axis_freq, s_2=1j*axis_freq, theta=theta, continuous_interface_op=False, k_c=k_c)), label="Continuous (disc. op.)", color=col_cont_discop)
    modified_in_space = np.abs(DNWR_s_c(setting, w=axis_freq, s_1=s_modified1, s_2=s_modified2, theta=theta, continuous_interface_op=False, k_c=k_c))
    ax.semilogx(axis_freq, np.abs(DNWR_c_FD(setting, axis_freq, theta=theta, k_c=k_c)), label="Semi-discrete in space", color=col_sdspace)
    ax.semilogx(axis_freq, modified_in_space, "--", label="Modified in space", color=col_modified)

    ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    # ax = axes[1, 0]
    # overlap_M = 1
    # setting.D1 = setting.D2

    # setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
    #         rho_c_FD, axis_freq, (0.1, -0.1), overlap_M=overlap_M)

    # ax.semilogx(axis_freq, np.abs(rho_s_c(setting, 1j*axis_freq, 1j*axis_freq,
    #     overlap_L=overlap_M*h, continuous_interface_op=False)),
    #     color=col_cont_discop)

    # modified_in_space = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h, continuous_interface_op=False))
    # ax.semilogx(axis_freq, np.abs(rho_c_FD(setting, axis_freq, overlap_M=overlap_M)), color=col_sdspace)
    # ax.semilogx(axis_freq, modified_in_space, "--", color=col_modified)

    # ax.set_title(r"$RR^{M=1}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    # ax.set_xlabel(r"$\omega$")
    # ax.set_ylabel(r"$\rho$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.7, 0.6))
    show_or_save("fig_modif_space")

def fig_combinedRate():
    setting = Builder()
    N = 1000
    w = get_discrete_freq(N, setting.DT)[int(N//2)+1:]
    overlap_M=0
    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.15, right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[2])
    ax = axes[0]

    from cv_factor_onestep import rho_c_FD, rho_c_c, DNWR_c_c, DNWR_c_FD
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD, DNWR_Pade_c, DNWR_Pade_FD

    def to_minimize_Pade(LAMBDAS, overlap_M, k_c):
        builder = setting.copy()
        builder.LAMBDA_1 = LAMBDAS[0]
        builder.LAMBDA_2 = LAMBDAS[1]
        return np.max(np.abs(rho_Pade_FD(builder, w, k_c=k_c, overlap_M=overlap_M)))

    def to_minimize_combined(LAMBDAS, overlap_M, k_c):
        builder = setting.copy()
        builder.LAMBDA_1 = LAMBDAS[0]
        builder.LAMBDA_2 = LAMBDAS[1]
        return np.max(np.abs(combined_Pade(builder, w, overlap_M=overlap_M, k_c=k_c)))

    from scipy.optimize import minimize
    # ret = minimize(method='Nelder-Mead', fun=to_minimize_combined, x0=np.array((0.15, -0.15)), args=overlap_M)
    ret = minimize(method='Nelder-Mead', fun=to_minimize_Pade, x0=np.array((0.15, -0.15)), args=(overlap_M, k_c))

    setting.LAMBDA_1 = ret.x[0]
    setting.LAMBDA_2 = ret.x[1]

    ax.semilogx(w, np.abs(rho_Pade_FD(setting, w, overlap_M=overlap_M, k_c=k_c)), color=col_discrete)
    ax.semilogx(w, np.abs(combined_Pade(setting, w, overlap_M=overlap_M, k_c=k_c)), color=col_combined)
    ax.semilogx(w, np.abs(rho_c_FD(setting, w, overlap_M=overlap_M, k_c=k_c)), "--", dashes=[7,9], color=col_sdspace)
    ax.semilogx(w, np.abs(rho_c_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[3,5], color=col_cont)
    ax.semilogx(w, np.abs(rho_Pade_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[7,9], color=col_sdtime)

    ax.set_xlabel(r"$\omega \Delta t$")
    ax.set_ylabel(r"$\rho$")
    #ax.set_ylim(bottom=0., top=0.5) # all axis are shared
    ax.set_title(r"$RR, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))

    ax = axes[1]

    def combined_Pade_DNWR(builder, w, theta, k_c):
        combined = - DNWR_c_c(builder, w, theta=theta) \
                    + DNWR_Pade_c(builder, w, theta=theta) \
                    + DNWR_c_FD(builder, w, theta=theta, k_c=k_c)
        return combined

    def to_minimize_Pade_DNWR(theta):
        return np.max(np.abs(DNWR_Pade_FD(setting, w, theta=theta, k_c=k_c)))

    def to_minimize_combined_DNWR(theta):
        return np.max(np.abs(combined_Pade_DNWR(setting, w, theta=theta, k_c=k_c)))

    from scipy.optimize import minimize_scalar
    theta = minimize_scalar(fun=to_minimize_Pade_DNWR).x

    ax.semilogx(w, np.abs(DNWR_Pade_FD(setting, w, theta, k_c=k_c)), label=r"$\rho^{\rm (DIRK, FD)}$", color=col_discrete)
    ax.semilogx(w, np.abs(combined_Pade_DNWR(setting, w, theta, k_c=k_c)), label=r"$\rho^{\rm (DIRK, FD)}_{\rm combined}$", color=col_combined)
    ax.semilogx(w, np.abs(DNWR_c_FD(setting, w, theta, k_c=k_c)), "--", label=r"$\rho^{\rm (c, FD)}$", dashes=[7,9], color=col_sdspace)
    ax.semilogx(w, np.abs(DNWR_c_c(setting, w, theta)), "--", label=r"$\rho^{\rm (c, c)}$", dashes=[3,5], color=col_cont)
    ax.semilogx(w, np.abs(DNWR_Pade_c(setting, w, theta)), "--", label=r"$\rho^{\rm (DIRK, c)}$", dashes=[7,9], color=col_sdtime)
    ax.set_xlabel(r"$\omega \Delta t$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))

    fig.legend(loc="upper left", bbox_to_anchor=(0.7, 0.6))
    show_or_save("fig_combinedRate")

def fig_validate_DNWR():
    from ocean_models.ocean_BE_FD import OceanBEFD
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
    from cv_factor_onestep import rho_c_c, rho_BE_c, rho_BE_FV, rho_BE_FD, rho_c_FV, rho_c_FD, DNWR_BE_FD
    setting = Builder()
    N = 10000
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    axis_freq = get_discrete_freq(N, setting.DT)
    ocean, atmosphere = setting.build(OceanBEFD, AtmosphereBEFD)
    alpha_w = frequency_simulation( atmosphere, ocean, number_samples=1, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="dirac", overlap=0, relaxation=.7)
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]))

    alpha_w_relaxed = frequency_simulation( atmosphere, ocean, number_samples=1, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="dirac", overlap=0, relaxation=0.5)
    ax.semilogx(axis_freq, np.abs(alpha_w_relaxed[2]/alpha_w_relaxed[1]))

    ax.semilogx(axis_freq, np.abs(DNWR_BE_FD(setting, axis_freq, theta=.7)), "--", label=r"$\theta=0.7$")
    ax.semilogx(axis_freq, np.abs(DNWR_BE_FD(setting, axis_freq, theta=.5)), "--", label=r"$\theta=0.5$")
    ax.set_title("BE")
    ax.legend()

    from ocean_models.ocean_Pade_FD import OceanPadeFD
    from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD
    from cv_factor_pade import rho_Pade_c, rho_Pade_FV, DNWR_Pade_c, DNWR_Pade_FD

    ax = axes[1]
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD)
    alpha_w = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", relaxation=.7)
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]))
    alpha_w_overlap = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", relaxation=.5)
    ax.semilogx(axis_freq, np.abs(alpha_w_overlap[2]/alpha_w_overlap[1]))

    ax.semilogx(axis_freq, np.abs(DNWR_Pade_FD(setting, axis_freq, theta=0.7)), "--", label=r"\theta=1.")
    ax.semilogx(axis_freq, np.abs(DNWR_Pade_FD(setting, axis_freq, theta=.5)), "--", label=r"\theta=0.5")

    ax.set_title("Pade")
    ax.legend()
    ax.set_xlabel(r"$\omega$")
    ax.set_xlabel(r"$\omega$")
    show_or_save("fig_validate_DNWR")

def fig_validate_overlap():
    from ocean_models.ocean_BE_FD import OceanBEFD
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
    from cv_factor_onestep import rho_c_c, rho_BE_c, rho_BE_FV, rho_BE_FD, rho_c_FV, rho_c_FD
    setting = Builder()
    setting.D1 = setting.D2
    N = 10000
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    axis_freq = get_discrete_freq(N, setting.DT)
    ocean, atmosphere = setting.build(OceanBEFD, AtmosphereBEFD)
    alpha_w_overlap = frequency_simulation( atmosphere, ocean, number_samples=1, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="dirac", overlap=1)
    ax.semilogx(axis_freq, np.abs(alpha_w_overlap[2]/alpha_w_overlap[1]))
    alpha_w_overlap = frequency_simulation( atmosphere, ocean, number_samples=1, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="dirac", overlap=2)
    ax.semilogx(axis_freq, np.abs(alpha_w_overlap[2]/alpha_w_overlap[1]))

    ax.semilogx(axis_freq, np.abs(rho_BE_FD(setting, axis_freq, overlap_M=2)), "--", label="M=2")
    ax.semilogx(axis_freq, np.abs(rho_BE_FD(setting, axis_freq, overlap_M=1)), "--", label="M=1")

    ocean.nu = setting.D1 = 0.5
    alpha_w = frequency_simulation(atmosphere, ocean, number_samples=1, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="dirac", overlap=0)
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]))
    ax.semilogx(axis_freq, np.abs(rho_BE_FD(setting, axis_freq, overlap_M=0)), "--", label=r"M=0, $\nu_1 \neq \nu_2$")
    ax.set_title("BE")
    ax.legend()
    ax.set_xlabel(r"$\omega$")

    from ocean_models.ocean_Pade_FD import OceanPadeFD
    from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD
    from cv_factor_pade import rho_Pade_c, rho_Pade_FV, rho_Pade_FD

    ax = axes[1]
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD)
    alpha_w_overlap = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", overlap=1)
    ax.semilogx(axis_freq, np.abs(alpha_w_overlap[2]/alpha_w_overlap[1]))
    alpha_w_overlap = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", overlap=2)
    ax.semilogx(axis_freq, np.abs(alpha_w_overlap[2]/alpha_w_overlap[1]))

    ax.semilogx(axis_freq, np.abs(rho_Pade_FD(setting, axis_freq, overlap_M=2)), "--", label="M=2")
    ax.semilogx(axis_freq, np.abs(rho_Pade_FD(setting, axis_freq, overlap_M=1)), "--", label="M=1")

    setting.D1 = 0.5
    ocean.nu = 0.5
    alpha_w = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", overlap=0)
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]))
    ax.semilogx(axis_freq, np.abs(rho_Pade_FD(setting, axis_freq, overlap_M=0)), "--", label=r"M=0, $\nu_1 \neq \nu_2$")

    ax.set_title(r"Pade, white needed when $\gamma$ uses future times")
    ax.set_xlabel(r"$\omega$")
    ax.legend()
    show_or_save("fig_validate_overlap")


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
        and then build all the schemes you want with theses parameters:
        dis_1 = builder.build(BackwardEuler, FiniteDifferencesNaive)
        dis_2 = builder.build(ThetaMethod, QuadSplinesFV)
        The comparison is thus then quite easy
    """
    def __init__(self): # changing defaults will result in needing to recompute all cache
        self.SIZE_DOMAIN_1 = 100
        self.SIZE_DOMAIN_2 = 100
        self.M1 = 101 # to have h=1 the number of points M_j must be 101
        self.M2 = 101
        self.h = 1.
        self.D1 = .5
        self.D2 = 1.
        self.R = 1e-3
        self.DT = 2.

        self.LAMBDA_1=1e10 # >=0
        self.LAMBDA_2=-0. # <= 0
        self.COURANT_NUMBER = self.D1 * self.DT / (self.SIZE_DOMAIN_1 / (self.M1-1))**2

    def copy(self):
        ret = Builder()
        ret.__dict__ = self.__dict__.copy()
        return ret

    def build(self, ocean_discretisation, atm_discretisation, **kwargs):
        """ build the models and returns tuple (ocean_model, atmosphere_model)"""
        ocean = ocean_discretisation(r=self.R, nu=self.D1, LAMBDA=self.LAMBDA_1,
            M=self.M1, SIZE_DOMAIN=self.SIZE_DOMAIN_1, DT=self.DT, **kwargs)
        atmosphere = atm_discretisation(r=self.R, nu=self.D2, LAMBDA=self.LAMBDA_2,
            M=self.M2, SIZE_DOMAIN=self.SIZE_DOMAIN_2, DT=self.DT, **kwargs)
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
    # Usually, we don't want the zero frequency so we use instead -1/T:
    if avoid_zero:
        all_k[int(N//2)] = -1.
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
