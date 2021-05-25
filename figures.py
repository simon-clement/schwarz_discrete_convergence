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
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams["figure.figsize"] = (8.4, 4.8)
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
import matplotlib.pyplot as plt
from simulator import frequency_simulation

REAL_FIG = True

def fig_introDiscreteAnalysis():
    setting = Builder()
    N = 10000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)

    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[1,1])
    lw_important = 2.1
    from cv_factor_onestep import rho_c_c, rho_c_FV, rho_c_FD, DNWR_c_c
    from ocean_models.ocean_Pade_FD import OceanPadeFD
    from atmosphere_models.atmosphere_Pade_FD import AtmospherePadeFD
    from cv_factor_pade import rho_Pade_c, rho_Pade_FV, rho_Pade_FD_corr0, DNWR_Pade_c, DNWR_Pade_FD

    ax = axes[0,1]

    setting.LAMBDA_1=1e10 # >=0
    setting.LAMBDA_2=-0. # <= 0
    theta = optimal_DNWR_parameter(setting, DNWR_c_c, axis_freq)
    ax.semilogx(axis_freq, np.abs(DNWR_c_c(setting, axis_freq, theta=theta)), lw=lw_important, color=col_cont)
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD)
    alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", relaxation=theta)
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]), color=col_numeric)
    ax.semilogx(axis_freq, np.abs(DNWR_Pade_FD(setting, axis_freq, theta=theta)), "--", lw=lw_important, color=col_discrete)
    ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    ax = axes[0,0]
    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)
    ax.semilogx(axis_freq, np.abs(rho_c_c(setting, axis_freq, overlap_L=0.)), lw=lw_important, color=col_cont)
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD)
    alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white")
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]), color=col_numeric)
    ax.semilogx(axis_freq, np.abs(rho_Pade_FD_corr0(setting, axis_freq, overlap_M=0)), "--", lw=lw_important, color=col_discrete)
    ax.set_title(r"$RR^{M=0}, "+ ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    overlap_M=1
    ax = axes[1,0]
    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)
    ax.semilogx(axis_freq, np.abs(rho_c_c(setting, axis_freq, overlap_L=overlap_M*h)), lw=lw_important,
            label="Continuous convergence rate", color=col_cont)
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD)
    alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", overlap=1)
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]), label="Numerical simulation", color=col_numeric)
    ax.semilogx(axis_freq, np.abs(rho_Pade_FD_corr0(setting, axis_freq, overlap_M=overlap_M)),
            "--", lw=lw_important, label="Discrete convergence rate", color=col_discrete)
    ax.set_title(r"$RR^{M=1}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.6, 0.4))
    show_or_save("fig_introDiscreteAnalysis")

# def fig_DNWR_why_not_optimal():
#     setting = Builder()
#     N = 10000
#     overlap_M = 0
# 
#     h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
#     assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10
# 
#     if overlap_M > 0:
#         setting.D1 = setting.D2
#     axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]
# 
#     fig, axes = plt.subplots(2, 2)
#     fig.delaxes(ax= axes[1,1])
# 
#     dt = setting.DT
# 
#     D1, D2 = setting.D1, setting.D2
#     Gamma_1 = dt * D1 / h**2
#     Gamma_2 = dt * D2 / h**2
#     # Finite differences:
#     d1 = 1/12
#     d2 = 1/360
#     s_c = 1j*axis_freq # BE_s(dt, axis_freq)
#     s_modified1 = s_c - d1 * dt/Gamma_1 * (s_c + setting.R)**2
#     s_modified2 = s_c - d1 * dt/Gamma_2 * (s_c + setting.R)**2
# 
#     from cv_factor_onestep import DNWR_c_c, DNWR_s_c, DNWR_c_FD
#     for ax, theta in zip((axes[0,0], axes[0,1], axes[1,0]), (0.6, 0.65, 0.7)):
#         ax.semilogx(axis_freq, np.abs(DNWR_c_FD(setting, axis_freq, theta=theta)), label="discrete")
#         ax.semilogx(axis_freq, np.abs(DNWR_s_c(setting, s_c, s_c, theta=theta, continuous_interface_op=False)), label="continuous")
#         ax.semilogx(axis_freq, np.abs(DNWR_s_c(setting, s_modified1, s_modified2, theta=theta, continuous_interface_op=False)), "--", label="modified")
#     fig.legend(loc='lower right')
#     ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))
#     show_or_save("fig_DNWR_why_not_optimal")

def fig_RobinTwoSided():
    symb_cont = "o"
    symb_sdspace = "^"
    symb_sdtime = "p"
    symb_combined = "s"
    symb_modified = "o"
    symb_discrete = "P"
    size_symb = 80

    setting = Builder()
    N = 10000
    overlap_M = 0
    res_x = 60

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.1, bottom=0.09, right=0.97, hspace=0.43, wspace=0.27)

    dt = setting.DT

    D1, D2 = setting.D1, setting.D2
    # Finite differences:
    s_c = 1j*axis_freq # BE_s(dt, axis_freq)

    from cv_factor_onestep import rho_c_FD, rho_c_c
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD_corr0

    def combined_Pade(setting, axis_freq, overlap_M=0):
        combined = - rho_c_c(setting, axis_freq, overlap_L=overlap_M*h) \
                    + rho_Pade_c(setting, axis_freq, overlap_L=overlap_M*h) \
                    + rho_c_FD(setting, axis_freq, overlap_M=overlap_M)
        return np.abs(combined)

    def optimal_robin_parameter(builder, func, w, x0, **kwargs):
        from scipy.optimize import minimize_scalar, minimize
        def to_optimize(x0):
            setting = builder.copy()
            setting.LAMBDA_1 = x0[0]
            setting.LAMBDA_2 = x0[1]
            return np.max(np.abs(func(setting, w, **kwargs)))
        optimal_lam = minimize(method='Nelder-Mead', fun=to_optimize, x0=x0)
        return optimal_lam


    def discrete_robin(builder, func, w, p1, p2, **kwargs):
        setting = builder.copy()
        setting.LAMBDA_1 = p1
        setting.LAMBDA_2 = p2
        return np.max(np.abs(func(setting, w, **kwargs)))

    ax = axes[0,0]
    p1min, p1max = 0.03, 1.
    p2min, p2max = -.4, -.01
    p1_range = np.arange(p1min, p1max, (p1max - p1min)/res_x)
    p2_range = np.arange(p2min, p2max, (p2max - p2min)/res_x)
    Z = [[discrete_robin(setting, rho_Pade_FD_corr0, axis_freq, p1, p2, overlap_M=0)
            for p2 in p2_range] for p1 in p1_range]
    p1_arr = [[p1 for p2 in p2_range] for p1 in p1_range]
    p2_arr = [[p2 for p2 in p2_range] for p1 in p1_range]
    fig.colorbar(ax.contour(p1_arr, p2_arr, Z, zorder=-1), ax=ax)

    def callrho(fun, p1p2, **kwargs):
        return discrete_robin(setting, fun, axis_freq, p1p2[0], p1p2[1], **kwargs)

    cont = optimal_robin_parameter(setting, rho_c_c, axis_freq, (0.1, -0.1))
    s_d_space = optimal_robin_parameter(setting, rho_c_FD, axis_freq, (0.1, -0.1))
    s_d_time = optimal_robin_parameter(setting, rho_Pade_c, axis_freq, (0.1, -0.1))
    discrete = optimal_robin_parameter(setting, rho_Pade_FD_corr0, axis_freq, (0.1, -0.1))
    combined = optimal_robin_parameter(setting, combined_Pade, axis_freq, (0.1, -0.1))
    ax.scatter(*cont.x, marker=symb_cont, alpha=1., s=size_symb, c=col_cont,
            label="Continuous: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                cont.x, overlap_M=overlap_M)))
    ax.scatter(*s_d_space.x, marker=symb_sdspace, alpha=1., s=size_symb, c=col_sdspace,
            label="S-d space: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                s_d_space.x, overlap_M=overlap_M)))
    ax.scatter(*s_d_time.x, marker=symb_sdtime, alpha=1., s=size_symb, c=col_sdtime,
            label="S-d time: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                s_d_time.x, overlap_M=overlap_M)))
    ax.scatter(*discrete.x, marker=symb_discrete, alpha=1., s=size_symb, c=col_discrete,
            label="Discrete: {:.3f}".format(discrete.fun))
    ax.scatter(*combined.x, marker=symb_combined, alpha=1., s=size_symb,
            edgecolors=col_combined, facecolors="none", linewidth=2.,
            label="Combined: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                combined.x, overlap_M=overlap_M)))
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.legend()
    ax.set_title("Combined without overlap")

    from cv_factor_onestep import rho_c_FD, rho_c_c, rho_s_c
    def maxrho(func, p, **kwargs):
        builder = setting.copy()
        builder.LAMBDA_1, builder.LAMBDA_2 = p, -p
        return np.max(np.abs(func(builder, **kwargs)))

    ax = axes[1,0]
    p1min, p1max = 0.03, .15
    p2min, p2max = -.3, -.15
    p1_range = np.arange(p1min, p1max, (p1max - p1min)/res_x)
    p2_range = np.arange(p2min, p2max, (p2max - p2min)/res_x)
    Z = [[discrete_robin(setting, rho_c_FD, axis_freq, p1, p2, overlap_M=0)
            for p2 in p2_range] for p1 in p1_range]
    p1_arr = [[p1 for p2 in p2_range] for p1 in p1_range]
    p2_arr = [[p2 for p2 in p2_range] for p1 in p1_range]
    fig.colorbar(ax.contour(p1_arr, p2_arr, Z, zorder=-1), ax=ax)

    def modified_FD(builder, axis_freq, overlap_L):
        Gamma_1 = builder.DT * builder.D1 / h**2
        Gamma_2 = builder.DT * builder.D2 / h**2
        # Finite differences:
        d1 = 1/12
        d2 = 1/360

        s_c = 1j*axis_freq # BE_s(dt, axis_freq)
        s_modified1 = s_c - d1 * dt/Gamma_1 * (s_c + setting.R)**2
        s_modified2 = s_c - d1 * dt/Gamma_2 * (s_c + setting.R)**2
        return np.abs(rho_s_c(builder, s_modified1, s_modified2, overlap_L=overlap_L,
            continuous_interface_op=False))

    cont = optimal_robin_parameter(setting, rho_c_c, axis_freq, (0.1, -0.1), continuous_interface_op=False)
    s_d_space = optimal_robin_parameter(setting, rho_c_FD, axis_freq, (0.1, -0.1))
    modified = optimal_robin_parameter(setting, modified_FD, axis_freq, (0.1, -0.1), overlap_L=0)
    ax.scatter(*cont.x, marker=symb_cont, alpha=1., s=size_symb, c=col_cont_discop,
            label="Continuous (disc. op.): {:.3f}".format(callrho(rho_c_FD,
                cont.x, overlap_M=overlap_M)))
    ax.scatter(*s_d_space.x, marker=symb_sdspace, alpha=1., s=size_symb, c=col_sdspace,
            label="S-d space: {:.3f}".format(s_d_space.fun))
    ax.scatter(*modified.x, marker=symb_modified, alpha=1., s=size_symb, edgecolors=col_modified, facecolors="none", linewidth=2.,
            label="Modified: {:.3f}".format(callrho(rho_c_FD,
                modified.x, overlap_M=overlap_M)))
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.legend()
    ax.set_title("Modified without overlap")

    overlap_M = 1
    setting.D1 = setting.D2
    ax = axes[0,1]
    p1min, p1max = 0.03, 0.3
    p2min, p2max = -.3, -.03
    p1_range = np.arange(p1min, p1max, (p1max - p1min)/res_x)
    p2_range = np.arange(p2min, p2max, (p2max - p2min)/res_x)
    Z = [[discrete_robin(setting, rho_Pade_FD_corr0, axis_freq, p1, p2, overlap_M=overlap_M)
            for p2 in p2_range] for p1 in p1_range]
    p1_arr = [[p1 for p2 in p2_range] for p1 in p1_range]
    p2_arr = [[p2 for p2 in p2_range] for p1 in p1_range]
    fig.colorbar(ax.contour(p1_arr, p2_arr, Z, zorder=-1), ax=ax)

    cont = optimal_robin_parameter(setting, rho_c_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)
    s_d_space = optimal_robin_parameter(setting, rho_c_FD, axis_freq, (0.1, -0.1), overlap_M=overlap_M)
    s_d_time = optimal_robin_parameter(setting, rho_Pade_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)
    discrete = optimal_robin_parameter(setting, rho_Pade_FD_corr0, axis_freq, (0.1, -0.1), overlap_M=overlap_M)
    combined = optimal_robin_parameter(setting, combined_Pade, axis_freq, (0.1, -0.1), overlap_M=overlap_M)
    ax.scatter(*cont.x, marker=symb_cont, alpha=1., s=size_symb, c=col_cont,
            label="Continuous: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                cont.x, overlap_M=overlap_M)))
    ax.scatter(*s_d_space.x, marker=symb_sdspace, alpha=1., s=size_symb, c=col_sdspace,
            label="S-d space: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                s_d_space.x, overlap_M=overlap_M)))
    ax.scatter(*s_d_time.x, marker=symb_sdtime, alpha=1., s=size_symb, c=col_sdtime,
            label="S-d time: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                s_d_time.x, overlap_M=overlap_M)))
    ax.scatter(*discrete.x, marker=symb_discrete, alpha=1., s=size_symb, c=col_discrete,
            label="Discrete: {:.3f}".format(discrete.fun))
    ax.scatter(*combined.x, marker=symb_combined, alpha=1., s=size_symb,
            edgecolors=col_combined, facecolors="none", linewidth=2.,
            label="Combined: {:.3f}".format(callrho(rho_Pade_FD_corr0,
                combined.x, overlap_M=overlap_M)))
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.legend()
    ax.set_title("Combined with overlap")



    ax = axes[1,1]
    p1min, p1max = 0.03, 0.3
    p2min, p2max = -.2, -.03
    p1_range = np.arange(p1min, p1max, (p1max - p1min)/res_x)
    p2_range = np.arange(p2min, p2max, (p2max - p2min)/res_x)
    Z = [[discrete_robin(setting, rho_c_FD, axis_freq, p1, p2, overlap_M=1)
            for p2 in p2_range] for p1 in p1_range]
    p1_arr = [[p1 for p2 in p2_range] for p1 in p1_range]
    p2_arr = [[p2 for p2 in p2_range] for p1 in p1_range]
    fig.colorbar(ax.contour(p1_arr, p2_arr, Z, zorder=-1), ax=ax)
    cont = optimal_robin_parameter(setting, rho_c_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h, continuous_interface_op=False)
    s_d_space = optimal_robin_parameter(setting, rho_c_FD, axis_freq, (0.1, -0.1), overlap_M=overlap_M)
    modified = optimal_robin_parameter(setting, modified_FD, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)
    ax.scatter(*cont.x, marker=symb_cont, alpha=1., s=size_symb, c=col_cont_discop,
            label="Continuous (disc. op.): {:.3f}".format(callrho(rho_c_FD,
                cont.x, overlap_M=overlap_M)))
    ax.scatter(*s_d_space.x, marker=symb_sdspace, alpha=1., s=size_symb, c=col_sdspace,
            label="S-d space: {:.3f}".format(s_d_space.fun))
    ax.scatter(*modified.x, marker=symb_modified, alpha=1., s=size_symb, edgecolors=col_modified, facecolors="none", linewidth=2.,
            label="Modified: {:.3f}".format(callrho(rho_c_FD,
                modified.x, overlap_M=overlap_M)))
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.legend(loc="lower right")
    ax.set_title("Modified with overlap")

    show_or_save("fig_RobinTwoSided")

def fig_dependency_maxrho_combined():
    setting = Builder()
    N = 10000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[1,1])
    lw_important = 2.1

    from cv_factor_onestep import rho_c_FD, rho_c_c, DNWR_c_c, DNWR_c_FD
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD_corr0, DNWR_Pade_c, DNWR_Pade_FD

    def combined_Pade(setting, overlap_M):
        combined = - rho_c_c(setting, axis_freq, overlap_L=overlap_M*h) \
                    + rho_Pade_c(setting, axis_freq, overlap_L=overlap_M*h) \
                    + rho_c_FD(setting, axis_freq, overlap_M=overlap_M)
        return np.abs(combined)
    def combined_Pade_DNWR(setting, axis_freq, theta):
        builder = setting.copy()
        combined = - DNWR_c_c(builder, axis_freq, theta=theta) \
                    + DNWR_Pade_c(builder, axis_freq, theta=theta) \
                    + DNWR_c_FD(builder, axis_freq, theta=theta)
        return np.abs(combined)

    ax = axes[0,1]
    all_theta = np.linspace(0.5,0.68,300)

    semidiscrete_space = [np.max(np.abs(DNWR_c_FD(setting, axis_freq, theta))) for theta in all_theta]
    semidiscrete_time = [np.max(np.abs(DNWR_Pade_c(setting, axis_freq, theta))) for theta in all_theta]
    discrete= [np.max(np.abs(DNWR_Pade_FD(setting, axis_freq, theta))) for theta in all_theta]
    continuous = [np.max(np.abs(DNWR_c_c(setting, axis_freq, theta))) for theta in all_theta]
    combined = [np.max(np.abs(combined_Pade_DNWR(setting, axis_freq, theta))) for theta in all_theta]
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

    from cv_factor_onestep import rho_c_FD, rho_c_c, rho_s_c
    def maxrho(func, p, **kwargs):
        builder = setting.copy()
        builder.LAMBDA_1, builder.LAMBDA_2 = p, -p
        return np.max(np.abs(func(builder, **kwargs)))

    ax = axes[0, 0]
    all_p1 = np.linspace(0.05,.22,300)
    discrete = [maxrho(rho_Pade_FD_corr0, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    semidiscrete_time = [maxrho(rho_Pade_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    semidiscrete_space = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    continuous = [maxrho(rho_c_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    combined = [maxrho(combined_Pade, p, overlap_M=overlap_M) for p in all_p1]

    ax.plot(all_p1, continuous, lw=lw_important, color=col_cont)
    ax.plot(all_p1, discrete, lw=lw_important, color=col_discrete)
    ax.plot(all_p1, combined, "--", lw=lw_important, color=col_combined)
    ax.plot(all_p1, semidiscrete_space, "--", color=col_sdspace)
    ax.plot(all_p1, semidiscrete_time, "--", color=col_sdtime)
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
    ax.set_title(r"$RR^{M=0}$")

    ax = axes[1, 0]
    overlap_M = 1
    setting.D1 = setting.D2

    discrete = [maxrho(rho_Pade_FD_corr0, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    semidiscrete_time = [maxrho(rho_Pade_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    semidiscrete_space = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    continuous = [maxrho(rho_c_c, p, w=axis_freq, overlap_L=overlap_M*h) for p in all_p1]
    combined = [maxrho(combined_Pade, p, overlap_M=overlap_M) for p in all_p1]
    ax.plot(all_p1, continuous, label="Continuous", lw=lw_important, color=col_cont)
    ax.plot(all_p1, discrete, label="Discrete", lw=lw_important, color=col_discrete)
    ax.plot(all_p1, combined, "--", label="Combined", lw=lw_important, color=col_combined)
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
    ax.set_title(r"$RR^{M=1}$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.6, 0.4))
    show_or_save("fig_dependency_maxrho_combined")

def fig_dependency_maxrho_modified():
    setting = Builder()
    N = 10000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(right=0.97, hspace=0.65, wspace=0.28)
    fig.delaxes(ax= axes[1,1])
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
    ax = axes[0,1]
    all_theta = np.linspace(0.6,0.64,300)
    discrete = np.array([np.max(np.abs(DNWR_c_FD(setting, axis_freq, theta=theta)))
        for theta in all_theta])
    continuous = [np.max(np.abs(DNWR_s_c(setting, s_c, s_c,
        theta=theta, continuous_interface_op=False))) for theta in all_theta]
    modified_in_space = [np.max(np.abs(DNWR_s_c(setting, s_modified1, s_modified2,
        theta=theta, continuous_interface_op=False))) for theta in all_theta]
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

    from cv_factor_onestep import rho_c_FD, rho_c_c, rho_s_c
    ax = axes[0, 0]
    all_p1 = np.linspace(0.12,.14,300)
    def maxrho(func, p, **kwargs):
        builder = setting.copy()
        builder.LAMBDA_1, builder.LAMBDA_2 = p, -p
        return np.max(np.abs(func(builder, **kwargs)))

    discrete = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    continuous = [maxrho(rho_s_c, p, s_1=s_c, s_2=s_c,
        overlap_L=overlap_M*h, continuous_interface_op=False) for p in all_p1]
    modified_in_space = [maxrho(rho_s_c, p, s_1=s_modified1, s_2=s_modified2,
        overlap_L=overlap_M*h, continuous_interface_op=False) for p in all_p1]
    ax.plot(all_p1, continuous, lw=lw_important, color=col_cont_discop)
    ax.plot(all_p1, discrete, lw=lw_important, color=col_sdspace)
    ax.plot(all_p1, modified_in_space, "--", lw=lw_important, color=col_modified)
    minima_indices = [np.argmin(discrete), np.argmin(continuous), np.argmin(modified_in_space)]
    col_minimas = [col_sdspace, col_cont_discop, col_modified]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=all_p1[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # xmin, xmax = ax.get_xlim()[0], all_p1[minima_indices]
    # ax.hlines(y=np.array(discrete)[minima_indices], xmin=xmin, xmax=xmax,
    #         colors=col_minimas, linestyle='dashed')
    ax.set_xlabel(r"$p_1 = -p_2$")
    ax.set_ylabel(r"$\max_\omega (\rho)$")
    ax.set_title(r"$RR^{M=0}$")

    ax = axes[1, 0]
    all_p1 = np.linspace(0.09,.1,300)
    overlap_M = 1
    setting.D1 = setting.D2

    discrete = [maxrho(rho_c_FD, p, w=axis_freq, overlap_M=overlap_M) for p in all_p1]
    continuous = [maxrho(rho_s_c, p, s_1=s_c, s_2=s_c,
        overlap_L=overlap_M*h, continuous_interface_op=False) for p in all_p1]
    modified_in_space = [maxrho(rho_s_c, p, s_1=s_modified1, s_2=s_modified2,
        overlap_L=overlap_M*h, continuous_interface_op=False) for p in all_p1]
    ax.plot(all_p1, continuous, label="Continuous (disc. op.)", lw=lw_important, color=col_cont_discop)
    ax.plot(all_p1, discrete, label="S-d space", lw=lw_important, color=col_sdspace)
    ax.plot(all_p1, modified_in_space, "--", label="Modified", lw=lw_important, color=col_modified)
    minima_indices = [np.argmin(discrete), np.argmin(continuous), np.argmin(modified_in_space)]
    col_minimas = [col_sdspace, col_cont_discop, col_modified]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=all_p1[minima_indices], ymin=ymin, ymax=ymax, colors=col_minimas, linestyle="dotted")
    # xmin, xmax = ax.get_xlim()[0], all_p1[minima_indices]
    # ax.hlines(y=np.array(discrete)[minima_indices], xmin=xmin, xmax=xmax,
    #         colors=col_minimas, linestyle='dashed')
    ax.set_xlabel(r"$p_1 = -p_2$")
    ax.set_ylabel(r"$\max_\omega (\rho)$")
    ax.set_title(r"$RR^{M=1}$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.6, 0.4))
    show_or_save("fig_dependency_maxrho_modified")

def optimal_DNWR_parameter(builder, func, w):
    from scipy.optimize import minimize_scalar
    def to_optimize(x0):
        return np.max(np.abs(func(builder, w, x0)))
    optimal_lam = minimize_scalar(to_optimize)
    return optimal_lam.x

def optimal_robin_parameter(builder, func, w, x0, **kwargs):
    from scipy.optimize import minimize_scalar, minimize
    def to_optimize(x0):
        setting = builder.copy()
        setting.LAMBDA_1 = x0[0]
        setting.LAMBDA_2 = x0[1]
        return np.max(np.abs(func(setting, w, **kwargs)))
    optimal_lam = minimize(method='Nelder-Mead', fun=to_optimize, x0=x0)
    return optimal_lam.x

def fig_modif_time():
    from cv_factor_onestep import rho_s_c, rho_c_c
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD_corr0
    setting = Builder()
    N = 10000
    overlap_M = 0
    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_Pade_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)

    fig, axes = plt.subplots(2, 2)
    fig.delaxes(ax= axes[1,1])
    fig.subplots_adjust(right=0.97, hspace=0.65)
    ax = axes[0,0]

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
    ax.set_title(r"$RR^{M=0}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    ax = axes[0,1]
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

    ax = axes[1,0]
    overlap_M = 1
    setting.D1 = setting.D2

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_Pade_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)

    discrete = np.abs(rho_Pade_c(setting, axis_freq, overlap_L=overlap_M*h))
    continuous = np.abs(rho_c_c(setting, axis_freq, overlap_L=overlap_M*h))
    modified_in_time = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h))
    ax.semilogx(axis_freq, continuous, color=col_cont)
    ax.semilogx(axis_freq, discrete, color=col_sdtime)
    ax.semilogx(axis_freq, modified_in_time, "--", color=col_modified)
    ax.set_title(r"$RR^{M=1}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.6, 0.4))
    show_or_save("fig_modif_time")


def fig_modif_space():
    from cv_factor_onestep import rho_c_FD, rho_s_c
    setting = Builder()
    setting.M1 = setting.M2 = 21 # warning, we change the parameter to highlight the differences
    N = 10000
    overlap_M = 0

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_FD, axis_freq, (0.1, -0.1), overlap_M=overlap_M)

    fig, axes = plt.subplots(2, 2)
    fig.delaxes(ax= axes[1,1])
    fig.subplots_adjust(right=0.97, hspace=0.65)
    ax = axes[0,0]
    ax.semilogx(axis_freq, np.abs(rho_s_c(setting, 1j*axis_freq, 1j*axis_freq,
        overlap_L=overlap_M*h, continuous_interface_op=False)),
        color=col_cont_discop)

    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

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

    modified_in_space = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h, continuous_interface_op=False))
    ax.semilogx(axis_freq, np.abs(rho_c_FD(setting, axis_freq, overlap_M=overlap_M)), color=col_sdspace)
    ax.semilogx(axis_freq, modified_in_space, "--", color=col_modified)

    ax.set_title(r"$RR^{M=0}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    from cv_factor_onestep import DNWR_s_c, DNWR_c_FD
    ax = axes[0,1]
    overlap_M = 0

    theta = optimal_DNWR_parameter(setting, DNWR_c_FD, axis_freq)

    ax.semilogx(axis_freq, np.abs(DNWR_s_c(setting, 1j*axis_freq, 1j*axis_freq, theta, continuous_interface_op=False)), label="Continuous (disc. op.)", color=col_cont_discop)
    modified_in_space = np.abs(DNWR_s_c(setting, s_modified1, s_modified2, theta=theta, continuous_interface_op=False))
    ax.semilogx(axis_freq, np.abs(DNWR_c_FD(setting, axis_freq, theta=theta)), label="Semi-discrete in space", color=col_sdspace)
    ax.semilogx(axis_freq, modified_in_space, "--", label="Modified in space", color=col_modified)

    ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    ax = axes[1, 0]
    overlap_M = 1
    setting.D1 = setting.D2

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_FD, axis_freq, (0.1, -0.1), overlap_M=overlap_M)

    ax.semilogx(axis_freq, np.abs(rho_s_c(setting, 1j*axis_freq, 1j*axis_freq,
        overlap_L=overlap_M*h, continuous_interface_op=False)),
        color=col_cont_discop)

    modified_in_space = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h, continuous_interface_op=False))
    ax.semilogx(axis_freq, np.abs(rho_c_FD(setting, axis_freq, overlap_M=overlap_M)), color=col_sdspace)
    ax.semilogx(axis_freq, modified_in_space, "--", color=col_modified)

    ax.set_title(r"$RR^{M=1}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    fig.legend(loc="upper left", bbox_to_anchor=(0.6, 0.4))
    show_or_save("fig_modif_space")

def fig_combinedRate():
    setting = Builder()
    N = 10000
    w = get_discrete_freq(N, setting.DT)[int(N//2)+1:]
    overlap_M=0
    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)

    fig, axes = plt.subplots(2, 2)
    fig.delaxes(ax= axes[1,1])
    fig.subplots_adjust(right=0.97, hspace=0.65)
    ax = axes[0,0]

    from cv_factor_onestep import rho_c_FD, rho_c_c, DNWR_c_c, DNWR_c_FD
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD_corr0, DNWR_Pade_c, DNWR_Pade_FD
    def combined_Pade(builder, w, overlap_M):
        combined = - rho_c_c(builder, w, overlap_L=overlap_M*h) \
                    + rho_Pade_c(builder, w, overlap_L=overlap_M*h) \
                    + rho_c_FD(builder, w, overlap_M=overlap_M)
        return combined

    def to_minimize_Pade(LAMBDAS, overlap_M):
        builder = setting.copy()
        builder.LAMBDA_1 = LAMBDAS[0]
        builder.LAMBDA_2 = LAMBDAS[1]
        return np.max(np.abs(rho_Pade_FD_corr0(builder, w, overlap_M=overlap_M)))

    def to_minimize_combined(LAMBDAS, overlap_M):
        builder = setting.copy()
        builder.LAMBDA_1 = LAMBDAS[0]
        builder.LAMBDA_2 = LAMBDAS[1]
        return np.max(np.abs(combined_Pade(builder, w, overlap_M=overlap_M)))

    from scipy.optimize import minimize
    # ret = minimize(method='Nelder-Mead', fun=to_minimize_combined, x0=np.array((0.15, -0.15)), args=overlap_M)
    ret = minimize(method='Nelder-Mead', fun=to_minimize_Pade, x0=np.array((0.15, -0.15)), args=overlap_M)

    setting.LAMBDA_1 = ret.x[0]
    setting.LAMBDA_2 = ret.x[1]

    ax.semilogx(w, np.abs(rho_Pade_FD_corr0(setting, w, overlap_M=overlap_M)), color=col_discrete)
    ax.semilogx(w, np.abs(combined_Pade(setting, w, overlap_M=overlap_M)), color=col_combined)
    ax.semilogx(w, np.abs(rho_c_FD(setting, w, overlap_M=overlap_M)), "--", dashes=[7,9], color=col_sdspace)
    ax.semilogx(w, np.abs(rho_c_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[3,5], color=col_cont)
    ax.semilogx(w, np.abs(rho_Pade_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[7,9], color=col_sdtime)

    ax.set_xlabel(r"$\omega \Delta t$")
    ax.set_ylabel(r"$\rho$")
    ax.set_ylim(bottom=0., top=0.5) # all axis are shared
    ax.set_title(r"$RR^{M=0}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))

    ax = axes[0,1]

    def combined_Pade_DNWR(builder, w, theta):
        combined = - DNWR_c_c(builder, w, theta=theta) \
                    + DNWR_Pade_c(builder, w, theta=theta) \
                    + DNWR_c_FD(builder, w, theta=theta)
        return combined

    def to_minimize_Pade_DNWR(theta):
        return np.max(np.abs(DNWR_Pade_FD(setting, w, theta=theta)))

    def to_minimize_combined_DNWR(theta):
        return np.max(np.abs(combined_Pade_DNWR(setting, w, theta=theta)))

    from scipy.optimize import minimize_scalar
    theta = minimize_scalar(fun=to_minimize_Pade_DNWR).x

    ax.semilogx(w, np.abs(DNWR_Pade_FD(setting, w, theta)), label=r"$\rho^{\rm (Pade, FD)}$", color=col_discrete)
    ax.semilogx(w, np.abs(combined_Pade_DNWR(setting, w, theta)), label=r"$\rho^{\rm (Pade, FD)}_{\rm combined}$", color=col_combined)
    ax.semilogx(w, np.abs(DNWR_c_FD(setting, w, theta)), "--", label=r"$\rho^{\rm (c, FD)}$", dashes=[7,9], color=col_sdspace)
    ax.semilogx(w, np.abs(DNWR_c_c(setting, w, theta)), "--", label=r"$\rho^{\rm (c, c)}$", dashes=[3,5], color=col_cont)
    ax.semilogx(w, np.abs(DNWR_Pade_c(setting, w, theta)), "--", label=r"$\rho^{\rm (Pade, c)}$", dashes=[7,9], color=col_sdtime)
    ax.set_xlabel(r"$\omega \Delta t$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("DNWR, " + (r"$\theta={:.3f}$").format(theta))

    ax = axes[1, 0]
    overlap_M = 1
    setting.D1 = setting.D2
    ret = minimize(method='Nelder-Mead', fun=to_minimize_Pade, x0=np.array((0.15, -0.15)), args=overlap_M)

    setting.LAMBDA_1 = ret.x[0]
    setting.LAMBDA_2 = ret.x[1]

    ax.semilogx(w, np.abs(rho_Pade_FD_corr0(setting, w, overlap_M=overlap_M)), color=col_discrete)
    ax.semilogx(w, np.abs(combined_Pade(setting, w, overlap_M=overlap_M)), color=col_combined)
    ax.semilogx(w, np.abs(rho_c_FD(setting, w, overlap_M=overlap_M)), "--", dashes=[7,9], color=col_sdspace)
    ax.semilogx(w, np.abs(rho_c_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[3,5], color=col_cont)
    ax.semilogx(w, np.abs(rho_Pade_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[7,9], color=col_sdtime)
    ax.set_xlabel(r"$\omega \Delta t$")
    ax.set_ylabel(r"$\rho$")
    ax.set_ylim(bottom=0.) # all axis are shared
    ax.set_title(r"$RR^{M=1}, " + ("(p_1, p_2) = ({:.3f}, {:.3f})$").format(setting.LAMBDA_1, setting.LAMBDA_2))

    fig.legend(loc="upper left", bbox_to_anchor=(0.65, 0.45))
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
    from cv_factor_pade import rho_Pade_c, rho_Pade_FV, rho_Pade_FD_corr0, DNWR_Pade_c, DNWR_Pade_FD

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
    from cv_factor_pade import rho_Pade_c, rho_Pade_FV, rho_Pade_FD_corr0

    ax = axes[1]
    ocean, atmosphere = setting.build(OceanPadeFD, AtmospherePadeFD)
    alpha_w_overlap = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", overlap=1)
    ax.semilogx(axis_freq, np.abs(alpha_w_overlap[2]/alpha_w_overlap[1]))
    alpha_w_overlap = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", overlap=2)
    ax.semilogx(axis_freq, np.abs(alpha_w_overlap[2]/alpha_w_overlap[1]))

    ax.semilogx(axis_freq, np.abs(rho_Pade_FD_corr0(setting, axis_freq, overlap_M=2)), "--", label="M=2")
    ax.semilogx(axis_freq, np.abs(rho_Pade_FD_corr0(setting, axis_freq, overlap_M=1)), "--", label="M=1")

    setting.D1 = 0.5
    ocean.nu = 0.5
    alpha_w = frequency_simulation(atmosphere, ocean, number_samples=4, NUMBER_IT=1,
            laplace_real_part=0, T=N*setting.DT, init="white", overlap=0)
    ax.semilogx(axis_freq, np.abs(alpha_w[2]/alpha_w[1]))
    ax.semilogx(axis_freq, np.abs(rho_Pade_FD_corr0(setting, axis_freq, overlap_M=0)), "--", label=r"M=0, $\nu_1 \neq \nu_2$")

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
        self.SIZE_DOMAIN_1 = 200
        self.SIZE_DOMAIN_2 = 200
        self.M1 = 201 # to have h=1 the number of points M_j must be 101
        self.M2 = 201
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

    def build(self, ocean_discretisation, atm_discretisation):
        """ build the models and returns tuple (ocean_model, atmosphere_model)"""
        ocean = ocean_discretisation(r=self.R, nu=self.D1, LAMBDA=self.LAMBDA_1,
            M=self.M1, SIZE_DOMAIN=self.SIZE_DOMAIN_1, DT=self.DT)
        atmosphere = atm_discretisation(r=self.R, nu=self.D2, LAMBDA=self.LAMBDA_2,
            M=self.M2, SIZE_DOMAIN=self.SIZE_DOMAIN_2, DT=self.DT)
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
