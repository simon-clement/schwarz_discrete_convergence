#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import numpy as np
from memoisation import memoised
import matplotlib.pyplot as plt
from simulator import frequency_simulation

REAL_FIG = True

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
    setting.M1 = 10
    setting.M2 = 10
    setting.SIZE_DOMAIN_1 = 100
    setting.SIZE_DOMAIN_2 = 100
    setting.R = 1e-3
    setting.DT = 100.
    N = 1000
    overlap_M = 0
    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    if overlap_M > 0:
        setting.D1 = setting.D2
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_Pade_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)

    fig, axes = plt.subplots(1, 3, figsize=[6.4*1.5, 2.4])
    fig.subplots_adjust(right=0.80,wspace=0.35, left=0.09, bottom=0.35)
    ax = axes[0]

    assert abs(h - setting.SIZE_DOMAIN_2 / (setting.M2 - 1)) < 1e-10

    dt = setting.DT

    # Finite differences:
    s_c = 1j*axis_freq # BE_s(dt, axis_freq)
    s_modified1 = s_c - (4 + 3*np.sqrt(2)) * dt**2/6 * 1j * axis_freq**3
    s_modified2 = s_c - (4 + 3*np.sqrt(2)) * dt**2/6 * 1j * axis_freq**3

    discrete = np.abs(rho_Pade_c(setting, axis_freq, overlap_L=overlap_M*h))
    continuous = np.abs(rho_c_c(setting, axis_freq, overlap_L=overlap_M*h))
    modified_in_space = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h))
    ax.semilogx(axis_freq, continuous, "k")
    ax.semilogx(axis_freq, discrete, "r")
    ax.semilogx(axis_freq, modified_in_space, "g--")
    ax.set_title("RR Without overlap")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")


    overlap_M = 1
    if overlap_M > 0:
        setting.D1 = setting.D2
    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_Pade_c, axis_freq, (0.1, -0.1), overlap_L=overlap_M*h)

    ax = axes[1]
    # Finite differences:
    discrete = np.abs(rho_Pade_c(setting, axis_freq, overlap_L=overlap_M*h))
    continuous = np.abs(rho_c_c(setting, axis_freq, overlap_L=overlap_M*h))
    modified_in_space = np.abs(rho_s_c(setting, s_modified1, s_modified2, overlap_L=overlap_M*h))
    ax.semilogx(axis_freq, continuous, "k")
    ax.semilogx(axis_freq, discrete, "r")
    ax.semilogx(axis_freq, modified_in_space, "g--")
    ax.set_title("RR With overlap")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")

    overlap_M = 0
    setting.D1 = 0.5
    from cv_factor_pade import DNWR_Pade_c
    from cv_factor_onestep import DNWR_c_c, DNWR_s_c
    theta = optimal_DNWR_parameter(setting, DNWR_Pade_c, axis_freq)

    ax = axes[2]
    # Finite differences:
    discrete = np.abs(DNWR_Pade_c(setting, axis_freq, theta=theta))
    continuous = np.abs(DNWR_c_c(setting, axis_freq, theta=theta))
    modified_in_space = np.abs(DNWR_s_c(setting, s_modified1, s_modified2, theta=theta))
    ax.semilogx(axis_freq, continuous, "k", label="Continuous")
    ax.semilogx(axis_freq, discrete, "r", label="Semi-Discrete in time")
    ax.semilogx(axis_freq, modified_in_space, "g--", label="Modified in time")
    ax.set_title("DNWR")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")
    fig.legend()

    show_or_save("fig_modif_time")


def fig_modif_space():
    from cv_factor_onestep import rho_c_FD, rho_s_c
    setting = Builder()
    setting.DT = 100.
    setting.R = 1e-3
    setting.M1 = 10
    setting.M2 = 10
    setting.SIZE_DOMAIN_1 = 100
    setting.SIZE_DOMAIN_2 = 100
    N = 10000
    overlap_M = 0
    if overlap_M > 0:
        setting.D1 = setting.D2

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)
    axis_freq = get_discrete_freq(N, setting.DT)[int(N//2)+1:]

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_FD, axis_freq, (0.1, -0.1), overlap_M=overlap_M)

    fig, axes = plt.subplots(1, 3, figsize=[6.4*1.5, 2.4])
    fig.subplots_adjust(right=0.80,wspace=0.35, left=0.09, bottom=0.35)
    ax = axes[0]
    ax.semilogx(axis_freq, np.abs(rho_s_c(setting, 1j*axis_freq, 1j*axis_freq,
        overlap_L=overlap_M*h, continuous_interface_op=False)),
        "k")

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
    ax.semilogx(axis_freq, np.abs(rho_c_FD(setting, axis_freq, overlap_M=overlap_M)), "r")
    ax.semilogx(axis_freq, modified_in_space, "g--")

    ax.set_title("RR Without overlap")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho$")


    ax = axes[1]
    setting = Builder()
    setting.R = 1e-3
    setting.M1 = 10
    setting.M2 = 10
    setting.SIZE_DOMAIN_1 = 100
    setting.SIZE_DOMAIN_2 = 100
    overlap_M = 1
    if overlap_M > 0:
        setting.D1 = setting.D2

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)

    setting.LAMBDA_1, setting.LAMBDA_2 = optimal_robin_parameter(setting,
            rho_c_FD, axis_freq, (0.1, -0.1), overlap_M=overlap_M)

    ax.semilogx(axis_freq, np.abs(rho_s_c(setting, 1j*axis_freq, 1j*axis_freq,
        overlap_L=overlap_M*h, continuous_interface_op=False)),
        "k")

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
    ax.semilogx(axis_freq, np.abs(rho_c_FD(setting, axis_freq, overlap_M=overlap_M)), "r")
    ax.semilogx(axis_freq, modified_in_space, "g--")

    ax.set_title("RR With overlap")
    ax.set_xlabel(r"$\omega$")

    from cv_factor_onestep import DNWR_s_c, DNWR_c_FD
    ax = axes[2]
    setting = Builder()
    setting.R = 1e-3
    setting.M1 = 10
    setting.M2 = 10
    setting.SIZE_DOMAIN_1 = 100
    setting.SIZE_DOMAIN_2 = 100
    overlap_M = 0
    if overlap_M > 0:
        setting.D1 = setting.D2

    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)

    theta = optimal_DNWR_parameter(setting, DNWR_c_FD, axis_freq)

    ax.semilogx(axis_freq, np.abs(DNWR_s_c(setting, 1j*axis_freq, 1j*axis_freq, theta, continuous_interface_op=False)), "k", label="Continuous")

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

    modified_in_space = np.abs(DNWR_s_c(setting, s_modified1, s_modified2, theta=theta, continuous_interface_op=False))
    ax.semilogx(axis_freq, np.abs(DNWR_c_FD(setting, axis_freq, theta=theta)), "r", label="Semi-discrete in space")
    ax.semilogx(axis_freq, modified_in_space, "g--", label="Modified in space")

    ax.set_title("DNWR")
    ax.set_xlabel(r"$\omega$")
    fig.legend()
    show_or_save("fig_modif_space")

def fig_combinedRate():
    setting = Builder()
    setting.M1 = 100
    setting.M2 = 100
    setting.SIZE_DOMAIN_1 = 200
    setting.SIZE_DOMAIN_2 = 200
    setting.D1 = .5
    setting.D2 = 1.
    setting.R = 1e-3
    setting.DT = 100.
    N = 3000
    setting.LAMBDA_1 = 0.14
    setting.LAMBDA_2 = -0.5
    w = get_discrete_freq(N, setting.DT)[int(N//2)+1:]
    overlap_M=0
    h = setting.SIZE_DOMAIN_1 / (setting.M1 - 1)

    fig, axes = plt.subplots(1, 3, figsize=[6.4*1.5, 2.4], sharex=True, sharey=True)
    # fig, axes = plt.subplots(1, 3, figsize=[6.4*1.5*2, 2.4*4])
    fig.subplots_adjust(right=0.80,wspace=0.35, left=0.09, bottom=0.35)
    ax = axes[0]

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

    ax.semilogx(w, np.abs(rho_Pade_FD_corr0(setting, w, overlap_M=overlap_M)), "k")
    ax.semilogx(w, np.abs(combined_Pade(setting, w, overlap_M=overlap_M)), "r")
    ax.semilogx(w, np.abs(rho_c_FD(setting, w, overlap_M=overlap_M)), "--", dashes=[7,9])
    ax.semilogx(w, np.abs(rho_c_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[3,5])
    ax.semilogx(w, np.abs(rho_Pade_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[7,9])

    ax.set_xlabel(r"$\omega \Delta t$")
    ax.set_ylabel(r"$\rho$")
    ax.set_ylim(top=0.15, bottom=0.) # all axis are shared
    ax.set_title("RR without overlap")

    ax = axes[1]
    overlap_M = 1
    setting.D1 = setting.D2
    ret = minimize(method='Nelder-Mead', fun=to_minimize_Pade, x0=np.array((0.15, -0.15)), args=overlap_M)

    setting.LAMBDA_1 = ret.x[0]
    setting.LAMBDA_2 = ret.x[1]

    ax.semilogx(w, np.abs(rho_Pade_FD_corr0(setting, w, overlap_M=overlap_M)), "k")
    ax.semilogx(w, np.abs(combined_Pade(setting, w, overlap_M=overlap_M)), "r")
    ax.semilogx(w, np.abs(rho_c_FD(setting, w, overlap_M=overlap_M)), "--", dashes=[7,9])
    ax.semilogx(w, np.abs(rho_c_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[3,5])
    ax.semilogx(w, np.abs(rho_Pade_c(setting, w, overlap_L=overlap_M*h)), "--", dashes=[7,9])
    ax.set_xlabel(r"$\omega \Delta t$")
    ax.set_ylim(top=0.15, bottom=0.) # all axis are shared
    ax.set_title("RR with overlap")

    ax = axes[2]
    overlap_M = 0
    setting.D1 = .5


    # setting.M1 = 100
    # setting.M2 = 100
    # setting.D1 = .5
    # setting.D2 = 1.
    # setting.R = 1e-3
    # setting.DT = 100.
    # N = 3000

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

    ax.semilogx(w, np.abs(DNWR_Pade_FD(setting, w, theta)), "k", label=r"$\rho^{\rm (Pade, FD)}$")
    ax.semilogx(w, np.abs(combined_Pade_DNWR(setting, w, theta)), "r", label=r"$\rho^{\rm (Pade, FD)}_{\rm combined}$")
    ax.semilogx(w, np.abs(DNWR_c_FD(setting, w, theta)), "--", label=r"$\rho^{\rm (c, FD)}$", dashes=[7,9])
    ax.semilogx(w, np.abs(DNWR_c_c(setting, w, theta)), "--", label=r"$\rho^{\rm (c, c)}$", dashes=[3,5])
    ax.semilogx(w, np.abs(DNWR_Pade_c(setting, w, theta)), "--", label=r"$\rho^{\rm (Pade, c)}$", dashes=[7,9])
    ax.set_xlabel(r"$\omega \Delta t$")


    ax.set_title("DNWR")
    fig.legend()
    show_or_save("fig_combinedRate")

def fig_validate_DNWR():
    from ocean_models.ocean_BE_FD import OceanBEFD
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
    from cv_factor_onestep import rho_c_c, rho_BE_c, rho_BE_FV, rho_BE_FD, rho_c_FV, rho_c_FD, DNWR_BE_FD
    setting = Builder()
    setting.R = 1e-3
    N = 3000
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
    setting.R = 1e-3
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


def fig_first_guess_N():
    from ocean_models.ocean_BE_FD import OceanBEFD
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
    fig, axes = plt.subplots(1, 3, figsize=[6.4*1.4, 4.4], sharex=True, sharey=True)
    setting = Builder()
    setting.M1 = 100
    setting.SIZE_DOMAIN_1= 100
    setting.M2 = 100
    setting.SIZE_DOMAIN_2= 100
    setting.D1 = .5
    setting.D2 = 1.
    setting.R = 1e-3
    setting.DT = 1.
    setting.LAMBDA_1 = 1e9
    setting.LAMBDA_2 = 0.

    for N, ax in zip((10, 100, 1000), axes):
        axis_freq = get_discrete_freq(N, setting.DT)
        ocean, atmosphere = setting.build(OceanBEFD, AtmosphereBEFD)
        for init in ("GP", "white", "dirac"):
            alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=16, NUMBER_IT=1,
                    laplace_real_part=0, T=N*setting.DT, init=init)
            ax.semilogx(axis_freq, alpha_w[2]/alpha_w[1], label=init)
        ax.set_title(str(N) + "time steps")
        ax.set_xlabel(r"$\omega$")

    axes[0].set_ylabel("$\\widehat{\\rho}$")
    h, l = axes[1].get_legend_handles_labels()
    axes[1].legend(h[:3], l[:3])
    show_or_save("fig_first_guess_N")

def fig_first_guess_DT():
    from ocean_models.ocean_BE_FD import OceanBEFD
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
    fig, axes = plt.subplots(1, 3, figsize=[6.4*1.4, 4.4], sharex=True, sharey=True)
    setting = Builder()
    setting.M1 = 100
    setting.SIZE_DOMAIN_1= 100
    setting.M2 = 100
    setting.SIZE_DOMAIN_2= 100
    setting.D1 = .5
    setting.D2 = 1.
    setting.R = 1e-3
    setting.LAMBDA_1 = 1e9
    setting.LAMBDA_2 = 0.

    for dt, N, ax in zip((1., 10., 100.), (1000, 100, 10), axes):
        setting.DT = dt
        axis_freq = get_discrete_freq(N, setting.DT)
        ocean, atmosphere = setting.build(OceanBEFD, AtmosphereBEFD)
        for init in ("GP", "white", "dirac"):
            alpha_w = memoised(frequency_simulation, atmosphere, ocean, number_samples=32, NUMBER_IT=1,
                    laplace_real_part=0, T=N*setting.DT, init=init)
            ax.semilogx(axis_freq, alpha_w[2]/alpha_w[1], label=init)
        ax.set_title("Time steps of " + str(dt) + " s")
        ax.set_xlabel(r"$\omega$")

    axes[0].set_ylabel("$\\widehat{\\rho}$")
    # interface_info = schwarz_simulator(atmosphere, ocean, seed=1, T=N*setting.DT,
    #         NUMBER_IT=1, init="GP")
    # plt.plot(interface_info[0])
    h, l = axes[1].get_legend_handles_labels()
    axes[1].legend(h[:3], l[:3])
    show_or_save("fig_first_guess_DT")

def fig_plot_initialisation():
    from ocean_models.ocean_BE_FD import OceanBEFD
    from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD
    fig, ax = plt.subplots(1, 1, figsize=[6.4, 2.4])
    setting = Builder()
    setting.M1 = 100
    setting.SIZE_DOMAIN_1= 100
    setting.M2 = 100
    setting.SIZE_DOMAIN_2= 100
    setting.D1 = .5
    setting.D2 = 1.
    setting.R = 1e-3
    setting.DT = 10.
    setting.LAMBDA_1 = 1e9
    setting.LAMBDA_2 = 0.
    N = 500

    axis_freq = get_discrete_freq(N, setting.DT)
    ocean, atmosphere = setting.build(OceanBEFD, AtmosphereBEFD)
    for init, col in zip(("white", "dirac","GP"),("r--","g","b")):
        if init == "white":
            interface_ocean = ((np.concatenate(([0], 2 * (np.random.rand(N) - 0.5)))) + 1j*(np.concatenate(([0], 2 * (np.random.rand(N) - 0.5)))))
        elif init == "GP":
            cov = np.array([[ np.exp(-.1*np.abs(i-j)) for i in range(N)] for j in range(N)])
            rand1, rand2 = np.random.default_rng().multivariate_normal(np.zeros(N), cov, 2)
            interface_ocean = np.concatenate(([0], rand2))
        elif init == "dirac":
            interface_ocean = np.concatenate(([0, 2], np.zeros(N-1)))
        ax.plot(interface_ocean, col, label=init)
    ax.set_ylabel("first guess (error)")
    ax.set_xlabel("time step")
    ax.legend()
    show_or_save("fig_plot_initialisation")

def fig_optiRates():
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
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
    from cv_factor_onestep import rho_c_c, rho_BE_c, rho_BE_FD, rho_BE_FV

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
    from cv_factor_pade import rho_Pade_c, rho_Pade_FD_corr0, rho_Pade_FV

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

def optiRatesGeneral(axes, all_rates, all_ocean, all_atmosphere, name_method="Unknown discretization", caracs={}, **args_for_discretization):
    """
        Creates a figure comparing analysis methods for a discretization.
    """

    setting = Builder()
    setting.M1 = 100
    setting.SIZE_DOMAIN_1= 100
    setting.M2 = 100
    setting.SIZE_DOMAIN_2= 100
    setting.D1 = .5
    setting.D2 = 1.
    setting.R = 1e-3
    setting.DT = .5
    N = 1000000
    axis_freq = get_discrete_freq(N, setting.DT)

    axes.set_xlabel("$\\omega \\Delta t$")
    axes.set_ylabel(r"${\rho}_{RR}^{"+name_method+r"}$")

    def rate_onesided(lam):
        builder = setting.copy()
        builder.LAMBDA_1 = lam
        builder.LAMBDA_2 = -lam
        return np.max(np.abs(all_rates[0](builder, axis_freq)))

    from scipy.optimize import minimize_scalar, minimize
    optimal_lam = minimize_scalar(fun=rate_onesided)
    x0_opti = (optimal_lam.x, -optimal_lam.x)

    for discrete_factor, oce_class, atm_class, names in zip(all_rates,
            all_ocean, all_atmosphere, caracs):
        def rate_twosided(lam):
            builder = setting.copy()
            builder.LAMBDA_1 = lam[0]
            builder.LAMBDA_2 = lam[1]
            return np.max(np.abs(discrete_factor(builder, axis_freq)))

        optimal_lam = minimize(method='Nelder-Mead',
                fun=rate_twosided, x0=x0_opti)
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

        #axes.semilogx(axis_freq * setting.DT, np.ones_like(axis_freq)*max(convergence_factor), linestyle="dashed", linewidth=caracs[names]["width"], color=caracs[names]["color"]+"90")


    axes.legend( loc=(0., 0.), ncol=1 )
    #axes.set_xlim(left=1e-3, right=3.4)
    #axes.set_ylim(bottom=0)

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
        self.COURANT_NUMBER = 1.
        self.R = 0.#1e-4j
        self.D1=.5
        self.D2=1.
        self.M1=100
        self.M2=100
        self.LAMBDA_1=1e10 # >=0
        self.LAMBDA_2=-0. # <= 0
        self.SIZE_DOMAIN_1=100
        self.SIZE_DOMAIN_2=100
        self.DT = self.COURANT_NUMBER * (self.SIZE_DOMAIN_1 / (self.M1-1))**2 / self.D1

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
