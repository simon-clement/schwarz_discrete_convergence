#!/usr/bin/python3
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from memoisation import memoised
from atm1DStratified import Atm1dStratified
from ocean1DStratified import Ocean1dStratified
from universal_functions import Businger_et_al_1971 as businger
from utils_linalg import solve_linear
import figures_unstable
from fortran.visu import import_data
from schwarz_coupler import StateAtm, NumericalSetting, projection
from matplotlib.animation import FuncAnimation

DEFAULT_U_STAR = 0.01 * np.sqrt(1024.)
DEFAULT_T_STAR = 1e-6

def forcedOcean(sf_scheme: str):
    dt = 90.
    f = 1e-4
    alpha = 0.0002
    N0 = np.sqrt(alpha*9.81* 0.1)
    T0 = 289. # Reference temperature
    z_levels = np.linspace(-50., 0., 51)
    simulator_oce = Ocean1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0, loading_bar=True)
    number_of_days = 4
    T = 86400 * number_of_days
    N = int(T / dt)
    atm_state = StateAtm(u_delta=np.ones(N+1)*1.1+0j,
            t_delta=np.ones(N+1)*290.,
            u_star=np.ones(N+1)*DEFAULT_U_STAR,
            t_star=np.ones(N+1)*DEFAULT_T_STAR,
            last_tstep=None, other=None)

    Qlw = -np.ones(N+1) * 100.
    Qswmax = np.pi * Qlw[0] # to get balance between radiative fluxes
    Qswmax = 800.
    alpha, N0, rho0, cp = 0.0002, 0.01, 1024., 3985.
    time = np.linspace(0, T) # number of time steps is not important
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw = srflx * rho0*cp
    numer_set = NumericalSetting(T=T,
            sf_scheme_a=sf_scheme, sf_scheme_o=sf_scheme,
            delta_sl_a=10., delta_sl_o=-0., Q_lw=Qlw, Q_sw=Qsw)

    N = int(numer_set.T/simulator_oce.dt) # Number of time steps
    theta_0 = T0 - simulator_oce.N0**2 * \
            np.abs(simulator_oce.z_half[:-1]) \
            / simulator_oce.alpha / 9.81 # Initial temperature
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    dz_theta_0 = np.ones(simulator_oce.M+1) * simulator_oce.N0**2 \
            / simulator_oce.alpha / 9.81

    Q_sw = projection(np.asarray(numer_set.Q_sw), N)
    Q_lw = projection(np.asarray(numer_set.Q_lw), N)
    delta_sl = numer_set.delta_sl_o
    sf_scheme = numer_set.sf_scheme_o
    wind_10m = projection(atm_state.u_delta, N)
    temp_10m = projection(atm_state.t_delta, N)
    u_star = projection(atm_state.u_star, N)
    t_star = projection(atm_state.t_star, N)

    if sf_scheme in {"FV free", "FV2"}:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                simulator_oce.initialization(\
                np.zeros(simulator_oce.M)+0j, # u_0
                np.copy(theta_0), # theta_0
                delta_sl, wind_10m[0], temp_10m[0], u_star[0],
                t_star[0], Q_sw[0], Q_lw[0],
                10., sf_scheme)
    else:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                u_0, phi_0, theta_0, dz_theta_0, 0., T0

    if sf_scheme[:2] == "FV":
        ret = simulator_oce.FV(u_t0=u_i, phi_t0=phi_i,
                theta_t0=theta_i, dz_theta_t0=dz_theta_i, Q_sw=Q_sw,
                Q_lw=Q_lw, delta_sl_a=numer_set.delta_sl_a,
                u_star=u_star, t_star=t_star,
                u_delta=u_delta, t_delta=t_delta, delta_sl=delta_sl,
                store_all=True,
                wind_10m=wind_10m,
                temp_10m=temp_10m, sf_scheme=sf_scheme)

    elif sf_scheme[:2] == "FD":
        ret = simulator_oce.FD(u_t0=u_0, theta_t0=theta_0,
                delta_sl_a=numer_set.delta_sl_a,
                u_star=u_star, t_star=t_star,
                Q_sw=Q_sw, Q_lw=Q_lw, wind_10m=wind_10m,
                temp_10m=temp_10m, store_all=True,
                sf_scheme=sf_scheme)
    else:
        raise NotImplementedError("Cannot infer discretization " + \
                "from surface flux scheme name " + sf_scheme)
    return ret, simulator_oce.z_half[:-1]

def fig_animForcedOcean():
    ret, z = memoised(forcedOcean, "FV Zeng")
    # ret, z = memoised(constantCoolingForAnimation, "FV free")
    fig, axes = plt.subplots(1, 2)
    for u, ud in zip(ret["all_u"], ret["u_delta"]):
        u[-1] = ud
    for t, td in zip(ret["all_theta"], ret["t_delta"]):
        t[-1] = td
    line_u, = axes[0].plot(np.real(ret["all_u"][-1]), z)
    line_t, = axes[1].plot(ret["all_theta"][-1], z)
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[1].set_yscale("symlog", linthresh=0.1)
    def init():
        axes[0].set_xlim(-1., 2.)
        axes[1].set_xlim(280., 290.)
        axes[0].set_ylim(z[0], z[-1])
        axes[1].set_ylim(z[0], z[-1])
        return line_u, line_t
    def update(frame):
        line_u.set_data(np.real(ret["all_u"][frame]), z)
        line_t.set_data(ret["all_theta"][frame], z)
        return line_u, line_t
    ani = FuncAnimation(fig, update,
            frames=range(0, len(ret["all_u"]), 100),
                    init_func=init, blit=True)

    show_or_save("fig_animForcedOcean")

def constantCoolingForAnimation(sf_scheme: str):
    dt = 30.
    f = 1e-4
    alpha = 0.0002
    N0 = np.sqrt(alpha*9.81* 0.1)
    T0 = 16.
    z_levels = np.linspace(-50., 0., 51)
    simulator_oce = Ocean1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0)

    # N_FOR = nb_steps = int(72 * 3600 / dt)
    N = int(72*3600/dt)
    time = dt * np.arange(N+1)
    rho0, cp, Qswmax = 1024., 3985., 0.
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw, Qlw = srflx * rho0*cp, np.zeros_like(srflx)
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    Qlw = -np.ones(N+1) * 100
    wind_10m = 1.1*np.ones(N+1) + 0j
    temp_10m = np.ones(N+1) * 5
    # if temp_10m<T0, then __friction_scales does not converge.
    delta_sl= -.5
    z_levels_free = np.concatenate((z_levels[:-1], [delta_sl]))

    if sf_scheme == "FV free":
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                simulator_oce.initialization(\
                np.zeros(simulator_oce.M)+0j, # u_0
                np.copy(theta_0), # theta_0
                delta_sl, wind_10m[0], temp_10m[0],
                DEFAULT_U_STAR, DEFAULT_T_STAR,
                Qsw[0], Qlw[0],
                10., sf_scheme)
    else:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                u_0, phi_0, theta_0, dz_theta_0, 0., T0

    ret =simulator_oce.FV(delta_sl_a=10.,
            u_t0=u_i, phi_t0=phi_i, theta_t0=theta_i,
            dz_theta_t0=dz_theta_i, Q_sw=Qsw, Q_lw=Qlw,
            u_delta=u_delta, t_delta=t_delta, delta_sl=delta_sl,
            u_star=np.ones(N+1)*DEFAULT_U_STAR,
            t_star=np.ones(N+1)*DEFAULT_T_STAR,
            wind_10m=wind_10m,
            store_all=True,
            temp_10m=temp_10m, sf_scheme=sf_scheme)
    return ret, simulator_oce.z_half[:-1]

def fig_constantCooling():
    dt = 30.
    f = 1e-2
    alpha = 0.0002
    N0 = np.sqrt(alpha*9.81* 0.1)
    T0 = 16.
    z_levels = np.linspace(-50., 0., 51)
    simulator_oce = Ocean1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0)

    # N_FOR = nb_steps = int(72 * 3600 / dt)
    N = int(72*3600/dt)
    time = dt * np.arange(N+1)
    rho0, cp, Qswmax = 1024., 3985., 0.
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw, Qlw = srflx * rho0*cp, np.zeros_like(srflx)
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    Qlw = -np.ones(N+1) * 100
    wind_10m = 1.1*np.ones(N+1) + 0j
    temp_10m = np.ones(N+1) * 5
    # if temp_10m<T0, then __friction_scales does not converge.

    fig, axes = plt.subplots(1, 6, figsize=(13,4))
    delta_sl= -.5
    z_levels_free = np.concatenate((z_levels[:-1], [delta_sl]))

    for sf_scheme in ("FV free",):
        if sf_scheme == "FV free":
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    simulator_oce.initialization(\
                    np.zeros(simulator_oce.M)+0j, # u_0
                    np.copy(theta_0), # theta_0
                    delta_sl, wind_10m[0], temp_10m[0],
                    DEFAULT_U_STAR, DEFAULT_T_STAR,
                    Qsw[0], Qlw[0],
                    10., sf_scheme)
        else:
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    u_0, phi_0, theta_0, dz_theta_0, 0., T0

        ret =simulator_oce.FV(delta_sl_a=10.,
                u_t0=u_i, phi_t0=phi_i, theta_t0=theta_i,
                dz_theta_t0=dz_theta_i, Q_sw=Qsw, Q_lw=Qlw,
                u_star=np.ones(N+1)*DEFAULT_U_STAR,
                t_star=np.ones(N+1)*DEFAULT_T_STAR,
                u_delta=u_delta, t_delta=t_delta, delta_sl=delta_sl,
                wind_10m=wind_10m,
                temp_10m=temp_10m, sf_scheme=sf_scheme)
        u_current, phi, tke, theta, dz_theta, l_eps, SL, viscosity = \
                [ret[x] for x in ("u", "phi", "tke", "theta",
                    "dz_theta", "l_eps", "SL", "Ktheta")]
        zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
                phi, theta, dz_theta, SL, ignore_loglaw=False)

        axes[0].plot(thetaFV, zFV, "--",
                label="Temperature Python FV")
        axes[1].plot(np.real(uFV), zFV, "--",
                label="wind speed Python FV")
        axes[2].plot(np.imag(uFV), zFV, "--",
                label="wind speed Python FV")
        axes[3].plot(viscosity, z_levels_free, "--",
                label="Diffusivity Python FV")
        axes[4].plot(l_eps, z_levels_free, "--",
                label="Diffusivity Python FV")
        axes[5].plot(tke, z_levels_free, "--",
                label="tke Python FV")

    retFD = simulator_oce.FD(delta_sl_a=10.,
            u_t0=u_0, theta_t0=theta_0,
            Q_sw=Qsw, Q_lw=Qlw, wind_10m=wind_10m,
            u_star=np.ones(N+1)*DEFAULT_U_STAR,
            t_star=np.ones(N+1)*DEFAULT_T_STAR,
            temp_10m=temp_10m,
            sf_scheme="FD pure")
    u_currentFD, tke, all_u_star, thetaFD, l_eps, viscosityFD = \
             [retFD[x] for x in("u", "tke", "all_u_star", "theta",
                 "l_eps", "Ktheta")]

    axes[0].plot(thetaFD, simulator_oce.z_half[:-1], "--",
            label="Temperature Python FD")
    axes[1].plot(np.real(u_currentFD), simulator_oce.z_half[:-1], "--",
            label="wind speed (u) Python FD")
    axes[2].plot(np.imag(u_currentFD), simulator_oce.z_half[:-1], "--",
            label="wind speed (v) Python FD")
    axes[3].plot(viscosityFD, simulator_oce.z_full, "--",
            label="Diffusivity Python FD")
    axes[4].plot(l_eps, simulator_oce.z_full, "--",
            label="Diffusivity Python FD")
    axes[5].plot(tke, simulator_oce.z_full, "--",
            label="Diffusivity Python FD")

    axes[0].legend()
    for i in range(6):
        axes[i].set_yscale("symlog", linthresh=0.1)
    axes[0].set_title("temperature")
    axes[1].set_title("wind (u)")
    axes[2].set_title("wind (v)")
    axes[3].set_title("diffusivity")
    axes[4].set_title("mxlm")
    axes[5].set_title("tke")
    show_or_save("fig_constantCooling")

def fig_windInduced():
    dt = 30.
    f = 1e-2
    T0, alpha, N0 = 16., 0.0002, 0.01
    z_levels = np.linspace(-50., 0., 51)
    simulator_oce = Ocean1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0)

    N = 1000
    time = dt * np.arange(N+1)
    rho0, cp, Qswmax = 1024., 3985., 0.
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw, Qlw = srflx * rho0*cp, np.zeros_like(srflx)
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    wind_10m = np.ones(N+1) * 11.6 + 0j
    temp_10m = np.ones(N+1) * T0

    fig, axes = plt.subplots(1, 5)
    axes[0].set_title("Temperature")
    axes[1].set_title("Diffusivity")
    axes[2].set_title("wind")
    axes[4].set_title("tke")
    # for sf_scheme in ("FV test", "FV1", "FV pure", "FV free"):
    for sf_scheme in ("FV free",):
        if sf_scheme == "FV free":
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    simulator_oce.initialization(\
                    np.zeros(simulator_oce.M)+0j, # u_0
                    np.copy(theta_0), # theta_0
                    -.5, wind_10m[0], temp_10m[0],
                    DEFAULT_U_STAR, DEFAULT_T_STAR,
                    Qsw[0], Qlw[0], 10., sf_scheme)
        else:
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    u_0, phi_0, theta_0, dz_theta_0, 0., T0



        ret = simulator_oce.FV(delta_sl_a=10.,
                u_t0=u_i, phi_t0=phi_i, theta_t0=theta_i,
                dz_theta_t0=dz_theta_i, Q_sw=Qsw, Q_lw=Qlw,
                u_delta=u_delta, t_delta=t_delta,
                u_star=np.ones(N+1)*DEFAULT_U_STAR,
                t_star=np.ones(N+1)*DEFAULT_T_STAR,
                wind_10m=wind_10m,
                temp_10m=temp_10m, sf_scheme=sf_scheme)
        u_current, phi, tke, theta, dz_theta, SL, viscosity = \
                [ret[x] for x in ("u", "phi", "tke", "theta",
                    "dz_theta", "SL", "Ktheta")]
        zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
                phi, theta, dz_theta, SL, ignore_loglaw=False)
        axes[0].plot(thetaFV, zFV, "--",
                label=sf_scheme)
        axes[1].plot(viscosity, simulator_oce.z_full, "--",
                label=sf_scheme)
        axes[2].plot(np.real(uFV), zFV, "--",
                label=sf_scheme)
        axes[4].plot(tke, simulator_oce.z_full, "--",
                label=sf_scheme)

    # for sf_scheme in ("FD test", "FD pure", "FD2"):
    for sf_scheme in ("FD pure",):
        retFD = simulator_oce.FD(delta_sl_a=10.,
                u_t0=u_0, theta_t0=theta_0,
                u_star=np.ones(N+1)*DEFAULT_U_STAR,
                t_star=np.ones(N+1)*DEFAULT_T_STAR,
                Q_sw=Qsw, Q_lw=Qlw, wind_10m=wind_10m,
                temp_10m=temp_10m,
                sf_scheme=sf_scheme)
        u_currentFD, tke, all_u_star, thetaFD, l_eps, viscosityFD = \
                 [retFD[x] for x in("u", "tke", "all_u_star", "theta",
                     "l_eps", "Ktheta")]
        axes[0].plot(thetaFD, simulator_oce.z_half[:-1], "--",
                label=sf_scheme)
        axes[1].plot(viscosityFD, simulator_oce.z_full, "--",
                label=sf_scheme)
        axes[2].plot(np.real(u_currentFD), simulator_oce.z_half[:-1], "--",
                label=sf_scheme)
        axes[4].plot(tke, simulator_oce.z_full, "--",
                label=sf_scheme)

    axes[0].legend()
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[3].set_yscale("symlog", linthresh=0.1)
    axes[4].set_yscale("symlog", linthresh=0.1)
    show_or_save("fig_windInduced")


def fig_comodoParamsConstantCooling():
    dt = 30.
    f = 0.
    alpha = 0.0002
    N0 = np.sqrt(alpha*9.81* 0.1)
    T0 = 289.
    z_levels = np.linspace(-50., 0., 51)
    simulator_oce = Ocean1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0)

    N_FOR = nb_steps = int(72 * 3600 / dt)
    N = N_FOR + 1
    time = dt * np.arange(N+1)
    rho0, cp, Qswmax = 1024., 3985., 0.
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw, Qlw = srflx * rho0*cp, np.zeros_like(srflx)
    u_0 = np.zeros(simulator_oce.M) + 0j
    phi_0 = np.zeros(simulator_oce.M+1) + 0j
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    Qlw = -np.ones(N+1) * 100 # /!\ definition of Q0 is not the same as Florian
    wind_10m = np.zeros(N+1) + 0j + 1.
    temp_10m = np.ones(N+1) * T0 - 1

    u_star = np.ones(N+1)*1e-5
    t_star = -np.ones(N+1)*DEFAULT_T_STAR
    fig, axes = plt.subplots(1, 2)
    sf_scheme = "FV free"
    for delta_sl in (0., -0.5, -1.2):
        if sf_scheme == "FV free":
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    simulator_oce.initialization(\
                    u_0, # u_0
                    np.copy(theta_0), # theta_0
                    delta_sl, wind_10m[0], temp_10m[0],
                    u_star[0], t_star[0],
                    Qsw[0], Qlw[0], 10., sf_scheme)
        else:
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    u_0, phi_0, theta_0, dz_theta_0, 10., 288.

        ret = simulator_oce.FV(delta_sl_a=10.,
                u_t0=u_i, phi_t0=phi_i, theta_t0=theta_i,
                u_delta=u_delta, t_delta=t_delta,
                u_star=u_star, t_star=t_star,
                delta_sl=delta_sl,
                dz_theta_t0=dz_theta_i, Q_sw=Qsw, Q_lw=Qlw,
                wind_10m=wind_10m,
                temp_10m=temp_10m, sf_scheme=sf_scheme)
        u_current, phi, theta, dz_theta, SL, viscosity = \
                    [ret[x] for x in ("u", "phi", "theta",
                        "dz_theta", "SL", "Ktheta")]
        zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
                phi, theta, dz_theta, SL, ignore_loglaw=False)

        axes[0].plot(thetaFV, zFV, "--",
                label="FV, delta=" + str(delta_sl))
        axes[1].plot(viscosity, simulator_oce.z_full, "--",
                label="FV, delta=" + str(delta_sl))

    for sf_scheme in ("FD pure", "FD2"):
        retFD = simulator_oce.FD(delta_sl_a=10.,
                u_t0=u_0, theta_t0=theta_0,
                u_star=u_star, t_star=t_star,
                Q_sw=Qsw, Q_lw=Qlw, wind_10m=wind_10m,
                temp_10m=temp_10m,
                sf_scheme=sf_scheme)
        u_currentFD, tke, all_u_star, thetaFD, l_eps, viscosityFD = \
                 [retFD[x] for x in("u", "tke", "all_u_star", "theta",
                     "l_eps", "Ktheta")]
        axes[0].plot(thetaFD, simulator_oce.z_half[:-1], "--",
                label=sf_scheme)
        axes[1].plot(viscosityFD, simulator_oce.z_full, "--",
                label=sf_scheme)

    axes[1].legend(loc="lower center")
    axes[0].set_xlabel("Temperature")
    axes[1].set_xlabel("Diffusivity")
    show_or_save("fig_comodoParamsConstantCooling")


def fig_comodoParamsWindInduced():
    dt = 30.
    f = 0.
    T0, alpha, N0 = 16., 0.0002, 0.01
    z_levels = np.linspace(-50., 0., 51)
    simulator_oce = Ocean1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0)

    N_FOR = nb_steps = int(30 * 3600 / dt)
    N = N_FOR + 1
    time = dt * np.arange(N+1)
    rho0, cp, Qswmax = 1024., 3985., 0.
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw, Qlw = srflx * rho0*cp, np.zeros_like(srflx)
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    wind_10m = np.ones(N+1) * 11.6 + 0j
    temp_10m = np.ones(N+1) * T0

    ret = simulator_oce.FV(delta_sl_a=10.,
            u_t0=u_0, phi_t0=phi_0, theta_t0=theta_0,
            u_delta=0., t_delta=T0,
            u_star=np.ones(N+1)*DEFAULT_U_STAR,
            t_star=np.ones(N+1)*0.,
            delta_sl=0.,
            dz_theta_t0=dz_theta_0, Q_sw=Qsw, Q_lw=Qlw,
            wind_10m=wind_10m, temp_10m=temp_10m,
            sf_scheme="FV free")
    u_current, phi, theta, dz_theta, SL, viscosity = \
                [ret[x] for x in ("u", "phi", "theta",
                    "dz_theta", "SL", "Ktheta")]
    zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
            phi, theta, dz_theta, SL, ignore_loglaw=True)

    retFD = simulator_oce.FD(delta_sl_a=10.,
            u_t0=u_0, theta_t0=theta_0,
            u_star=np.ones(N+1)*DEFAULT_U_STAR,
            t_star=np.ones(N+1)*0.,
            Q_sw=Qsw, Q_lw=Qlw, wind_10m=wind_10m, temp_10m=temp_10m,
            sf_scheme="FD pure")
    thetaFD, viscosityFD = [retFD[x] for x in("theta", "Ktheta")]

    fig, axes = plt.subplots(1, 2)
    #### Getting fortran part ####
    name_file = "fortran/output_debug.out"
    t_for, zt_for = import_data("fortran/t_final_tke.out")
    Kt_for, zKt_for = import_data("fortran/Akt_final_tke.out")
    axes[0].plot(t_for, zt_for, label="Temperature Fortran")
    axes[1].plot(Kt_for, zKt_for, label="Diffusivity Fortran")

    #### Python plotting ####
    axes[0].plot(thetaFV, zFV, "--",
            label="Temperature Python FV")
    axes[0].plot(thetaFD, simulator_oce.z_half[:-1], "--",
            label="Temperature Python FD")
    axes[1].plot(viscosity, simulator_oce.z_full, "--",
            label="Diffusivity Python FV")
    axes[1].plot(viscosityFD, simulator_oce.z_full, "--",
            label="Diffusivity Python FD")

    axes[0].legend()
    axes[1].legend()
    show_or_save("fig_comodoParamsWindInduced")

def show_or_save(name_func):
    """
    By using this function instead plt.show(),
    the user has the possibiliy to use ./figsave name_func
    name_func must be the name of your function
    as a string, e.g. "fig_comparisonData"
    """
    from figures import SAVE_TO_PNG, SAVE_TO_PGF, SAVE_TO_PDF
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
