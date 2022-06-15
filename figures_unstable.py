#!/usr/bin/python3
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple
from memoisation import memoised
from atm1DStratified import Atm1dStratified
from ocean1DStratified import Ocean1dStratified
from universal_functions import Businger_et_al_1971 as businger
from utils_linalg import solve_linear, full_to_half, oversample
from schwarz_coupler import NumericalSetting, schwarz_coupling, projection

def simulation_unstable(dt_atm, T, store_all: bool,
        sf_scheme_a: str, delta_sl_a: float=None,
        high_res: bool=False, u_G: float=8.):
    dt_oce: float = dt_atm
    sf_scheme_o: str = "FD pure" # this will not be used
    delta_sl_o: float= 0.
    NUMBER_SCHWARZ_ITERATION: int=0
    f = 1e-4 # Coriolis parameter
    time = np.linspace(0, T) # number of time steps is not important
    # because of the projection
    alpha, N0, rho0, cp = 0.0002, 0.01, 1024., 3985.
    Qswmax = 500.
    Qlw = -np.ones_like(time) * Qswmax / np.pi
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.26)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw = srflx * rho0*cp
    z_levels_oce = np.linspace(-50., 0., 51)
    z_levels_atm = np.concatenate((np.linspace(0.,500, 51),
        10*np.linspace(1,15,15)**1.5+500))
    if high_res:
        z_levels_atm = oversample(z_levels_atm, 3)
    simulator_oce = Ocean1dStratified(z_levels=z_levels_oce,
            dt=dt_oce, u_geostrophy=0., f=f, alpha=alpha, N0=N0)
    mu_m = 6.7e-2 # value of mu_m taken in bulk.py
    K_mol_a = simulator_oce.K_mol / mu_m
    simulator_atm = Atm1dStratified(z_levels=z_levels_atm,
            dt=dt_atm, u_geostrophy=u_G, K_mol=K_mol_a, f=f)

    if delta_sl_a is None:
        if sf_scheme_a in {"FV free", "FV test",
                "FD pure", "FD test"}:
            delta_sl_a = z_levels_atm[1]/2.
        else: # sf_scheme_a == "FD2":
            delta_sl_a = z_levels_atm[1]
    assert not (sf_scheme_a == "FV free" and abs(delta_sl_a)<1e-2)

    numer_setting = NumericalSetting(T=T,
            sf_scheme_a=sf_scheme_a, sf_scheme_o=sf_scheme_o,
            delta_sl_a=delta_sl_a, delta_sl_o=delta_sl_o,
            Q_lw=Qlw, Q_sw=Qsw)
    state_atm = schwarz_coupling(simulator_oce,
            simulator_atm, numer_setting, store_all=store_all,
            NUMBER_SCHWARZ_ITERATION=NUMBER_SCHWARZ_ITERATION)

    if store_all and sf_scheme_a[:2] == "FV":
        print("Reconstructing solutions...")
        all_u = state_atm.other["all_u"]
        all_phi = state_atm.other["all_phi"]
        all_t = state_atm.other["all_theta"]
        all_dzt = state_atm.other["all_dz_theta"]
        all_SL = state_atm.other["all_SL"]
        for frame in range(len(all_u)):
            za, all_u[frame], all_t[frame] = \
                    simulator_atm.reconstruct_FV(all_u[frame],
                    all_phi[frame], all_t[frame], all_dzt[frame],
                    all_SL[frame],
                    ignore_loglaw=(sf_scheme_a != "FV free"))
    elif not store_all and sf_scheme_a[:2] == "FV":
        za, state_atm.last_tstep["u"], \
                state_atm.last_tstep["theta"], = \
                    simulator_atm.reconstruct_FV(\
                    state_atm.last_tstep["u"],
                    state_atm.last_tstep["phi"],
                    state_atm.last_tstep["theta"],
                    state_atm.last_tstep["dz_theta"],
                    state_atm.other["SL"],
                    ignore_loglaw=(sf_scheme_a not in {"FV free",
                        "FV2"}))
    else:
        za = simulator_atm.z_half[:-1]
    return state_atm, za

def colorplot(z_levels: np.ndarray, sf_scheme: str,
        delta_sl: float):
    dt = 30.
    N = int(4*24*3600 / dt)
    T = dt*N
    skip_dt = 120
    state_atm, z_fv = memoised(simulation_unstable, dt, T, True,
            sf_scheme, delta_sl, high_res=False)

    variables = ("all_u", "all_theta", "z_tke", "all_tke",
            "all_u_star", "all_leps")
    all_u_fv, all_theta_fv, z_tke, all_tke_fv, all_ustar, \
            all_leps = [state_atm.other[x] for x in variables]

    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=1., right=0.93)
    u_fv, theta_fv = np.array(all_u_fv), np.array(all_theta_fv)
    x_time = np.linspace(0, (skip_dt*dt*u_fv.shape[0])/3600.,
            u_fv.shape[0])
    cmap = axes[0,0].pcolormesh(x_time, z_fv, np.real(u_fv.T),
            shading="nearest", cmap="jet", vmin=2., vmax=10.)
    fig.colorbar(cmap, ax=axes[0,0], label=r"U", location="right")

    cmap = axes[0,1].pcolormesh(x_time, z_fv, np.imag(u_fv.T),
            shading="nearest", cmap="jet", vmin=-3., vmax=3.)
    fig.colorbar(cmap, ax=axes[0,1], label=r"V", location="right")
    axes[0,0].set_ylabel("height (m)")
    cmap = axes[1, 0].pcolormesh(x_time, z_fv, theta_fv.T,
            shading="nearest", cmap="jet",
            vmin=263., vmax=269.)
    fig.colorbar(cmap, ax=axes[1,0], label=r"$\theta$", location="right")
    axes[1,0].set_xlabel("time (in hours)")
    axes[1,0].set_ylabel("height (m)")

    cmap = axes[1, 1].pcolormesh(x_time, z_tke,
            np.array(all_tke_fv).T, shading="nearest", cmap="jet",
            vmin=0., vmax=0.17)
    fig.colorbar(cmap, ax=axes[1,1], label=r"TKE", location="right")
    axes[1,1].set_xlabel("time (in hours)")

    if sf_scheme[:2] == "FV":
        cmap = axes[1, 2].pcolormesh(x_time, z_levels,
                np.array(all_leps).T, shading="nearest", cmap="jet",
                vmin=0., vmax=150.)
    else:
        cmap = axes[1, 2].pcolormesh(x_time, z_tke,
                np.array(all_leps).T, shading="nearest", cmap="jet",
                vmin=0., vmax=150.)
    fig.colorbar(cmap, ax=axes[1,2], label=r"$l_\epsilon$", location="right")
    axes[1,2].set_xlabel("time (in hours)")

    axes[0, 2].plot(x_time, all_ustar)
    axes[0, 2].set_ylabel("Friction scale u*")

def simulation(FV:bool, *args, **kwargs):
    if FV:
        return simulation_FV(*args, **kwargs)
    else:
        return simulation_FD(*args, **kwargs)

def simulation_FD(sf_scheme: str, z_levels: np.ndarray, dt: float,
        N: int, stable: bool, delta_sl: float, z_constant: float,
        spinup: int, skip_dt: int, skip_dx: int):
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=8.,
            K_mol=1e-4, f=1.39e-4)
    u_0 = 8*np.ones(M)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    if stable:
        SST = np.concatenate(([265],
            [265 - 0.25*(dt*(n-1))/3600. for n in range(1, N+1)]))
    else: # diurnal cycle:
        SST = np.concatenate(([265],
            [265 + 2.*np.sin((dt*(n-1))/3600. * np.pi / 12.)\
                    for n in range(1, N+1)]))
    ret = simulator.FD(u_t0=u_0, SST=SST,
            forcing_theta=np.zeros(simulator.M),
            sf_scheme=sf_scheme, forcing=forcing, store_all=True)
    all_u, all_TKE, all_ustar, all_temperature, all_leps = [ret[x] \
            for x in ("all_u", "all_tke", "all_theta", "all_leps")]
    for j in range(len(all_u)):
        all_u[j] = all_u[j][::skip_dx]
        all_TKE[j] = all_TKE[j][::skip_dx]
        all_temperature[j] = all_temperature[j][::skip_dx]
        all_leps[j] = all_leps[j][::skip_dx]
    z_tke = np.copy(simulator.z_full)
    z_half = np.copy(simulator.z_half[:-1])

    return z_half[::skip_dx], all_u[spinup::skip_dt], \
            all_temperature[spinup::skip_dt], z_tke[::skip_dx], \
            all_TKE[spinup::skip_dt], all_ustar[spinup::skip_dt], \
            all_leps[spinup::skip_dt]

def simulation_FV(sf_scheme: str, z_levels: np.ndarray, dt: float, N: int,
        stable: bool, delta_sl: float, z_constant: float, spinup: int,
        skip_dt: int, skip_dx: int):
    """
        performs a simulation and return all data during it.
    """
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=8.,
            K_mol=1e-4, f=1.39e-4)
    u_0 = 8*np.ones(M)
    phi_0 = np.zeros(M+1)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    if stable:
        SST = np.concatenate(([265],
            [265 - 0.25*(dt*(n-1))/3600. for n in range(1, N+1)]))
    else: # diurnal cycle:
        SST = np.concatenate(([265],
            [265 + 2.*np.sin((dt*(n-1))/3600. * np.pi / 12.)\
                    for n in range(1, N+1)]))

    z_tke = np.copy(simulator.z_full)
    k = bisect.bisect_right(z_levels[1:], delta_sl)
    z_tke[k] = delta_sl
    u_deltasl = 8. # first guess before the iterations
    if sf_scheme in {"FV1 free", "FV2 free", "FV free", "FV2"}:
        k_constant = bisect.bisect_right(z_levels[1:], z_constant)
        zk, zkp1 = z_levels[k], z_levels[k+1]
        h_tilde = z_levels[k+1] - delta_sl
        h_kp12 = z_levels[k+1] - z_levels[k]
        z_0M = 1e-1
        u_constant = 8.
        u_kp1 = 8.
        K_mol, kappa = simulator.K_mol, simulator.kappa
        for _ in range(15):
            u_star = kappa / np.log(1+delta_sl/z_0M) * np.abs(u_deltasl)
            z_0M = K_mol / kappa / u_star

            phi_0[k] = u_deltasl / (z_0M+delta_sl) / \
                    np.log(1+delta_sl/z_0M)
            # u_tilde + h_tilde (phi_0 / 6 + phi_1 / 3) = u_kp1
            # (subgrid reconstruction at the top of the volume)
            u_tilde = u_kp1 - h_tilde/6 * (phi_0[k]+2*phi_0[k+1])
            u_deltasl = u_tilde - h_tilde / 3 * phi_0[k]
            # For LES simulation, putting a quadratic profile between
            # the log law and the constant profile :
            def func_z(z):
                return 1-((z_constant - z) / (z_constant - delta_sl))**2

            u_kp1 = u_deltasl + (u_constant - u_deltasl) * func_z(zkp1)
            u_0[k+1:k_constant] = u_deltasl + (u_constant-u_deltasl) *\
                    func_z(simulator.z_half[k+1:k_constant])
            # compute_phi: with phi[k] = phi_0[k], 
            # with phi[k_constant] = 0,
            # and the FV approximation
            def compute_phi(bottom_cond, u_0, h_half):
                """ solving the system of finite volumes:
                phi_{m-1}/12 + 10 phi_m / 12 + phi_{m+1} / 12 =
                        (tke_{m+1/2} - tke_{m-1/2})/h
                """
                ldiag = h_half[:-1] /6.
                diag = (h_half[1:] + h_half[:-1]) * 1/3.
                udiag = h_half[1:] /6.
                diag = np.concatenate(([1.], diag, [1.]))
                udiag = np.concatenate(([0.], udiag))
                ldiag = np.concatenate((ldiag, [0.]))
                rhs = np.concatenate(([bottom_cond],
                    np.diff(u_0), [0.]))
                return solve_linear((ldiag, diag, udiag), rhs)
            phi_0[k:] = compute_phi(phi_0[k],
                    np.concatenate(([u_tilde], u_0[k+1:])),
                    np.concatenate(([h_tilde],
                        simulator.h_half[k+1:-1])))

        neutral_tau_sl = (delta_sl / (h_kp12))* \
                (1+z_0M/delta_sl - 1/np.log(1+delta_sl/z_0M) \
                + (zk - (zk+z_0M)*np.log(1+zk/z_0M)) \
                / (delta_sl * np.log(1+delta_sl/z_0M)))

        alpha_sl = h_tilde/h_kp12 + neutral_tau_sl
        u_0[k] = alpha_sl * u_tilde - neutral_tau_sl*h_tilde*phi_0[k]/3

    print("Starting the simulation")
    ret = simulator.FV(u_t0=u_0, phi_t0=phi_0,
                    forcing_theta=np.zeros(simulator.M),
                    SST=SST, sf_scheme=sf_scheme, u_delta=u_deltasl,
                    forcing=forcing, delta_sl=delta_sl, store_all=True)
    all_u, all_phi, all_TKE, all_dz_tke, all_ustar, all_temperature, \
        all_dz_theta, all_leps, all_SL = [ret[x] for x in ("all_u",
            "all_phi", "all_tke_bar", "all_dz_tke", 
            "all_u_star", "all_theta",
            "all_dz_theta", "all_leps", "all_SL")]

    print("Reconstructing the solution")
    all_u_fv, all_theta_fv, all_tke_fv = [], [], []
    for u, phi, temperature, dz_theta, SL in zip(all_u[spinup::skip_dt],
            all_phi[spinup::skip_dt], all_temperature[spinup::skip_dt],
            all_dz_theta[spinup::skip_dt], all_SL[spinup::skip_dt]):
        z_fv, u_fv, theta_fv = simulator.reconstruct_FV(u,
                phi, temperature, dz_theta, SL=SL)
        z_fv = z_fv[::skip_dx]
        u_fv = u_fv[::skip_dx]
        theta_fv = theta_fv[::skip_dx]
        all_u_fv += [u_fv]
        all_theta_fv += [theta_fv]
    for TKE, dz_tke, SL, l_eps in zip(all_TKE[spinup::skip_dt],
            all_dz_tke[spinup::skip_dt],
            all_SL[spinup::skip_dt], all_leps[spinup::skip_dt]):
        z_tke, tke_fv = simulator.reconstruct_TKE(TKE,
                dz_tke, SL, sf_scheme, businger, l_eps)
        z_tke = z_tke[::skip_dx]
        tke_fv = tke_fv[::skip_dx]
        all_tke_fv += [tke_fv]
    # z_tke = simulator.z_full
    return z_fv, all_u_fv, all_theta_fv, z_tke, \
            all_tke_fv, all_ustar[spinup::skip_dt], all_leps[spinup::skip_dt]
