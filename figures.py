#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from memoisation import memoised
from atm1DStratified import Atm1dStratified
from ocean1DStratified import Ocean1dStratified
from universal_functions import Businger_et_al_1971 as businger
from universal_functions import Large_et_al_2019 as large_ocean
from utils_linalg import solve_linear, full_to_half, oversample
from utils_linalg import undersample
import figures_unstable
from fortran.visu import import_data
from validation_oce1D import fig_comodoParamsConstantCooling
from validation_oce1D import fig_comodoParamsWindInduced
from validation_oce1D import fig_windInduced, fig_constantCooling
from validation_oce1D import fig_animForcedOcean
from schwarz_coupler import NumericalSetting, schwarz_coupling, projection

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath, amsfonts}"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = ':'
mpl.rcParams["grid.alpha"] = '0.7'
mpl.rcParams["grid.linewidth"] = '0.5'
def palette():
    """returns an array of colors"""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    DEFAULT_mpl = prop_cycle.by_key()['color']
    return ['#d84f37', '#6588cd', '#94b927', '#8033ac',
            '#00a9b2', '#c65c8a', '#e3a21a']

DEFAULT_z_levels = np.linspace(0, 1500, 41)
DEFAULT_z_levels_stratified = np.linspace(0, 400, 65)
IFS_z_levels = np.flipud(np.array((1600.04, 1459.58, 1328.43,
    1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70,
    566.49, 500.91, 440.58, 385.14, 334.22, 287.51, 244.68,
    205.44, 169.50, 136.62, 106.54, 79.04, 53.92, 30.,
    10.00))) - 10. # Not really IFS levels, since the list here
# is actually z_half. So the correct way would be to add 0
# and to take the middle of all space step here

IFS_z_levels_stratified = np.flipud(np.array((500.91, 440.58, 385.14,
    334.22, 287.51, 244.68,
    205.44, 169.50, 136.62, 106.54, 79.04, 53.92, 30.96,
    10.00))) - 10. # less levels in the stratified case

def fig_forcedOcean():
    from validation_oce1D import forcedOcean
    return forcedOcean("FD2")

def simulation_coupling(dt_oce, dt_atm, T, store_all: bool,
        sf_scheme_a: str, sf_scheme_o: str,
        delta_sl_o: float=None, delta_sl_a: float=None,
        NUMBER_SCHWARZ_ITERATION: int=3, high_res: bool=False):
    f = 1e-4 # Coriolis parameter
    time = np.linspace(0, T) # number of time steps is not important
    # because of the projection
    alpha, N0, rho0, cp = 0.0002, 0.01, 1024., 3985.
    Qswmax = 500.
    Qlw = -np.ones_like(time) * Qswmax / np.pi
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.26)), 0. ) * \
            Qswmax / (rho0*cp)
    Qsw = srflx * rho0*cp
    z_levels_oce = np.concatenate((-np.linspace(30,1,30)**1.5-50,
        np.linspace(-50., 0., 51)))
    z_levels_atm = np.concatenate((np.linspace(0.,500, 51),
        10*np.linspace(1,15,15)**1.5+500))
    if high_res:
        z_levels_oce = oversample(z_levels_oce, 3)
        z_levels_atm = oversample(z_levels_atm, 3)
    simulator_oce = Ocean1dStratified(z_levels=z_levels_oce,
            dt=dt_oce, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0)
    mu_m = 6.7e-2 # value of mu_m taken in bulk.py
    K_mol_a = simulator_oce.K_mol / mu_m
    simulator_atm = Atm1dStratified(z_levels=z_levels_atm,
            dt=dt_atm, u_geostrophy=8., K_mol=K_mol_a, f=f)
    if delta_sl_o is None:
        if sf_scheme_o == "FV free":
            delta_sl_o = z_levels_oce[-2]
        elif sf_scheme_o == "FD2":
            delta_sl_o = z_levels_oce[-2]
        elif sf_scheme_o in {"FV test", "FD test", "FD pure"}:
            delta_sl_o = 0.

    if delta_sl_a is None:
        if sf_scheme_a in {"FV free", "FV test",
                "FD pure", "FD test", "FV1"}:
            delta_sl_a = z_levels_atm[1]/2.
        else: # sf_scheme_a == "FD2", "FV2", "FVNishizawa":
            delta_sl_a = z_levels_atm[1]
    assert not (sf_scheme_o == "FV free" and abs(delta_sl_o)<1e-2)
    assert not (sf_scheme_a == "FV free" and abs(delta_sl_a)<1e-2)

    numer_setting = NumericalSetting(T=T,
            sf_scheme_a=sf_scheme_a, sf_scheme_o=sf_scheme_o,
            delta_sl_a=delta_sl_a,
            delta_sl_o=delta_sl_o,
            Q_lw=Qlw,
            Q_sw=Qsw)
    states_atm, states_oce = schwarz_coupling(simulator_oce,
            simulator_atm, numer_setting, store_all=store_all,
            NUMBER_SCHWARZ_ITERATION=NUMBER_SCHWARZ_ITERATION)
    if store_all and sf_scheme_a[:2] == "FV":
        print("Reconstructing solutions...")
        for state_atm in states_atm:
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
        for state_atm in states_atm:
            za, state_atm.last_tstep["u"], \
                    state_atm.last_tstep["theta"], = \
                        simulator_atm.reconstruct_FV(\
                        state_atm.last_tstep["u"],
                        state_atm.last_tstep["phi"],
                        state_atm.last_tstep["theta"],
                        state_atm.last_tstep["dz_theta"],
                        state_atm.other["SL"],
                        ignore_loglaw=(sf_scheme_a not in \
                                {"FV free", "FV2"}))
    else:
        za = simulator_atm.z_half[:-1]

    if store_all and sf_scheme_o[:2] == "FV":
        for state_oce in states_oce[1:]:
            all_u = state_oce.other["all_u"]
            all_phi = state_oce.other["all_phi"]
            all_t = state_oce.other["all_theta"]
            all_dzt = state_oce.other["all_dz_theta"]
            all_SL = state_oce.other["all_SL"]
            for frame in range(len(all_u)):
                zo, all_u[frame], all_t[frame] = \
                        simulator_oce.reconstruct_FV(all_u[frame],
                        all_phi[frame], all_t[frame], all_dzt[frame],
                        all_SL[frame],
                        ignore_loglaw=(sf_scheme_o != "FV free"))
        print("... Done.")
    elif not store_all and sf_scheme_o[:2] == "FV":
        for state_oce in states_oce[1:]:
            zo, state_oce.last_tstep["u"], \
                    state_oce.last_tstep["theta"], = \
                        simulator_oce.reconstruct_FV(\
                        state_oce.last_tstep["u"],
                        state_oce.last_tstep["phi"],
                        state_oce.last_tstep["theta"],
                        state_oce.last_tstep["dz_theta"],
                        state_oce.other["SL"],
                        ignore_loglaw=(sf_scheme_o != "FV free"))
    else:
        zo = simulator_oce.z_half[:-1]

    return states_atm, states_oce, za, zo

def half_to_full(z_half: np.ndarray, ocean: bool):
    z_min = z_half[0] + z_half[0] - z_half[1] if ocean else 0.
    z_max = z_half[-1] + z_half[-1] - z_half[-2] if not ocean else 0.
    return np.concatenate(([z_min], (z_half[1:] + z_half[:-1])/2,
            [z_max]))


def endProfile_coupling(axu, axtheta,
        sf_scheme_a: str, sf_scheme_o: str,
        delta_sl_o: float=None, delta_sl_a: float=None,
        ignore_cached: bool=False,
        ITERATION: int=1, high_res: bool=False, style: str=None,
        label_atm: bool=True):
    dt_oce = 90. # oceanic time step
    dt_atm = 30. # atmosphere time step
    number_of_days = 3.3
    T = 86400 * number_of_days # length of the time window
    states_atm, states_oce, za, zo = \
        memoised(simulation_coupling, dt_oce, dt_atm, T, False,
                sf_scheme_a=sf_scheme_a, sf_scheme_o=sf_scheme_o,
                ignore_cached=ignore_cached,
                delta_sl_a=delta_sl_a,
                delta_sl_o=delta_sl_o, high_res=high_res,
                NUMBER_SCHWARZ_ITERATION=max(1, ITERATION+1))
    delta_atm = {"FD2": (za[0] + za[1])/2,
            "FD pure": za[0], "FV free" : za[0]}
    delta_oce = {"FD2": -1.,
            "FD pure": 0., "FV free" : -1., "FV test": 0.}
    state_atm = states_atm[ITERATION]
    state_oce = states_oce[ITERATION+1]

    delta_a = delta_atm if delta_sl_a is None else delta_sl_a
    delta_o = delta_oce if delta_sl_o is None else delta_sl_o
    if label_atm:
        label = sf_scheme_a[:2] +r", $\delta_{sl}=$" + \
                  f"{delta_a:.2f}m"
    else:
        label = f"{sf_scheme_o[:2]}" +r", $\delta_{o}=$" + \
                f"{delta_o:.2f}m"
    if high_res:
        label += " (high res)"
    axu.plot(np.real(state_atm.last_tstep["u"]), za, style, label=label)
    axu.plot(np.real(state_oce.last_tstep["u"]), zo, style)
    axtheta.plot(state_atm.last_tstep["theta"], za, style, label=label)
    axtheta.plot(state_oce.last_tstep["theta"], zo, style)

def fig_endProfile_coupling():
    fig, axes = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.67)
    colors= {False:["r", "b", "y"],
            True:["r--", "b--", "y--"],
            }
    for high_res in (False, True):
        all_delta_sl = (5./3, 5., 1.) if high_res \
                else (5., 5., 1.)
        for sf_scheme_a, delta_sl_a, style in tqdm(zip(
            ("FD pure", "FV free", "FV free"),
            all_delta_sl, colors[high_res]), leave=False, total=4):
            endProfile_coupling(axes[0], axes[1], sf_scheme_a,
                "FD pure", ITERATION=1,
                delta_sl_a=delta_sl_a, high_res=high_res,
                ignore_cached=False, style=style, label_atm=True)

    axes[0].set_ylabel("z")
    axes[1].set_ylabel(r"$\theta^\star$")
    axes[1].set_xlabel(r"$\theta$")
    axes[0].set_xlabel("u")
    axes[1].legend()
    show_or_save("fig_endProfile_coupling")

def colorplot_coupling(ax, sf_scheme_a: str, sf_scheme_o: str,
        vmin: float=None, vmax: float=None, delta_sl_o: float=None,
        delta_sl_a: float=None,
        ignore_cached: bool=False,
        ITERATION: int=1, high_res: bool=False,
        label_atm: bool=True):
    dt_oce = 90. # oceanic time step
    dt_atm = 30. # atmosphere time step
    number_of_days = 3.3
    T = 86400 * number_of_days # length of the time window
    states_atm, states_oce, za, zo = \
        memoised(simulation_coupling, dt_oce, dt_atm, T, True,
                sf_scheme_a=sf_scheme_a, sf_scheme_o=sf_scheme_o,
                ignore_cached=ignore_cached,
                delta_sl_a=delta_sl_a,
                delta_sl_o=delta_sl_o, high_res=high_res,
                NUMBER_SCHWARZ_ITERATION=max(1, ITERATION+1))

    delta_atm = {"FD2": (za[0] + za[1])/2,
            "FD pure": za[0], "FV free" : za[0]}
    delta_oce = {"FD2": -1.,
            "FD pure": 0., "FV free" : -1., "FV test": 0.}

    state_atm = states_atm[ITERATION]
    state_oce = states_oce[ITERATION+1]
    all_ua = np.real(np.array(state_atm.other["all_u"]))
    all_ta = np.array(state_atm.other["all_theta"])
    all_uo = np.real(np.array(state_oce.other["all_u"]))
    all_to = np.array(state_oce.other["all_theta"])
    N_oce = int(T/dt_oce)
    N_plot = 120
    ua = np.zeros((N_plot+1, all_ua.shape[1]))
    ta = np.zeros((N_plot+1, all_ta.shape[1]))
    uo = np.zeros((N_plot+1, all_uo.shape[1]))
    to = np.zeros((N_plot+1, all_to.shape[1]))
    for i in range(all_ua.shape[1]):
        ua[:, i] = projection(all_ua[:, i], N_plot)
        ta[:, i] = projection(all_ta[:, i], N_plot)
    for i in range(uo.shape[1]):
        uo[:, i] = projection(all_uo[:, i], N_plot)
        to[:, i] = projection(all_to[:, i], N_plot)

    N_threshold = 0
    T_threshold = T * N_threshold / N_plot
    ########## pcolormesh
    x = np.linspace(T_threshold/86400, T/86400,
            N_plot+2 - N_threshold)
    Xa, Ya = np.meshgrid(half_to_full(za, ocean=False), x)
    Xo, Yo = np.meshgrid(half_to_full(zo, ocean=True), x)

    vmin = min(np.min(ta), np.min(to)) if vmin is None else vmin
    vmax = max(np.max(ta), np.max(to)) if vmax is None else vmax

    col_a = ax.pcolormesh(Ya, Xa, ta[N_threshold:], vmin=vmin,
            vmax=vmax, cmap="seismic", shading='flat',
            rasterized=True)
    ax.pcolormesh(Yo, Xo, to[N_threshold:], vmin=vmin,
            vmax=vmax, cmap="seismic", shading='flat',
            rasterized=True)
    delta_o =  delta_oce[sf_scheme_o] if delta_sl_o is None \
            else delta_sl_o
    delta_a = delta_atm[sf_scheme_a] if delta_sl_a is None \
            else delta_sl_a
    title = ""
    if label_atm:
        title = "Atm: " + sf_scheme_a[:2] +r", $\delta_a=$" + \
                f"{delta_a:.2f}m"
    else:
        title += "Ocean: "+ sf_scheme_o[:2] + \
                r", $\delta_o=$" + f"{delta_o:.2f}m"
    ax.set_title(title)
    ax.set_yscale("symlog", linthresh=1.)
    return col_a

def fig_mixing_lengths():
    """
    return z_fv, u_fv, theta_fv, z_tke, TKE, ustar
    """
    sf_scheme: str = "FV free"
    z_levels: np.nd_array = np.copy(IFS_z_levels)
    dt: float = 10.
    N: int = 3240
    u_G=8.
    stable: bool = False
    delta_sl: float = IFS_z_levels_stratified[1]/2.
    z_constant: float = 2 * delta_sl
    colors = palette()

    M: int = z_levels.shape[0] - 1
    simulator: Atm1dStratified = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=u_G,
            K_mol=1.5e-5, f=1.39e-4)
    T0 = 265.
    u_0 = u_G*np.ones(M) + 0j
    phi_0 = np.zeros(M+1) + 0j
    t_0, dz_theta_0 = simulator.initialize_theta(Neutral_case=False)
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
    u_deltasl = u_G # first guess before the iterations
    t_deltasl = T0 # first guess before the iterations
    Q_sw, Q_lw, delta_sl_o = np.zeros(N+1), np.zeros(N+1), 0.
    u_o, t_o = np.zeros(N+1), SST
    if sf_scheme in {"FV1 free", "FV2 free", "FV free", "FV2"}:
        u_i, phi_i, t_i, dz_theta_i, u_delta_i, t_delta_i = \
                simulator.initialization(u_0, phi_0, t_0, dz_theta_0,
                        delta_sl, u_o[0], t_o[0], Q_sw[0], Q_lw[0],
                        z_constant, delta_sl_o, u_G)
    else:
        u_i, phi_i, t_i, dz_theta_i, u_delta_i, t_delta_i = \
                u_0, phi_0, t_0, dz_theta_0, u_deltasl, t_deltasl

    ret = simulator.FV(u_t0=u_i, phi_t0=phi_i, theta_t0=t_i,
                    delta_sl_o=0.,
                    forcing_theta=np.zeros(simulator.M),
                    dz_theta_t0=dz_theta_i, Q_sw=Q_sw, Q_lw=Q_lw,
                    u_o=u_o, SST=SST, sf_scheme=sf_scheme,
                    u_delta=u_delta_i, t_delta=t_delta_i,
                    forcing=forcing, delta_sl=delta_sl)
    fig, axes = plt.subplots(1,2, figsize=(6.5, 3.5))
    fig.subplots_adjust(bottom=0.16, left=0.1)
    z_levels[0] = delta_sl
    axes[0].plot(simulator.lD80_copy, z_levels,
            label=r"$l^\star_{D80}$", color=colors[0])
    axes[0].plot(simulator.lup_copy, z_levels,
            label=r"$l_{up}$", color=colors[1])
    axes[0].plot(simulator.ldown_copy, z_levels,
            label=r"$l_{down}$", color=colors[2])
    axes[0].plot(np.sqrt(simulator.lup_copy * simulator.ldown_copy),
            z_levels, "--", label=r"$l_{\epsilon}$",
            color=colors[3])
    axes[0].set_ylabel("z (m)")
    axes[0].set_xlabel("Mixing length (m)")
    axes[0].set_ylim(top=300., bottom=0.)
    axes[1].set_ylim(top=300., bottom=0.)
    axes[1].set_xlim(left=0., right=0.255)
    axes[0].legend()

    axes[1].plot(ret["tke"], z_levels, "k")
    axes[1].set_xlabel(r"TKE (${\rm m}^2 . {\rm s}^{-2}$)")
    show_or_save("fig_mixing_lengths")


def fig_testBulk():
    from bulk import friction_scales
    ua_delta = 6.
    delta_sl_a = 10.
    ta_delta = 293.
    Q_lw = -50.
    delta_sl_o = -1.
    univ_funcs_a = businger
    univ_funcs_o = large_ocean
    uo_delta= 0.
    sf_scheme = "FV free"
    t_expectation = {"Unstable": (294.8, 294.95, 295.025, ),
            "Stable": (290.8, 290.97, 291.03)}
    for to_delta, stability in zip((295., 291.),
            t_expectation.keys()):
        for Q_sw, t_expected in zip((300., 100., 0.),
                t_expectation[stability]):
            SL = friction_scales(ua_delta, delta_sl_a,
                ta_delta, univ_funcs_a,
                uo_delta, delta_sl_o,
                to_delta, univ_funcs_o,
                sf_scheme, Q_sw, Q_lw,
                k=0, absorbed_Qsw_const=False)
            print(f"{stability}: Q_sw={Q_sw}, t_0={SL.t_0} " + \
                    f"instead of {t_expected}")

def fig_colorplotReconstruction():
    fig, axes = plt.subplots(4,2)
    fig.subplots_adjust(hspace=0.67)
    ignore_cached=False
    dt_oce = 90. # oceanic time step
    dt_atm = 30. # atmosphere time step
    number_of_days = 2.3
    T = 86400 * number_of_days # length of the time window
    for delta_sl_o, ax, sf_scheme_o in zip((.0, -.5),
            (axes[:, 0], axes[:, 1]), ("FV test", "FV free")):
        states_atm, states_oce, za, zo = \
            memoised(simulation_coupling, dt_oce, dt_atm, T, True,
                    sf_scheme_a="FD pure", sf_scheme_o=sf_scheme_o,
                    ignore_cached=ignore_cached,
                    delta_sl_o=delta_sl_o, NUMBER_SCHWARZ_ITERATION=1)
        state_atm = states_atm[0]
        state_oce = states_oce[1]
        t = np.linspace(0, number_of_days,
                states_atm[0].t_star.shape[0])
        # ax[3].plot(t, states_atm[1].t_star, label="2nd iteration")
        # ax[2].plot(t, states_atm[1].u_star, label="2nd iteration")
        # ax[3].plot(t, states_atm[2].t_star, label="3rd iteration")
        # ax[2].plot(t, states_atm[2].u_star, label="3rd iteration")
        # ax[3].plot(t, states_atm[3].t_star, label="4th iteration")
        # ax[2].plot(t, states_atm[3].u_star, label="4th iteration")
        ax[3].plot(t, states_atm[0].t_star, label="1st iteration")
        ax[2].plot(t, states_atm[0].u_star, label="1st iteration")
        ax[2].legend()
        ax[2].set_title(r"$u*$")
        ax[3].set_title(r"$t*$, $\delta_o=$"+str(delta_sl_o))
        ax[3].set_xlabel("days")
        all_u = np.real(np.array(state_atm.other["all_u"]))
        all_tke = np.real(np.array(state_atm.other["all_tke"]))
        all_t = np.array(state_atm.other["all_theta"])
        all_uo = np.real(np.array(state_oce.other["all_u"]))
        all_tkeo = np.real(np.array(state_oce.other["all_tke"]))
        all_to = np.array(state_oce.other["all_theta"])
        N_plot = 45
        ua = np.zeros((N_plot+1, all_u.shape[1]))
        ta = np.zeros((N_plot+1, all_t.shape[1]))
        tke= np.zeros((N_plot+1, all_tke.shape[1]))
        uo = np.zeros((N_plot+1, all_uo.shape[1]))
        to = np.zeros((N_plot+1, all_to.shape[1]))
        tke_o= np.zeros((N_plot+1, all_tkeo.shape[1]))
        for i in range(all_u.shape[1]):
            ua[:, i] = projection(all_u[:, i], N_plot)
        for i in range(all_t.shape[1]):
            ta[:, i] = projection(all_t[:, i], N_plot)
        for i in range(all_tke.shape[1]):
            tke[:,i] = projection(all_tke[:, i], N_plot)
        for i in range(all_uo.shape[1]):
            uo[:, i] = projection(all_uo[:, i], N_plot)
        for i in range(all_to.shape[1]):
            to[:, i] = projection(all_to[:, i], N_plot)
        for i in range(all_tkeo.shape[1]):
            tke_o[:,i] = projection(all_tkeo[:, i], N_plot)
        ########## pcolormesh
        x = np.linspace(0., T/86400, N_plot+2) # u.shape+1
        Xu, Yu = np.meshgrid(half_to_full(za, ocean=False), x)
        Xt, Yt = np.meshgrid(half_to_full(za, ocean=False), x)
        z_tke = full_to_half(state_atm.other["z_tke"])
        Xtke, Ytke = np.meshgrid(np.concatenate(([0.], z_tke,
            [state_atm.other["z_tke"][-1]])) , x)
        umin = -1.
        umax = 6.
        tmin = 278.
        tmax = 281.5
        tkemin = 0.
        tkemax = 0.2
        col_u = ax[0].pcolormesh(Yu, Xu, ua, vmin=umin,
                vmax=umax, cmap="seismic", shading='flat')
        col_t = ax[1].pcolormesh(Yt, Xt, ta, vmin=tmin,
                vmax=tmax, cmap="seismic", shading='flat')
        # col_tke = ax[2].pcolormesh(Ytke, Xtke, tke, vmin=tkemin,
        #         vmax=tkemax, cmap="seismic", shading='flat')

        Xuo, Yuo = np.meshgrid(half_to_full(zo, ocean=True), x)
        Xto, Yto = np.meshgrid(half_to_full(zo, ocean=True), x)
        z_tke = full_to_half(state_oce.other["z_tke"])
        # Xtkeo, Ytkeo = np.meshgrid(np.concatenate(([0.], z_tke,
        #     [state_oce.other["z_tke"][-1]])), x)

        ax[0].pcolormesh(Yuo, Xuo, uo, vmin=umin,
                vmax=umax, cmap="seismic", shading='flat')
        ax[1].pcolormesh(Yto, Xto, to, vmin=tmin,
                vmax=tmax, cmap="seismic", shading='flat')
        # ax[2].pcolormesh(Ytkeo, Xtkeo, tke_o, vmin=tkemin,
        #         vmax=tkemax, cmap="seismic", shading='flat')
        fig.colorbar(col_u, ax=ax[0])
        fig.colorbar(col_t, ax=ax[1])
        # fig.colorbar(col_tke, ax=ax[2])
        ax[0].set_title(r"$u$")
        ax[1].set_title(r"$\theta$")
        # ax[2].set_title("tke")
        for ax in ax[:2]:
            ax.set_yscale("symlog", linthresh=5.)

    show_or_save("fig_colorplotParameterizing")

def ustar_comparison(*args, **kwargs): # see star_comparison
    return star_comparison(*args, **kwargs, ustar=True)
def tstar_comparison(*args, **kwargs): # see star_comparison
    return star_comparison(*args, **kwargs, ustar=False)

def star_comparison(ax, sf_scheme_a: str, sf_scheme_o: str,
        vmin: float=None, vmax: float=None, delta_sl_o: float=None,
        delta_sl_a: float=None,
        ignore_cached: bool=False,
        ITERATION: int=1, high_res: bool=False,
        ustar:bool=True, style:str=None, label_atm: bool=True):
    """
    same arguments as in colorplot_coupling. vmax, vmin are unused.
    if ustar=True, plots ustar otherwise plot t_star
    """
    dt_oce = 90. # oceanic time step
    dt_atm = 30. # atmosphere time step
    number_of_days = 3.3
    T = 86400 * number_of_days # length of the time window
    states_atm, states_oce, za, zo = \
        memoised(simulation_coupling, dt_oce, dt_atm, T, True,
                sf_scheme_a=sf_scheme_a, sf_scheme_o=sf_scheme_o,
                ignore_cached=ignore_cached,
                delta_sl_a=delta_sl_a,
                delta_sl_o=delta_sl_o, high_res=high_res,
                NUMBER_SCHWARZ_ITERATION=max(1, ITERATION+1))
    state_atm = states_atm[ITERATION]
    N_plot = 400
    friction_scale = projection(
            state_atm.u_star if ustar else state_atm.t_star,
            N_plot)
    delta_atm = {"FD2": 10.,
            "FD pure": 5., "FV free" : 5.}
    delta_oce = {"FD2": -1.,
            "FD pure": 0., "FV free" : -1., "FV test": 0.}
    if high_res:
        delta_atm = {"FD2": 10./3, "FD pure": 5./3, "FV free" : 5.}
        delta_oce = {"FD2": -1./3,
                "FD pure": 0., "FV free" : -1., "FV test": 0.}

    delta_o =  delta_oce[sf_scheme_o] if delta_sl_o is None \
            else delta_sl_o
    delta_a = delta_atm[sf_scheme_a] if delta_sl_a is None \
            else delta_sl_a
    if label_atm:
        label = sf_scheme_a[:2] +r", $\delta_{sl}=$" + \
                  f"{delta_a:.2f}m"
    else:
        label = f"{sf_scheme_o[:2]}" +r", $\delta_{o}=$" + \
                f"{delta_o:.2f}m"
    if high_res:
        label += " (high res)"
    ax.plot(np.linspace(0, number_of_days, N_plot+1), friction_scale,
            style, label=label)

def fig_referenceCoupling():
    fig, axes2D = plt.subplots(4,2, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.67)
    for axes, high_res in zip((axes2D[:,0], axes2D[:,1]),
            (True, False)):
        all_delta_sl = (-.5/3, 0., -.5, 0.) if high_res \
                else (-.5, 0., -.5, 0.)
        for ax, sf_scheme_o, delta_sl_o in tqdm(zip(axes,
            ("FD pure", "FD pure", "FV free", "FV test"),
            all_delta_sl,
            ), leave=False, total=4):
            fig.colorbar(colorplot_coupling(ax, "FD pure", sf_scheme_o,
                vmin=278., vmax=281.5, ITERATION=1,
                delta_sl_o=delta_sl_o, high_res=high_res,
                ignore_cached=False, label_atm=False), ax=ax)
    for ax in axes2D[:, 0]:
        ax.set_ylabel("z")
        ax.set_ylim(bottom=-1e2, top=1e3)

    axes2D[-1, 0].set_xlabel("days (high resolution)")
    axes2D[-1, 1].set_xlabel("days")
    show_or_save("fig_referenceCoupling")

def fig_colorplotCoupling():
    fig, axes2D = plt.subplots(4,2, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.67)
    for axes, iteration in zip((axes2D[:,0], axes2D[:,1]), (0, 1)):
        for ax, sf_scheme_o, delta_sl_o in tqdm(zip(axes,
            ("FD pure", "FD pure", "FV free", "FV test"), # Oce sf scheme
            (-.5, 0., -.5, 0.)
            ), leave=False, total=4):
            fig.colorbar(colorplot_coupling(ax, "FD pure", sf_scheme_o,
                vmin=278., vmax=281.5, ITERATION=iteration,
                delta_sl_o=delta_sl_o, high_res=False,
                label_atm=False), ax=ax)
    for ax in axes2D[:, 0]:
        ax.set_ylabel("z")
    for ax in axes2D[-1, :]:
        ax.set_xlabel("days")
    show_or_save("fig_colorplotCoupling")

def fig_alpha_sl():
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    fig.subplots_adjust(left=0.182, bottom=0.32, right=0.65)
    h_1_2 = 200
    for z_u_exp, color in zip((-4, -3, -2, -1), palette()):
        z_u = 10**(z_u_exp)
        delta_sl = np.linspace(1e-10, h_1_2)
        log = np.log(1+delta_sl/z_u)
        alpha_sl = ((h_1_2 + z_u) * log - delta_sl) / log / h_1_2
        ax.plot(delta_sl/h_1_2, alpha_sl, color=color,
                label=r"$z_u=10^{"+str(z_u_exp)+r"}$")
    ax.set_xlabel(r"$\delta_{sl} / z_1$")
    ax.set_ylabel(r"$\alpha_{sl}$")
    fig.legend(loc="center right")
    show_or_save("fig_alpha_sl")

def fig_consistency_comparisonUnstable():

    setting_FVfree = {'sf_scheme': 'FV free',
                'delta_sl_lr': 5.,
                'delta_sl_hr': 5.,
                }
    setting_FVNishizawa = {'sf_scheme': 'FVNishizawa',
                'delta_sl_lr': 9.999,
                'delta_sl_hr': 9.999/3,
                }
    setting_FV2 = {'sf_scheme': 'FV2',
                'delta_sl_lr': 10.,
                'delta_sl_hr': 10./3,
                }
    setting_FVpure = {'sf_scheme': 'FV pure',
                'delta_sl_lr': 5.,
                'delta_sl_hr': 5./3,
                }
    setting_FDpure = {'sf_scheme': 'FD pure',
                'delta_sl_lr': 5.,
                'delta_sl_hr': 5./3,
                }
    settings_plot = settings_plot_sf_scheme(\
            z_levels=np.array([0,10]))
    settings = (setting_FVpure, setting_FVNishizawa,
            setting_FV2, setting_FDpure,
            setting_FVfree)

    dt: float = 30.
    number_of_days = 4.2
    u_G: float = 8.
    N: int = int(number_of_days*24*3600 / dt)
    T: float = dt*N
    days_simu = np.linspace(0, T/3600/24, N)
    days_plot = np.linspace(0, T/3600/24, 300)
    fig, axd = plt.subplot_mosaic([['ustar', 'u', 'theta'],
                                   ['tstar', 'u', 'theta'],
                           ['ustar-diff', 'u-diff', 'theta-diff'],
                           ['tstar-diff', 'u-diff', 'theta-diff']],
            figsize=(7.5, 6.5))
    axesAbsolute = [axd[k] for k in ('ustar', 'tstar', 'u', 'theta')]
    axes = [axd[k] for k in ('ustar-diff', 'tstar-diff', 'u-diff', 'theta-diff')]
    fig.subplots_adjust(left=0.108, right=0.9,
            wspace=0.55, hspace=0.68, top=0.95)
    for setting in settings:
        sf_scheme = setting["sf_scheme"]
        delta_sl_lr = setting["delta_sl_lr"]
        delta_sl_hr = setting["delta_sl_hr"]
        state_atm_lr, z_fv_lr = memoised(\
                figures_unstable.simulation_unstable, dt, T,
                False, sf_scheme, delta_sl_lr, high_res=False,
                u_G=u_G)
        state_atm_hr, z_fv_hr = memoised(\
                figures_unstable.simulation_unstable, dt, T,
                False, sf_scheme, delta_sl_hr, high_res=True,
                u_G=u_G)

        plot_args = settings_plot[sf_scheme]
        plot_args.pop("delta_sl")
        plot_args.pop("sf_scheme")

        u_hr = undersample(state_atm_hr.last_tstep["u"],
                z_fv_hr, z_fv_lr)
        t_hr = undersample(state_atm_hr.last_tstep["theta"],
                z_fv_hr, z_fv_lr)
        u_lr= state_atm_lr.last_tstep["u"]
        t_lr= state_atm_lr.last_tstep["theta"]

        ustar_lowres = undersample(state_atm_lr.u_star,
                days_simu, days_plot)
        ustar_highres = undersample(state_atm_hr.u_star,
                days_simu, days_plot)

        tstar_lowres = undersample(state_atm_lr.t_star,
                days_simu, days_plot)
        tstar_highres = undersample(state_atm_hr.t_star,
                days_simu, days_plot)

        diff_ustar = np.abs(ustar_lowres - ustar_highres) / \
                np.abs(ustar_lowres)
        diff_tstar = np.abs(tstar_lowres - tstar_highres)
        axes[2].semilogx(np.abs(u_lr - u_hr) / \
                            np.abs(u_lr), z_fv_lr, **plot_args)
        plot_args.pop("label")
        axes[3].semilogx(np.abs(t_lr-t_hr), z_fv_lr, **plot_args)
        axesAbsolute[2].plot(np.abs(u_lr), z_fv_lr, **plot_args)
        axesAbsolute[3].plot(t_lr, z_fv_lr, **plot_args)
        plot_args.pop("marker", None)
        axes[0].plot(days_plot, diff_ustar, **plot_args)
        axes[1].plot(days_plot, diff_tstar, **plot_args)
        axesAbsolute[0].plot(days_plot, ustar_lowres, **plot_args)
        axesAbsolute[1].plot(days_plot, tstar_lowres, **plot_args)

    axes[0].set_xlim(left=2.2, right=4.2)
    axes[1].set_xlim(left=2.2, right=4.2)
    axes[0].set_ylim(bottom=0., top=0.036)
    axes[1].set_ylim(bottom=0.)
    axes[1].set_xlabel("Time (days)")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel(r"Relative $u_\star$ difference")
    axes[1].set_ylabel(r"$t_\star$ difference")
    axes[3].set_xlabel(r"$\theta$ difference (K)")
    axes[3].set_ylabel(r"$z$ (m)")
    axes[3].set_xlim(right=0.13, left=1e-4)
    axes[3].set_ylim(top=300., bottom=0.)

    axes[2].set_xlabel(r"Relative $u$ difference")
    axes[2].set_ylabel(r"$z$ (m)")
    axes[2].set_xlim(right=0.04, left=1e-3)
    axes[2].set_ylim(top=300., bottom=0.)
    fig.legend(loc=(0.78, 0.65))

    ###### axesAbsolute legend, {x,y}lim
    axesAbsolute[1].set_xlabel("Time (days)")
    axesAbsolute[0].set_xlabel("Time (days)")
    axesAbsolute[0].set_ylabel(r"$u_\star$")
    axesAbsolute[1].set_ylabel(r"$t_\star$")
    axesAbsolute[3].set_xlabel(r"$\theta$ (K)")
    axesAbsolute[2].set_xlabel(r"$||u|| \;({\rm m}.{\rm s}^{-1})$")
    axesAbsolute[2].set_ylabel(r"$z$ (m)")
    axesAbsolute[3].set_ylabel(r"$z$ (m)")
    axesAbsolute[0].set_xlim(left=2.2, right=4.2)
    axesAbsolute[0].set_ylim(bottom=0.20, top=0.24)
    axesAbsolute[1].set_xlim(left=2.2, right=4.2)
    axesAbsolute[2].set_xlim(left=6., right=8.7)
    axesAbsolute[2].set_ylim(top=300., bottom=0.)
    axesAbsolute[3].set_ylim(top=300., bottom=0.)

    show_or_save("fig_consistency_comparisonUnstable")

def fig_consistency_comparisonCoupled():

    setting_FVfree = {'sf_scheme': 'FV free',
                'delta_sl_lr': 5.,
                'delta_sl_hr': 5.,
                }
    setting_FVNishizawa = {'sf_scheme': 'FVNishizawa',
                'delta_sl_lr': 9.999,
                'delta_sl_hr': 9.999/3,
                }
    setting_FV2 = {'sf_scheme': 'FV2',
                'delta_sl_lr': 10.,
                'delta_sl_hr': 10./3,
                }
    setting_FVpure = {'sf_scheme': 'FV pure',
                'delta_sl_lr': 5.,
                'delta_sl_hr': 5./3,
                }
    setting_FDpure = {'sf_scheme': 'FD pure',
                'delta_sl_lr': 5.,
                'delta_sl_hr': 5./3,
                }
    settings_plot = settings_plot_sf_scheme(\
            z_levels=np.array([0,10]))
    settings = (setting_FVpure, setting_FVNishizawa,
            setting_FV2, setting_FDpure,
            setting_FVfree)

    dt: float = 30.
    number_of_days = 4.
    u_G: float = 8.
    N: int = int(number_of_days*24*3600 / dt)
    T: float = dt*N
    days_simu = np.linspace(0, T/3600/24, N)
    days_plot = np.linspace(0, T/3600/24, 300)
    fig, axd = plt.subplot_mosaic([['ustar', 'u', 'theta'],
                                   ['tstar', 'u', 'theta'],
                           ['ustar-diff', 'u-diff', 'theta-diff'],
                           ['tstar-diff', 'u-diff', 'theta-diff']],
            figsize=(7.5, 6.5))
    axesAbsolute = [axd[k] for k in ('ustar', 'tstar', 'u', 'theta')]
    axes = [axd[k] for k in ('ustar-diff', 'tstar-diff', 'u-diff', 'theta-diff')]
    fig.subplots_adjust(left=0.108, right=0.9,
            wspace=0.55, hspace=0.68, top=0.95)

    for setting in settings:
        sf_scheme = setting["sf_scheme"]
        delta_sl_lr = setting["delta_sl_lr"]
        delta_sl_hr = setting["delta_sl_hr"]
        state_atm_lr, _, z_fv_lr, _ = memoised(\
                simulation_coupling, dt, dt, T,
                False, sf_scheme, "FD pure", 0., delta_sl_lr,
                high_res=False)
        state_atm_hr, _, z_fv_hr, _ = memoised(\
                simulation_coupling, dt, dt, T,
                False, sf_scheme, "FD pure", 0., delta_sl_hr,
                high_res=True)
        state_atm_lr = state_atm_lr[-1]
        state_atm_hr = state_atm_hr[-1]

        plot_args = settings_plot[sf_scheme]
        plot_args.pop("delta_sl")
        plot_args.pop("sf_scheme")

        u_hr = undersample(state_atm_hr.last_tstep["u"],
                z_fv_hr, z_fv_lr)
        t_hr = undersample(state_atm_hr.last_tstep["theta"],
                z_fv_hr, z_fv_lr)
        u_lr= state_atm_lr.last_tstep["u"]
        t_lr= state_atm_lr.last_tstep["theta"]

        ustar_lowres = undersample(state_atm_lr.u_star,
                days_simu, days_plot)
        ustar_highres = undersample(state_atm_hr.u_star,
                days_simu, days_plot)

        tstar_lowres = undersample(state_atm_lr.t_star,
                days_simu, days_plot)
        tstar_highres = undersample(state_atm_hr.t_star,
                days_simu, days_plot)

        diff_ustar = np.abs(ustar_lowres - ustar_highres) / \
                np.abs(ustar_lowres)
        diff_tstar = np.abs(tstar_lowres - tstar_highres)
        axes[2].semilogx(np.abs(u_lr - u_hr) / \
                            np.abs(u_lr), z_fv_lr, **plot_args)
        plot_args.pop("label")
        axes[3].plot(np.abs(t_lr-t_hr), z_fv_lr, **plot_args)
        axesAbsolute[2].plot(np.abs(u_lr), z_fv_lr, **plot_args)
        axesAbsolute[3].plot(t_lr, z_fv_lr, **plot_args)
        plot_args.pop("marker", None)
        axes[0].plot(days_plot, diff_ustar, **plot_args)
        axes[1].plot(days_plot, diff_tstar, **plot_args)
        axesAbsolute[0].plot(days_plot, ustar_lowres, **plot_args)
        axesAbsolute[1].plot(days_plot, tstar_lowres, **plot_args)

    axes[0].set_xlim(left=2.2, right=4.)
    axes[1].set_xlim(left=2.2, right=4.)
    axes[0].set_ylim(bottom=0., top=0.036)
    axes[1].set_ylim(bottom=0., top=0.0005)
    axes[1].set_xlabel("Time (days)")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel(r"Relative $u_\star$ difference")
    axes[1].set_ylabel(r"$t_\star$ difference")
    axes[3].set_xlabel(r"$\theta$ difference (K)")
    axes[3].set_ylabel(r"$z$ (m)")
    axes[3].set_xlim(right=0.075, left=0.)
    axes[3].set_ylim(top=300., bottom=0.)

    axes[2].set_xlabel(r"Relative $u$ difference")
    axes[2].set_ylabel(r"$z$ (m)")
    axes[2].set_xlim(right=0.14, left=2e-3)
    axes[2].set_ylim(top=300., bottom=0.)
    fig.legend(loc=(0.78, 0.65))

    ###### axesAbsolute legend, {x,y}lim
    axesAbsolute[1].set_xlabel("Time (days)")
    axesAbsolute[0].set_xlabel("Time (days)")
    axesAbsolute[0].set_ylabel(r"$u_\star$")
    axesAbsolute[1].set_ylabel(r"$t_\star$")
    axesAbsolute[3].set_xlabel(r"$\theta$ (K)")
    axesAbsolute[2].set_xlabel(r"$||u|| \;({\rm m}.{\rm s}^{-1})$")
    axesAbsolute[2].set_ylabel(r"$z$ (m)")
    axesAbsolute[3].set_ylabel(r"$z$ (m)")
    axesAbsolute[0].set_xlim(left=2.2, right=4.)
    axesAbsolute[0].set_ylim(bottom=0.205, top=0.22)
    axesAbsolute[1].set_xlim(left=2.2, right=4.)
    axesAbsolute[1].set_ylim(bottom=-0.003, top=0.005)
    axesAbsolute[2].set_xlim(left=4., right=9.1)
    axesAbsolute[2].set_ylim(top=300., bottom=0.)
    axesAbsolute[3].set_ylim(top=300., bottom=0.)

    show_or_save("fig_consistency_comparisonCoupled")


def ustar_u_low_and_high_res_neutral(sf_scheme: str, u_G: float):
    z_levels: np.ndarray = np.copy(IFS_z_levels_stratified)
    z_levels_high_res: np.ndarray = oversample(z_levels, 3)
    dt: float = 30.
    T: float = 24 * 3600.
    N: int = int(T/dt)
    delta_sl_lr: float = z_levels[1]/2
    delta_sl_hr: float = z_levels_high_res[1]/2
    if sf_scheme == "FV free":
        delta_sl_lr = delta_sl_hr = z_levels[1]/2
    elif sf_scheme == "FV2":
        delta_sl_lr: float = z_levels[1]
        delta_sl_hr: float = z_levels_high_res[1]
    elif sf_scheme == "FVNishizawa":
        delta_sl_lr: float = z_levels[1]*0.999
        delta_sl_hr: float = z_levels_high_res[1]*0.999

    z_lr, u_lr, _, z_tke_lr, TKE_lr, ustar_lr = \
            compute_with_sfNeutral(sf_scheme,
                    z_levels, dt, N, delta_sl_lr, u_G)
    z_hr, u_hr, _, z_tke_hr, TKE_hr, ustar_hr = \
            compute_with_sfNeutral(sf_scheme,
                    z_levels_high_res, dt, N,
                    delta_sl_hr, u_G)
    return ustar_lr, z_lr, u_lr, ustar_hr, z_hr, u_hr

def ustar_u_t_low_and_high_res(sf_scheme: str, u_G: float):
    z_levels: np.ndarray = np.copy(IFS_z_levels_stratified)
    z_levels_high_res: np.ndarray = oversample(z_levels, 3)
    dt: float = 30.
    T: float = 3*24 * 3600.
    N: int = int(T/dt)
    stable: bool = True
    delta_sl_lr: float = z_levels[1]/2
    delta_sl_hr: float = z_levels_high_res[1]/2
    if sf_scheme == "FV free":
        delta_sl_lr = delta_sl_hr = z_levels[1]/2
    elif sf_scheme == "FV2":
        delta_sl_lr: float = z_levels[1]
        delta_sl_hr: float = z_levels_high_res[1]
    elif sf_scheme == "FVNishizawa":
        delta_sl_lr: float = z_levels[1]*0.999
        delta_sl_hr: float = z_levels_high_res[1]*0.999

    z_lr, u_lr, theta_lr, z_tke_lr, TKE_lr, ustar_lr, _ = \
            compute_with_sfStratified(sf_scheme,
                    z_levels, dt, N, stable, delta_sl_lr, u_G)
    z_hr, u_hr, theta_hr, z_tke_hr, TKE_hr, ustar_hr, _ = \
            compute_with_sfStratified(sf_scheme,
                    z_levels_high_res, dt, N, stable,
                    delta_sl_hr, u_G)
    return ustar_lr, z_lr, u_lr, theta_lr, ustar_hr, z_hr, u_hr, theta_hr

def fig_sensitivity_delta_sl():
    # simulation neutre avec plusieurs delta_sl différents
    T = 3600 * 24
    dt = 30.
    N = int(T/dt)
    colors = palette()
    all_settings = (
            {'delta_sl': 5.,
                    "linewidth": 1.8,
                    "color": colors[0],
                    "label": r"$\delta_{sl} = 5 \;{\rm m}$",
                    },
            {'delta_sl': 10.,
                    "linewidth": 1.8,
                    "color": colors[1],
                    "label": r"$\delta_{sl} = 10 \;{\rm m}$",
                    },
            {'delta_sl': 20.,
                    "linewidth": 1.8,
                    "color": colors[2],
                    "label": r"$\delta_{sl} = 20 \;{\rm m}$",
                    },
            )
    fig, axd = plt.subplot_mosaic([['.', 'abs', 'angle', '.'],
                                ['abs_low', '.', '.', 'angle_low']],
            figsize=(7.5, 3.5))
    fig.subplots_adjust(hspace=0.27, right=0.98, top=0.95,
            left=0.07, wspace=0.038, bottom=0.12)
    axes = [axd[k] for k in ('abs', 'angle')]
    axes_zoom = [axd[k] for k in ('abs_low', 'angle_low')]
    for settings in all_settings:
        z_fv, u_fv, _, _, _, _ = \
                memoised(memoisable_compute_sfNeutral,
                        sf_scheme="FV free",
                        delta_sl=settings.pop("delta_sl"),
                        dt=dt, N=N)
        axes[0].plot(np.abs(u_fv), z_fv, **settings)
        settings.pop("label")
        axes_zoom[0].plot(np.abs(u_fv), z_fv, **settings)
        axes[1].plot(np.angle(u_fv), z_fv, **settings)
        axes_zoom[1].plot(np.angle(u_fv), z_fv, **settings)
    axes[0].set_xlabel(r"$||u||\;({\rm m}.{\rm s}^{-1})$")
    axes[1].set_xlabel(r"Arg$(u)$ (rad)")
    axes[0].set_ylabel(r"$z\;({\rm m})$")
    # axes[1].set_ylabel(r"$z\;({\rm m})$")
    axes[1].tick_params('y', labelleft=False)
    axes[0].set_ylim(top=200, bottom=30.)
    axes[1].set_ylim(top=200, bottom=30.)
    axes[0].set_xlim(left=5.48, right=8.82)
    axes[1].set_xlim(right=0.47, left=-0.05)

    # axes zoom:
    axes_zoom[0].set_xlabel(r"$||u||\;({\rm m}.{\rm s}^{-1})$")
    axes_zoom[1].set_xlabel(r"Arg$(u)$ (rad)")
    axes_zoom[0].set_ylabel(r"$z\;({\rm m})$")
    axes_zoom[1].set_ylabel(r"$z\;({\rm m})$")
    axes_zoom[0].set_ylim(top=30, bottom=0.)
    axes_zoom[1].set_ylim(top=30, bottom=0.)
    axes_zoom[0].set_xlim(right=6., left=3.25)
    axes_zoom[1].set_xlim(right=0.5001, left=0.467)
    fig.legend(loc=(0.45, 0.2))
    show_or_save("fig_sensitivity_delta_sl")

def fig_consistency_comparisonNeutral():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    u_G = 8.

    fig, axd = plt.subplot_mosaic([['ustar', 'u', 'u-diff'],
                                    ['.',    'u', 'u-diff']],
            figsize=(7.5, 4.5))
    axes = [axd[k] for k in ('ustar', 'u', 'u-diff')]

    fig.subplots_adjust(hspace=0., right=0.95, top=0.95,
            wspace=0.57)
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":1.5, **kwargs}

    z_levels = np.copy(IFS_z_levels_stratified)
    dic_settings = settings_plot_sf_scheme(z_levels)
    all_settings = ( dic_settings["FV pure"],
                    dic_settings["FVNishizawa"],
                    dic_settings["FV2"],
                    dic_settings["FD pure"],
                    dic_settings["FV free"])

    for settings in all_settings:
        sf_scheme = settings.pop("sf_scheme")
        delta_sl = settings.pop("delta_sl")
        dt: float = 30.
        T: float = 24 * 3600.
        N: int = int(T/dt)
        hours_simu = np.linspace(0, T/3600, N)
        hours_plot = np.linspace(0, T/3600, 30)
        ustar_lowres, z_lr, u_lr, ustar_highres, z_hr, u_hr = \
                memoised(ustar_u_low_and_high_res_neutral,
                        sf_scheme, u_G)
        #projection:
        u_hr = undersample(u_hr, z_hr, z_lr)
        # abs plot:
        axes[2].semilogx((np.abs(u_lr - u_hr))/np.abs(u_hr),
                z_lr, **settings)
        settings.pop("label")

        # temperature plot:
        axes[1].plot(np.abs(u_lr), z_lr, **settings)

        # ustar plot:
        ustar_highres = undersample(ustar_highres, hours_simu,
                hours_plot)
        ustar_lowres = undersample(ustar_lowres, hours_simu,
                hours_plot)
        settings.pop("marker", None)
        axes[0].plot(hours_plot, (np.abs(ustar_lowres - \
                ustar_highres))/ustar_lowres,
                **settings)
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel(r"Relative $u_\star$ difference")
    axes[1].set_xlabel(r"$||u||\;({\rm m}.{\rm s}^{-1})$")
    axes[1].set_ylabel(r"$z$ (m)")
    axes[2].set_xlabel(r"Relative $u$ difference")
    axes[2].set_ylabel(r"$z$ (m)")

    axes[0].set_ylim(top=0.125, bottom=0.)
    axes[1].set_ylim(top=220., bottom=0.)
    axes[2].set_ylim(top=220., bottom=0.)
    axes[2].set_xlim(left=3e-4, right=0.14)
    # axes[2].set_xlim(left=1e-4, right=1.4)

    fig.legend(loc=(0.12, 0.12))
    show_or_save("fig_consistency_comparisonNeutral")

def fig_Stratified():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    u_G = 8.
    z_levels = oversample(IFS_z_levels_stratified, 3)

    fig, axd = plt.subplot_mosaic([['ustar', 'u', 'theta'],
                                    ['.',    'u', 'theta']],
            figsize=(7.5, 4.5))
    axes = [axd[k] for k in ('ustar', 'u', 'theta')]


    fig.subplots_adjust(hspace=0., right=0.95, top=0.95,
            wspace=0.57)
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":1.5, **kwargs}

    dic_settings = settings_plot_sf_scheme(z_levels)
    all_settings = ( dic_settings["FV pure"],
                    dic_settings["FVNishizawa"],
                    dic_settings["FV2"],
                    dic_settings["FD pure"],
                    dic_settings["FV free"])

    for settings in all_settings:
        sf_scheme = settings.pop("sf_scheme")
        delta_sl = settings.pop("delta_sl")
        dt: float = 30.
        T: float = 3*24 * 3600.
        N: int = int(T/dt)
        hours_simu = np.linspace(0, T/3600, N)
        hours_plot = np.linspace(0, T/3600, 30)
        ustar_lowres, z_lr, u_lr, t_lr, *_, = memoised(\
                ustar_u_t_low_and_high_res, sf_scheme, u_G)
        # abs plot:
        axes[1].plot(np.abs(u_lr), z_lr, **settings)
        settings.pop("label")

        # temperature plot:
        axes[2].plot(t_lr, z_lr, **settings)

        # ustar plot:
        ustar_lowres = undersample(ustar_lowres, hours_simu, hours_plot)
        settings.pop("marker", None)
        axes[0].plot(hours_plot, ustar_lowres, **settings)
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel(r"$u_\star$")
    axes[1].set_xlabel(r"$||u||\;({\rm m}.{\rm s}^{-1})$")
    axes[1].set_ylabel(r"$z$ (m)")
    axes[2].set_xlabel(r"$\theta$ (K)")
    axes[2].set_ylabel(r"$z$ (m)")

    axes[0].set_xlim(left=0., right=72.)
    axes[0].set_ylim(top=0.24, bottom=0.172)

    axes[1].set_xlim(left=5., right=9.2)
    axes[1].set_ylim(top=280., bottom=0.)

    axes[2].set_xlim(left=258., right=268)
    axes[2].set_ylim(top=280., bottom=0.)

    fig.legend(loc=(0.12,0.12))

    show_or_save("fig_Stratified")

def fig_consistency_comparisonStratified():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    u_G = 8.
    z_levels = np.copy(IFS_z_levels_stratified)
    z_levels_les= oversample(z_levels, 3)

    fig, axd = plt.subplot_mosaic([['ustar', 'u', 'theta'],
                                    ['.',    'u', 'theta']],
            figsize=(7.5, 4.5))
    axes = [axd[k] for k in ('ustar', 'u', 'theta')]

    fig.subplots_adjust(hspace=0., right=0.95, top=0.95,
            wspace=0.57)
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":1.5, **kwargs}

    dic_settings = settings_plot_sf_scheme(z_levels)
    all_settings = ( dic_settings["FV pure"],
                    dic_settings["FVNishizawa"],
                    dic_settings["FV2"],
                    dic_settings["FD pure"],
                    dic_settings["FV free"])

    for settings in all_settings:
        sf_scheme = settings.pop("sf_scheme")
        delta_sl = settings.pop("delta_sl")
        dt: float = 30.
        T: float = 3*24 * 3600.
        N: int = int(T/dt)
        hours_simu = np.linspace(0, T/3600, N)
        hours_plot = np.linspace(0, T/3600, 30)
        ustar_lowres, z_lr, u_lr, t_lr, ustar_highres, z_hr, u_hr, \
                t_hr = memoised(ustar_u_t_low_and_high_res, sf_scheme, u_G)
        #projection:
        u_hr = undersample(u_hr, z_hr, z_lr)
        t_hr = undersample(t_hr, z_hr, z_lr)
        # abs plot:
        axes[1].semilogx(np.abs(u_lr - u_hr) / \
                                np.abs(u_hr), z_lr, **settings)
        settings.pop("label")

        # temperature plot:
        axes[2].semilogx(np.abs(t_lr - t_hr), z_lr, **settings)

        # ustar plot:
        ustar_highres = undersample(ustar_highres, hours_simu, hours_plot)
        ustar_lowres = undersample(ustar_lowres, hours_simu, hours_plot)
        settings.pop("marker", None)
        axes[0].plot(hours_plot, (np.abs(ustar_lowres - \
                ustar_highres))/ustar_lowres,
                **settings)
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel(r"Relative $u_\star$ difference")
    axes[1].set_xlabel(r"Relative $u$ difference")
    axes[1].set_ylabel(r"$z$ (m)")
    axes[2].set_xlabel(r"$\theta$ difference")
    axes[2].set_ylabel(r"$z$ (m)")

    axes[0].set_ylim(top=0.125, bottom=0.)
    axes[1].set_ylim(top=220., bottom=0.)
    axes[1].set_xlim(left=9e-4, right=0.09)
    axes[2].set_xlim(left=1e-4, right=1.4)
    axes[2].set_ylim(top=220., bottom=0.)

    fig.legend(loc=(0.12, 0.12))
    show_or_save("fig_consistency_comparisonStratified")

def compute_with_sfStratified(sf_scheme, z_levels, dt=10., N=3240,
        stable=True, delta_sl=None, z_constant=None, u_G=8.):
    """
    return z_fv, u_fv, theta_fv, z_tke, TKE, ustar
    """
    if delta_sl is None:
        print("warning: no delta_sl entered")
        delta_sl = z_levels[1]/2
    if z_constant is None:
        z_constant = 2*delta_sl

    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=u_G,
            K_mol=1.5e-5, f=1.39e-4)
    T0 = 265.
    u_0 = u_G*np.ones(M) + 0j
    phi_0 = np.zeros(M+1) + 0j
    t_0, dz_theta_0 = simulator.initialize_theta(Neutral_case=False)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    if stable:
        # we used 1 degrees per 4 hour, now it's 1 degree per 10 hours
        SST = np.concatenate(([265],
            [265 - 0.1*(dt*(n-1))/3600. for n in range(1, N+1)]))
    else: # diurnal cycle:
        SST = np.concatenate(([265],
            [265 + 2.*np.sin((dt*(n-1))/3600. * np.pi / 12.)\
                    for n in range(1, N+1)]))

    z_tke = np.copy(simulator.z_full)
    k = bisect.bisect_right(z_levels[1:], delta_sl)
    z_tke[k] = delta_sl #
    u_deltasl = u_G # first guess before the iterations
    t_deltasl = T0 # first guess before the iterations
    Q_sw, Q_lw, delta_sl_o = np.zeros(N+1), np.zeros(N+1), 0.
    u_o, t_o = np.zeros(N+1), SST
    if sf_scheme in {"FV1 free", "FV2 free", "FV free", "FV2"}:
        u_i, phi_i, t_i, dz_theta_i, u_delta_i, t_delta_i = \
                simulator.initialization(u_0, phi_0, t_0, dz_theta_0,
                        delta_sl, u_o[0], t_o[0], Q_sw[0], Q_lw[0],
                        z_constant, delta_sl_o, u_G)
    else:
        u_i, phi_i, t_i, dz_theta_i, u_delta_i, t_delta_i = \
                u_0, phi_0, t_0, dz_theta_0, u_deltasl, t_deltasl

    if sf_scheme[:2] == "FV":
        ret = simulator.FV(u_t0=u_i, phi_t0=phi_i, theta_t0=t_i,
                        delta_sl_o=0.,
                        forcing_theta=np.zeros(simulator.M),
                        dz_theta_t0=dz_theta_i, Q_sw=Q_sw, Q_lw=Q_lw,
                        u_o=u_o, SST=SST, sf_scheme=sf_scheme,
                        u_delta=u_delta_i, t_delta=t_delta_i,
                        forcing=forcing, delta_sl=delta_sl)
        u, phi, tke_full, ustar, temperature, dz_theta, l_eps, SL = \
                [ret[x] for x in ("u", "phi", "tke", "all_u_star",
                    "theta", "dz_theta", "l_eps", "SL")]

        z_fv, u_fv, theta_fv = simulator.reconstruct_FV(u, phi,
                temperature, dz_theta, SL=SL)
        z_tke = simulator.z_full
        return z_fv, u_fv, theta_fv, z_tke, tke_full, ustar, l_eps
    else:
        ret = simulator.FD(u_t0=u_i, theta_t0=t_i,
                        delta_sl_o=0.,
                        forcing_theta=np.zeros(simulator.M),
                        Q_sw=Q_sw, Q_lw=Q_lw,
                        u_o=u_o, SST=SST, sf_scheme=sf_scheme,
                        forcing=forcing)
        u, tke_full, ustar, temperature, l_eps, SL = \
                [ret[x] for x in ("u", "tke", "all_u_star",
                    "theta", "l_eps", "SL")]
        z_tke = simulator.z_full
        z_fd = simulator.z_half[:-1]
        return z_fd, u, temperature, z_tke, tke_full, ustar, l_eps


def compute_with_sfNeutral(sf_scheme, z_levels, dt, N, delta_sl,
        u_G=8., **_):
    """
    return z_fv, u_fv, theta_fv, z_tke, TKE, ustar
    """
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=u_G,
            K_mol=1.5e-5, f=1e-4)
    u_0 = u_G*np.ones(M) + 0j
    phi_0 = np.zeros(M+1) + 0j
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    SST = np.ones(N+1)*265.
    k = bisect.bisect_right(z_levels[1:], delta_sl)
    u_deltasl = u_G # first guess before the iterations
    t_0, dz_theta_0 = 265. * np.ones(M), np.zeros(M+1)

    if sf_scheme in {"FV1 free", "FV2 free", "FV free", "FV2"}:
        u_i, phi_i, t_i, dz_theta_i, u_delta_i, _ = \
                simulator.initialization(u_0, phi_0, t_0, dz_theta_0,
                        delta_sl, 0., 265., 0., 0.,
                        z_levels[k+1], 0., u_G)
    else:
        u_i, phi_i, t_i, dz_theta_i, u_delta_i = \
                u_0, phi_0, t_0, dz_theta_0, u_deltasl

    if sf_scheme[:2] == "FV":
        ret = simulator.FV(u_t0=u_i, phi_t0=phi_i, theta_t0=t_i,
                        delta_sl_o=0.,
                        forcing_theta=np.zeros(simulator.M),
                        dz_theta_t0=dz_theta_i,
                        Q_sw=np.zeros(N+1), Q_lw=np.zeros(N+1),
                        u_o=np.zeros(N+1), Neutral_case=True,
                        SST=SST, sf_scheme=sf_scheme, u_delta=u_delta_i,
                        forcing=forcing, delta_sl=delta_sl)
        u, phi, tke_full, u_star, temperature, dz_theta, SL = \
                [ret[x] for x in ("u", "phi", "tke", "all_u_star",
                    "theta", "dz_theta", "SL")]

        z_fv, u_fv, theta_fv = simulator.reconstruct_FV(u, phi, temperature,
                dz_theta, SL=SL)
        z_tke = simulator.z_full
        return z_fv, u_fv, theta_fv, z_tke, tke_full, ret["all_u_star"]
    else:
        ret = simulator.FD(u_t0=u_i, theta_t0=t_i,
                        delta_sl_o=0.,
                        forcing_theta=np.zeros(simulator.M),
                        Q_sw=np.zeros(N+1), Q_lw=np.zeros(N+1),
                        u_o=np.zeros(N+1), SST=SST, Neutral_case=True,
                        sf_scheme=sf_scheme, forcing=forcing)
        u, tke_full, ustar, temperature = \
                [ret[x] for x in ("u", "tke", "all_u_star", "theta")]
        z_tke = simulator.z_full
        z_fd = simulator.z_half[:-1]
        return z_fd, u, temperature, z_tke, tke_full, ustar

def plot_FVStratified(axes, sf_scheme, dt=10., N=3240,
        z_levels=DEFAULT_z_levels_stratified,
        stable: bool=True, delta_sl=None,
        name=None, style={}):

    z_fv, u_fv, theta_fv, z_tke, TKE, ustar, l_eps = \
            compute_with_sfStratified(sf_scheme, z_levels, dt, N,
                    stable, delta_sl)
    axes[0].semilogy(np.abs(u_fv), z_fv, **style)
    axes[1].semilogy(np.angle(u_fv), z_fv, **style)
    # axes[1].semilogy(theta_fv, z_fv, **style)
    axes[2].semilogy(TKE, z_tke, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)
    k = bisect.bisect_right(z_levels[1:], delta_sl)
    z_leps = np.copy(z_levels)
    z_leps[k] = delta_sl
    axes[4].semilogy(l_eps, z_leps, **style)

def plot_FDStratified(axes, sf_scheme, dt=10., N=3240,
        z_levels=DEFAULT_z_levels_stratified, stable: bool=True,
        name=None, style={}):
    if name is None:
        name = sf_scheme
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=8.,
            K_mol=1.5e-5, f=1.39e-4)
    u_0 = 8*np.ones(M)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    if stable:
        SST = np.concatenate(([265],
            [265 - 0.25*(dt*(n-1))/3600. for n in range(1, N+1)]))
    else: # diurnal cycle:
        SST = np.concatenate(([265],
            [265 + 2.*np.sin((dt*(n-1))/3600. * np.pi / 12.)\
                    for n in range(1, N+1)]))
    theta, _ = simulator.initialize_theta(Neutral_case=False)
    ret = simulator.FD(u_t0=u_0, u_o=np.zeros(N+1),
            forcing_theta=np.zeros(simulator.M),
            theta_t0=theta, Q_sw=np.zeros(N+1),
            delta_sl_o=0.,
            SST=SST, Q_lw=np.zeros(N+1),
            sf_scheme=sf_scheme, forcing=forcing)
    u, TKE, ustar, temperature, l_eps = [ret[x] for x in \
            ("u", "tke", "all_u_star", "theta", "l_eps")]
    z_tke = np.copy(simulator.z_full)
    z_tke[0] = 0.1

    axes[0].semilogy(np.abs(u), simulator.z_half[:-1], **style)
    axes[1].semilogy(np.angle(u), simulator.z_half[:-1], **style)
    # axes[1].semilogy(temperature, simulator.z_half[:-1], **style)
    axes[2].semilogy(TKE, z_tke, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)
    axes[4].semilogy(l_eps, z_tke, **style)

def plot_FD(axes, sf_scheme, dt=60., N=1680,
        z_levels=DEFAULT_z_levels, name=None, style={}):
    if name is None:
        name = sf_scheme
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=10., K_mol=1.5e-5, f=1e-4)

    u_0 = 10*np.ones(M)
    forcing = 1j*simulator.f * simulator.u_g*np.ones((N+1, M))
    SST = 265. * np.ones(N+1) # Neutral SST with theta=const=265.
    ret = simulator.FD(u_t0=u_0,
            forcing_theta=np.zeros(simulator.M),
            u_o=np.zeros(N+1),
            delta_sl_o=0.,
            theta_t0=265*np.ones(M), Q_sw=np.zeros(N+1),
            SST=SST, Q_lw=np.zeros(N+1), Neutral_case=True,
            sf_scheme=sf_scheme, forcing=forcing)
    u, TKE, ustar = [ret[x] for x in ("u", "tke", "all_u_star")]
    z_tke = np.copy(simulator.z_full)

    z_tke = np.copy(simulator.z_full)
    z_tke[0] = z_levels[1]/2 if sf_scheme != "FD2" else 0.1

    axes[0].plot(np.real(u), simulator.z_half[:-1], **style)
    axes[1].plot(np.imag(u), simulator.z_half[:-1], **style)
    axes[2].plot(TKE, z_tke, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)

def plot_FV(axes, sf_scheme, delta_sl, dt=60., N=1680,
        z_levels=DEFAULT_z_levels, name=None, style={}):
    z_fv, u_fv, theta_fv, z_tke, TKE, ustar = \
            compute_with_sfNeutral(sf_scheme, z_levels, dt, N,
                    delta_sl)

    axes[0].semilogy(np.real(u_fv), z_fv, **style)
    axes[1].semilogy(np.imag(u_fv), z_fv, **style)
    axes[2].semilogy(TKE, z_tke, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)

def memoisable_compute_sfNeutral(sf_scheme, delta_sl, dt, N):
    return compute_with_sfNeutral(z_levels=np.copy(IFS_z_levels),
            sf_scheme=sf_scheme, delta_sl=delta_sl, dt=dt, N=N)

def compute_FD_sfNeutral(sf_scheme, dt, N):
    z_levels = IFS_z_levels
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=10., K_mol=1.5e-5, f=1e-4)
    u_0 = 10*np.ones(M) + 0j
    forcing = 1j*simulator.f * simulator.u_g*np.ones((N+1, M))
    SST = 265. * np.ones(N+1) # Neutral SST with theta=const=265.
    ret = simulator.FD(u_t0=u_0,
            forcing_theta=np.zeros(simulator.M),
            u_o=np.zeros(N+1),
            delta_sl_o=0.,
            theta_t0=265*np.ones(M), Q_sw=np.zeros(N+1),
            SST=SST, Q_lw=np.zeros(N+1), Neutral_case=True,
            sf_scheme=sf_scheme, forcing=forcing)
    return z_levels, ret['u']

def settings_plot_sf_scheme(z_levels: np.ndarray):
    """
    Defines the styles for sf schemes comparison figures.
    It is better to use the same style for all figures
    to avoir losing people.
    """
    colors = palette()

    settings_FVfree = {"sf_scheme": "FV free",
            "delta_sl":z_levels[1]/2,
            "linewidth": 1.8,
            "color": colors[1],
            "label": "FV free",
            "linestyle": "dashed"}
    settings_FVNishizawa = {"sf_scheme": "FVNishizawa",
            "delta_sl":z_levels[1]*0.99999,
            "linewidth": 1.8,
            "color": colors[6],
            "label": "FV Nishizawa",
            "linestyle": "dashed"}
    settings_FV1_bug = {"sf_scheme": "FV1 bug",
            "linewidth": 1.,
            "delta_sl":z_levels[1]/2,
            "color": colors[4],
            "label": "FV1, " + r"$K_0=K_{mol}$",
            "linestyle": "dashed"
            }
    settings_FV1 = {"sf_scheme": "FV1",
            "linewidth": 1.8,
            "color": colors[2],
            "delta_sl":z_levels[1]/2,
            "label": "FV1"}
    settings_FVpure = {"sf_scheme": "FV pure",
            "linewidth": 1.8,
            "delta_sl":z_levels[1]/2,
            "color": colors[3],
            "label": "FV pure",
            "linestyle": (0, (4, 4))}
    settings_FV2 = {"sf_scheme": "FV2",
            "linewidth": 1.8,
            "color": colors[0],
            "delta_sl":z_levels[1],
            "label": "FV2"}
    settings_FDpure = {"sf_scheme": "FD pure",
            "label": "Finite Differences",
            "marker":"o",
            "fillstyle":"none",
            "delta_sl":z_levels[1]/2,
            "color": colors[5],
            "linewidth": 0.3}
    ret = {"FV1 bug": settings_FV1_bug,
            "FV1": settings_FV1,
            "FV pure": settings_FVpure,
            "FV2": settings_FV2,
            "FV free": settings_FVfree,
            "FVNishizawa": settings_FVNishizawa,
            "FD pure": settings_FDpure,
            }
    return ret

def fig_neutral_comparisonPlot():
    fig, axes = plt.subplots(1, 2, figsize=(6., 4.))
    fig.subplots_adjust(right=0.675, wspace=0.29)
    T = 3600 * 24
    dt = 30.
    N = int(T/dt)

    dic_settings = settings_plot_sf_scheme(IFS_z_levels)
    all_settings = (
                    dic_settings["FV1 bug"],
                    dic_settings["FV1"],
                    dic_settings["FV pure"],
                    dic_settings["FVNishizawa"],
                    dic_settings["FV2"],
                    dic_settings["FD pure"],
                    dic_settings["FV free"],
                    )

    for settings in all_settings:
        z_fv, u_fv, _, _, _, _ = \
                memoised(memoisable_compute_sfNeutral,
                        sf_scheme=settings["sf_scheme"],
                        delta_sl=settings["delta_sl"],
                        dt=dt, N=N)
        settings.pop("delta_sl")
        settings.pop("sf_scheme")
        axes[0].plot(np.abs(u_fv), z_fv, **settings)
        settings.pop("label")
        axes[1].plot(np.angle(u_fv), z_fv, **settings)

    half_levels = full_to_half(IFS_z_levels)
    axes[0].hlines(half_levels, xmin=0., xmax=100., color="k",
            linestyle="dotted", linewidth=0.6)
    #, label=r"$z_{m+1/2}$")
    axes[1].hlines(half_levels, xmin=0., xmax=10., color="k",
            linestyle="dotted", linewidth=0.6)

    axes[0].set_xlim(left=2.5, right=8.5)
    axes[1].set_xlim(left=0.2, right=.54)


    axes[0].set_xlabel(r"$||u(z)|| \;({\rm m.s}^{-1})$")
    axes[1].set_xlabel(r"$\arg(u(z)) \;({\rm rad})$")
    # now we want to set ticks labels for $z_{1/2}$, ...
    z_half = full_to_half(IFS_z_levels)

    ticks = np.concatenate((z_half[:4], [0, 100]))
    axes[1].set_yticks(ticks)
    labels = [r"$z_\frac{1}{2}$",
            r"$z_\frac{3}{2}$", r"$z_\frac{5}{2}$",
            r"$z_\frac{7}{2}$", "$0$", "$100$"]
    axes[1].set_yticklabels(labels)

    axes[0].set_ylabel(r"$z$ (m)")

    fig.legend(loc="right")
    LOGSCALE = False
    if LOGSCALE:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    else:
        axes[0].set_ylim(bottom=0., top=100.)
        axes[1].set_ylim(bottom=0., top=100.)
    show_or_save("fig_neutral_comparisonPlot")


def fig_consistency_comparison():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels= np.linspace(0, 1500, 41)
    z_levels= np.copy(IFS_z_levels)
    z_levels_les= np.linspace(0, IFS_z_levels[-1], 401)
    z_levels_FV2 = np.concatenate(([0., z_levels[1]/2], z_levels[1:]))
    # for FV with FV interpretation of sf scheme,
    # the first grid level is divided by 2 so that
    # delta_{sl} is the same in all the schemes.
    dt = 60.
    N = 1680 # 28*60=1680

    fig, axes = plt.subplots(1,4, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)
    col_FDpure = "g"
    col_FV1 = "b"
    col_FVfree = "r"
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":1.5, **kwargs}

    # High resolution:
    plot_FD(axes, "FD pure", N=N, dt=dt, z_levels=z_levels_les,
            name="FD, M=400", style=style(col_FDpure))
    plot_FV(axes, "FV1", delta_sl=z_levels_les[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV1, M=400", style=style(col_FV1))
    plot_FV(axes, "FV2", delta_sl=z_levels_les[1],
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV2, M=400", style=style("c"))
    # plot_FV(axes, "FV pure", delta_sl=z_levels_les[1]/2,
    #         N=N, dt=dt, z_levels=z_levels_les,
    #         name="FV pure, M=400", style=style("m"))
    plot_FV(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV free, M=400", style=style(col_FVfree))

    # Low resolution:

    plot_FD(axes, "FD pure", N=N, dt=dt, z_levels=z_levels,
            name="FD, M="+str(z_levels.shape[0] - 1),
            style=style(col_FDpure, "dotted"))
    plot_FV(axes, "FV1", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FV1, "dotted"))
    plot_FV(axes, "FV2", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_FV2,
            name=None, style=style("c", "dotted"))
    # plot_FV(axes, "FV pure", delta_sl=z_levels[1]/2,
    #         N=N, dt=dt, z_levels=z_levels,
    #         name=None, style=style("m", "dotted"))
    plot_FV(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FVfree, "dotted"))

    axes[0].set_ylim(top=1500.)
    axes[1].set_ylim(top=1500.)
    axes[0].set_xlabel("wind speed (u, $m.s^{-1}$)")
    axes[0].set_ylabel("height (m)")
    axes[1].set_xlabel("wind speed (v, $m.s^{-1}$)")
    axes[1].set_ylabel("height (m)")
    axes[2].set_xlabel("energy (J)")
    axes[2].set_ylabel("height (m)")
    axes[3].set_ylabel("friction velocity (u*, $m.s^{-1}$)")
    axes[3].set_ylim(top=0.5, bottom=0.38)
    axes[3].set_xlabel("time (s)")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    show_or_save("fig_consistency_comparison")

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
