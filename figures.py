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
from memoisation import memoised
from atm1DStratified import Atm1dStratified
from ocean1DStratified import Ocean1dStratified
from universal_functions import Businger_et_al_1971 as businger
from utils_linalg import solve_linear
import figures_unstable
from fortran.visu import import_data
from validation_oce1D import fig_comodoParamsConstantCooling
from validation_oce1D import fig_comodoParamsWindInduced
from validation_oce1D import fig_windInduced

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath, amsfonts}"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = ':'
mpl.rcParams["grid.alpha"] = '0.7'
mpl.rcParams["grid.linewidth"] = '0.5'

DEFAULT_z_levels = np.linspace(0, 1500, 41)
DEFAULT_z_levels_stratified = np.linspace(0, 400, 65)
IFS_z_levels = np.flipud(np.array((1600.04, 1459.58, 1328.43,
    1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70,
    566.49, 500.91, 440.58, 385.14, 334.22, 287.51, 244.68,
    205.44, 169.50, 136.62, 106.54, 79.04, 53.92, 30.96,
    10.00))) - 10. # Not really IFS levels, since the list here
# is actually z_half. So the correct way would be to add 0
# and to take the middle of all space step here

IFS_z_levels_stratified = np.flipud(np.array((500.91, 440.58, 385.14,
    334.22, 287.51, 244.68,
    205.44, 169.50, 136.62, 106.54, 79.04, 53.92, 30.96,
    10.00))) - 10. # less levels in the stratified case

def fig_launchOcean():
    PLOT_FOR = True
    dt = 30.
    f = 0.
    T0, alpha, N0 = 16., 0.0002, 0.01
    z_levels = np.linspace(-50., 0., 51)
    simulator_oce = Ocean1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=0., f=f, alpha=alpha,
            N0=N0)

    N_FOR = nb_steps = 3600
    N = N_FOR + 1
    time = dt * np.arange(N+1)
    rho0, cp, Qswmax = 1024., 3985., 0.
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    heatloss = np.zeros(N+1)
    wind_10m = np.ones(N+1) * 2. + 0j
    temp_10m = np.ones(N+1) * 240

    u_current, phi, tke, all_u_star, theta, \
                dz_theta, l_eps, SL, viscosity = simulator_oce.FV(\
            u_t0=u_0, phi_t0=phi_0, theta_t0=theta_0,
            dz_theta_t0=dz_theta_0, solar_flux=srflx,
            heatloss=heatloss, wind_10m=wind_10m,
            temp_10m=temp_10m, sf_scheme="FV test")

    zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
            phi, theta, dz_theta, SL, ignore_loglaw=True)

    #### Getting fortran part ####
    name_file = "fortran/t_final_tke.out"
    ret_for, z_for = import_data(name_file)

    #### Plotting both #####
    fig, ax = plt.subplots(1, 1)
    if PLOT_FOR:
        ax.plot(ret_for, z_for, label="Fortran")
    ax.plot(thetaFV, zFV, "--", label="Python")
    ax.legend()
    show_or_save("fig_launchOcean")


def fig_colorplots_FDlowres():
    """
        plots several (2D) variables on a colorplot.
    """
    figures_unstable.colorplot(IFS_z_levels_stratified, False,
            "FD pure", IFS_z_levels_stratified[1]/2, 1)
    show_or_save("fig_colorplots_FDlowres")

def fig_colorplots_FDhighres():
    """
        plots several (2D) variables on a colorplot.
    """
    z_levels= np.linspace(0, IFS_z_levels_stratified[-1], 351)
    figures_unstable.colorplot(z_levels, False, "FD pure",
            z_levels[1]/2, 3)
    show_or_save("fig_colorplots_FDhighres")

def fig_colorplots_FVhighres():
    """
        plots several (2D) variables on a colorplot.
    """
    z_levels= np.linspace(0, IFS_z_levels_stratified[-1], 351)
    figures_unstable.colorplot(z_levels, True, "FV free",
            IFS_z_levels_stratified[1]/2, 35)
    show_or_save("fig_colorplots_FVhighres")

def fig_colorplots_FVlowres():
    """
        plots several (2D) variables on a colorplot.
    """
    figures_unstable.colorplot(IFS_z_levels_stratified, True,
            "FV free", IFS_z_levels_stratified[1]/2, 3)
    show_or_save("fig_colorplots_FVlowres")

def fig_colorplots_FV2highres():
    """
        plots several (2D) variables on a colorplot.
    """
    z_levels= np.linspace(0, IFS_z_levels_stratified[-1], 351)
    figures_unstable.colorplot(z_levels, True, "FV2",
            z_levels[1], 35)
    show_or_save("fig_colorplots_FVhighres")

def fig_colorplots_FV2lowres():
    """
        plots several (2D) variables on a colorplot.
    """
    z_levels= IFS_z_levels_stratified
    z_levels_FV2 = np.concatenate(([0., z_levels[1]/2], z_levels[1:]))
    figures_unstable.colorplot(z_levels_FV2, True,
            "FV2", z_levels_FV2[1], 3)
    show_or_save("fig_colorplots_FVlowres")


def fig_consistency_comparisonUnstable():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels = DEFAULT_z_levels_stratified
    z_levels = IFS_z_levels_stratified
    z_levels_FV2 = np.concatenate(([0., z_levels[1]/2], z_levels[1:]))
    # z_levels_les= np.linspace(0, 400, 651)
    z_levels_les= np.linspace(0, z_levels[-1], 351)
    dt = 50.
    N = int(3*24*3600/dt) # 28*3600/10=3240

    fig, axes = plt.subplots(1,5, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)
    col_FDpure = "g"
    col_FV1 = "b"
    col_FVfree = "r"
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":1.5, **kwargs}
    # High resolution:
    plot_FDStratified(axes, "FD pure", N=N, dt=dt, z_levels=z_levels_les, stable=False,
            name="FD, M=350", style=style(col_FDpure))
    # plot_FVStratified(axes, "FV1", delta_sl=z_levels_les[1]/2,
    #         N=N, dt=dt, z_levels=z_levels_les, stable=False,
    #         name="FV1, M=350", style=style(col_FV1))
    plot_FVStratified(axes, "FV2", delta_sl=z_levels_les[1],
            N=N, dt=dt, z_levels=z_levels_les, stable=False,
            name="FV2, M=350", style=style("c"))
    # plot_FVStratified(axes, "FV pure", delta_sl=z_levels_les[1]/2,
    #         N=N, dt=dt, z_levels=z_levels_les, stable=False,
    #         name="FV pure, M=350", style=style("m"))
    plot_FVStratified(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_les, stable=False,
            name="FV free, M=350", style=style(col_FVfree))

    # Low resolution:
    plot_FDStratified(axes, "FD pure", N=N, dt=dt, z_levels=z_levels,
            name="FD, M="+str(z_levels.shape[0]),
            style=style(col_FDpure, "dotted"), stable=False)
    # plot_FVStratified(axes, "FV1", delta_sl=z_levels[1]/2,
    #         N=N, dt=dt, z_levels=z_levels, stable=False,
    #         name=None, style=style(col_FV1, "dotted"))
    plot_FVStratified(axes, "FV2", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_FV2, stable=False,
            name=None, style=style("c", "dotted"))
    # plot_FVStratified(axes, "FV pure", delta_sl=z_levels[1]/2,
    #         N=N, dt=dt, z_levels=z_levels, stable=False,
    #         name=None, style=style("m", "dotted"))
    plot_FVStratified(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels, stable=False,
            name=None, style=style(col_FVfree, "dotted"))

    axes[0].set_xlabel(r"wind speed ($|u|, m.s^{-1}$)")
    axes[0].set_ylabel("height (m)")
    axes[1].set_xlabel(r"Potential Temperature ($\theta$, K)")
    axes[1].set_ylabel("height (m)")
    axes[2].set_xlabel("energy (J)")
    axes[2].set_ylabel("height (m)")
    axes[2].legend(loc="upper right")
    axes[3].set_ylim(top=0.28, bottom=0.16)
    axes[3].set_ylabel("friction velocity (u*, $m.s^{-1}$)")
    axes[3].set_xlabel("time (s)")
    axes[4].set_xlabel("mixing length (m)")
    axes[4].set_ylabel("height (m)")
    show_or_save("fig_consistency_comparisonUnstable")

def fig_consistency_comparisonStratified():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels = DEFAULT_z_levels_stratified
    z_levels = IFS_z_levels_stratified
    z_levels_FV2 = np.concatenate(([0., z_levels[1]/2], z_levels[1:]))
    # z_levels_les= np.linspace(0, 400, 651)
    z_levels_les= np.linspace(0, z_levels[-1], 351)
    dt = 10.
    N = 5*650 # 28*3600/10=3240

    fig, axes = plt.subplots(1,5, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)
    col_FDpure = "g"
    col_FV1 = "b"
    col_FVfree = "r"
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":1.5, **kwargs}

    # High resolution:
    plot_FDStratified(axes, "FD pure", N=N, dt=dt, z_levels=z_levels_les,
            name="FD, M=350", style=style(col_FDpure))
    plot_FVStratified(axes, "FV1", delta_sl=z_levels_les[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV1, M=350", style=style(col_FV1))
    plot_FVStratified(axes, "FV2", delta_sl=z_levels_les[1],
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV2, M=350", style=style("c"))
    # plot_FVStratified(axes, "FV pure", delta_sl=z_levels_les[1]/2,
    #         N=N, dt=dt, z_levels=z_levels_les,
    #         name="FV pure, M=350", style=style("m"))
    plot_FVStratified(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV free, M=350", style=style(col_FVfree))

    # Low resolution:
    plot_FDStratified(axes, "FD pure", N=N, dt=dt, z_levels=z_levels,
            name="FD, M="+str(z_levels.shape[0]), style=style(col_FDpure, "dotted"))
    plot_FVStratified(axes, "FV1", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FV1, "dotted"))
    plot_FVStratified(axes, "FV2", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_FV2,
            name=None, style=style("c", "dotted"))
    # plot_FVStratified(axes, "FV pure", delta_sl=z_levels[1]/2,
    #         N=N, dt=dt, z_levels=z_levels,
    #         name=None, style=style("m", "dotted"))
    plot_FVStratified(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FVfree, "dotted"))

    axes[0].set_xlabel(r"wind speed ($|u|, m.s^{-1}$)")
    axes[0].set_ylabel("height (m)")
    axes[1].set_xlabel(r"Potential Temperature ($\theta$, K)")
    axes[1].set_ylabel("height (m)")
    axes[2].set_xlabel("energy (J)")
    axes[2].set_ylabel("height (m)")
    axes[2].legend(loc="upper right")
    axes[3].set_ylim(top=0.28, bottom=0.16)
    axes[3].set_ylabel("friction velocity (u*, $m.s^{-1}$)")
    axes[3].set_xlabel("time (s)")
    axes[4].set_xlabel("mixing length (m)")
    axes[4].set_ylabel("height (m)")
    show_or_save("fig_consistency_comparisonStratified")

def compute_with_sfStratified(sf_scheme, z_levels, dt=10., N=3240,
        stable=True, delta_sl=None, z_constant=None):
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
    z_tke[k] = delta_sl #
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

    u, phi, TKE, dz_tke, ustar, temperature, dz_theta, l_eps, SL = \
            simulator.FV(u_t0=u_0, phi_t0=phi_0,
                    SST=SST, sf_scheme=sf_scheme, u_delta=u_deltasl,
                    forcing=forcing, delta_sl=delta_sl)

    z_fv, u_fv, theta_fv = simulator.reconstruct_FV(u, phi, temperature,
            dz_theta, SL=SL)
    z_tke, tke_fv = simulator.reconstruct_TKE(TKE,
            dz_tke, SL, sf_scheme, businger(), l_eps)
    # z_tke = simulator.z_full
    return z_fv, u_fv, theta_fv, z_tke, tke_fv, ustar, l_eps

def compute_with_sfNeutral(sf_scheme, z_levels, dt, N, delta_sl):
    """
    return z_fv, u_fv, theta_fv, z_tke, TKE, ustar
    """
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=10.,
            K_mol=1e-4, f=1e-4)
    u_0 = 10.*np.ones(M)
    phi_0 = np.zeros(M+1)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    SST = np.ones(N+1)*265.
    z_tke = np.copy(simulator.z_full)
    k = bisect.bisect_right(z_levels[1:], delta_sl)
    z_tke[k] = delta_sl #
    u_deltasl = 10. # first guess before the iterations
    if sf_scheme in {"FV1 free", "FV2 free", "FV free", "FV2"}:
        zk = z_levels[k]
        h_tilde = z_levels[k+1] - delta_sl
        h_kp12 = z_levels[k+1] - z_levels[k]
        z_0M = 1e-1
        u_constant = 10.
        K_mol, kappa = simulator.K_mol, simulator.kappa
        for _ in range(15):
            u_star = kappa / np.log(1+delta_sl/z_0M) * np.abs(u_deltasl)
            z_0M = K_mol / kappa / u_star

            phi_0[k] = u_deltasl / (z_0M+delta_sl) / \
                    np.log(1+delta_sl/z_0M)
            # u_tilde + h_tilde (phi_0 / 6 + phi_1 / 3) = u_constant
            # (subgrid reconstruction at the top of the volume)
            u_tilde = u_constant - h_tilde/6 * phi_0[k]
            u_deltasl = u_tilde - h_tilde / 3 * phi_0[k]

        neutral_tau_sl = (delta_sl / (h_kp12))* \
                (1+z_0M/delta_sl - 1/np.log(1+delta_sl/z_0M) \
                + (zk - (zk+z_0M)*np.log(1+zk/z_0M)) \
                / (delta_sl * np.log(1+delta_sl/z_0M)))

        alpha_sl = h_tilde/h_kp12 + neutral_tau_sl
        u_0[k] = alpha_sl * u_tilde - neutral_tau_sl*h_tilde*phi_0[k]/3

    u, phi, TKE, dz_tke, ustar, temperature, dz_theta, l_eps, SL = \
            simulator.FV(u_t0=u_0, phi_t0=phi_0, Neutral_case=True,
                    SST=SST, sf_scheme=sf_scheme, u_delta=u_deltasl,
                    forcing=forcing, delta_sl=delta_sl)

    z_fv, u_fv, theta_fv = simulator.reconstruct_FV(u, phi, temperature,
            dz_theta, SL=SL)
    z_tke, tke_fv = simulator.reconstruct_TKE(TKE,
            dz_tke, SL, sf_scheme, businger(), l_eps)
    # z_tke = simulator.z_full
    return z_fv, u_fv, theta_fv, z_tke, tke_fv, ustar

def plot_FVStratified(axes, sf_scheme, dt=10., N=3240,
        z_levels=DEFAULT_z_levels_stratified,
        stable: bool=True, delta_sl=None,
        name=None, style={}):

    z_fv, u_fv, theta_fv, z_tke, TKE, ustar, l_eps = \
            compute_with_sfStratified(sf_scheme, z_levels, dt, N,
                    stable, delta_sl)
    axes[0].semilogy(np.abs(u_fv), z_fv, **style)
    axes[1].semilogy(theta_fv, z_fv, **style)
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
    u, TKE, ustar, temperature, l_eps = simulator.FD(u_t0=u_0, SST=SST,
            sf_scheme=sf_scheme, forcing=forcing)
    z_tke = np.copy(simulator.z_full)
    z_tke[0] = 0.1

    axes[0].semilogy(np.abs(u), simulator.z_half[:-1], **style)
    axes[1].semilogy(temperature, simulator.z_half[:-1], **style)
    axes[2].semilogy(TKE, z_tke, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)
    axes[4].semilogy(l_eps, z_tke, **style)

def plot_FD(axes, sf_scheme, dt=60., N=1680,
        z_levels=DEFAULT_z_levels, name=None, style={}):
    if name == None:
        name = sf_scheme
    M = z_levels.shape[0] - 1
    simulator = Atm1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=10., K_mol=1e-4, f=1e-4)

    u_0 = 10*np.ones(M)
    forcing = 1j*simulator.f * simulator.u_g*np.ones((N+1, M))
    SST = 265. * np.ones(N+1) # Neutral SST with theta=const=265.
    u, TKE, ustar, _, _ = simulator.FD(u_0,
            forcing, SST, sf_scheme=sf_scheme,
            Neutral_case=True)
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

def fig_consistency_comparison():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels= np.linspace(0, 1500, 41)
    z_levels= IFS_z_levels
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
