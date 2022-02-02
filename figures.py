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
from simu1DStratified import Simu1dStratified
from utils_linalg import solve_linear
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

def fig_verify_FVfreeStrat():
    """
        Integrates for 1 day a 1D ekman stratified equation
        with TKE turbulence scheme.
        compares FV free with FV2.
    """
    z_levels = IFS_z_levels
    dt = 10.
    N = 3240 # 28*3600/10=3240

    fig, axes = plt.subplots(1,4, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)

    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":0.8, **kwargs}

    sf_scheme = "FV2"
    plot_FVStratified(axes, sf_scheme, N=N, dt=dt,
            z_levels=z_levels, name=sf_scheme,
            style=style('b'), delta_sl=z_levels[1])

    plot_FVStratified(axes, "FV free", N=N, dt=dt,
            z_levels=z_levels, name=r"FV free, $\delta_{sl}<z_1$",
            style=style('r', linestyle='dashed'),
            delta_sl=z_levels[1]*0.7)

    plot_FDStratified(axes, "FD2", N=N, dt=dt,
            z_levels=z_levels, name="FD2, M=64",
            style=style('k', linestyle='dashed'))

    axes[0].set_ylim(top=400.)
    axes[1].set_ylim(top=400.)
    axes[0].set_xlabel("wind speed ($|u|, ~m.s^{-1}$)")
    axes[0].set_ylabel("height (m)")
    axes[1].set_xlabel(r"Temperature ($\theta$, $K$)")
    axes[1].set_ylabel("height (m)")
    axes[2].set_xlabel("energy (J)")
    axes[2].set_ylabel("height (m)")
    axes[3].set_xlabel("length scale ($l_m,~ m$)")
    axes[3].set_ylabel("height (m)")
    axes[2].legend(loc="upper right")
    show_or_save("fig_verify_FVfreeStrat")

def fig_consistency_comparisonStratified():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels = DEFAULT_z_levels_stratified
    z_levels = IFS_z_levels_stratified
    # z_levels_les= np.linspace(0, 400, 651)
    z_levels_les= np.linspace(0, z_levels[-1], 351)
    dt = 10.
    N = 3240 # 28*3600/10=3240

    fig, axes = plt.subplots(1,4, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)
    col_FDpure = "g"
    col_FV1 = "b"
    col_FVfree = "r"
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":0.8, **kwargs}

    plot_FDStratified(axes, "FD pure", N=N, dt=dt, z_levels=z_levels,
            name="FD, M=65", style=style(col_FDpure, "dotted"))
    plot_FDStratified(axes, "FD pure", N=N, dt=dt, z_levels=z_levels_les,
            name="FD, M=650", style=style(col_FDpure))
    plot_FVStratified(axes, "FV1", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FV1, "dotted"))
    plot_FVStratified(axes, "FV1", delta_sl=z_levels_les[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV1, M=650", style=style(col_FV1))
    plot_FVStratified(axes, "FV2", delta_sl=z_levels[1],
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style("c", "dotted"))
    plot_FVStratified(axes, "FV2", delta_sl=z_levels_les[1],
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV2, M=650", style=style("c"))
    plot_FVStratified(axes, "FV pure", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style("m", "dotted"))
    plot_FVStratified(axes, "FV pure", delta_sl=z_levels_les[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV pure, M=650", style=style("m"))
    plot_FVStratified(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FVfree, "dotted"))
    plot_FVStratified(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV free, M=650", style=style(col_FVfree))

    axes[0].set_ylim(top=1500.)
    axes[1].set_ylim(top=1500.)
    axes[0].set_xlabel(r"wind speed ($|u|, m.s^{-1}$)")
    axes[0].set_ylabel("height (m)")
    axes[1].set_xlabel(r"Potential Temperature ($\theta$, K)")
    axes[1].set_ylabel("height (m)")
    axes[2].set_xlabel("energy (J)")
    axes[2].set_ylabel("height (m)")
    # axes[0].legend(loc="upper right")
    # axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    axes[3].set_ylim(top=0.4, bottom=0.2)
    axes[3].set_ylabel("friction velocity (u*, $m.s^{-1}$)")
    axes[3].set_xlabel("time (s)")
    show_or_save("fig_consistency_comparisonStratified")

def fig_verify_FDStratified():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels = DEFAULT_z_levels_stratified
    dt = 10.
    N = 3240 # 28*3600/10=3240

    fig, axes = plt.subplots(1,4, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)

    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":0.8, **kwargs}

    # plot_FVStratified(axes, "FV pure", N=N, dt=dt,
    #         z_levels=z_levels, name="FV pure, M=64",
    #         style=style('g'))

    # plot_FDStratified(axes, "FD pure", N=N, dt=dt,
    #         z_levels=z_levels, name="FD pure, M=64",
    #         style=style('r', linestyle='dashed'))

    plot_FDStratified(axes, "FD2", N=N, dt=dt,
            z_levels=z_levels, name="FD2, M=64",
            style=style('k', linestyle='dashed'))

    # plot_FVStratified(axes, "FV pure", N=N, dt=dt,
    #         z_levels=z_levels, name="FV pure, M=64",
    #         style=style('b'), delta_sl=z_levels[1]/2)

    plot_FVStratified(axes, "FV2", N=N, dt=dt,
            z_levels=z_levels, name="FV2, M=64",
            style=style('b'), delta_sl=z_levels[1])

    # from fortran.visu import import_data
    # u, z = import_data("fortran/u_final_gabls.out")
    # v, z = import_data("fortran/v_final_gabls.out")
    # theta, ztheta = import_data("fortran/t_final_gabls.out")

    # tke, z_tke = import_data("fortran/tke_final_gabls.out")
    # l_m, z_lm = import_data("fortran/mxl_final_gabls.out")
    # axes[0].plot(np.sqrt(u**2+v**2), z, "--", label="fortran $|u|$")
    # axes[1].plot(theta, ztheta, "--", label="fortran temperature")
    # axes[2].plot(tke, z_tke, "--", label="fortran")
    # axes[3].plot(l_m, z_lm, "--", label="fortran")

    axes[0].set_ylim(top=400., bottom=0.)
    axes[1].set_ylim(top=400., bottom=0.)
    axes[0].set_xlabel("wind speed ($|u|, ~m.s^{-1}$)")
    axes[0].set_ylabel("height (m)")
    axes[1].set_xlabel(r"Temperature ($\theta$, $K$)")
    axes[1].set_ylabel("height (m)")
    axes[2].set_xlabel("energy (J)")
    axes[2].set_ylabel("height (m)")
    axes[3].set_xlabel("length scale ($l_m,~ m$)")
    axes[3].set_ylabel("height (m)")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    show_or_save("fig_verify_FDFVStratified")

def compute_with_sfStratified(sf_scheme, z_levels, dt=10., N=3240,
        delta_sl=None, z_constant=None):
    """
    return z_fv, u_fv, theta_fv, z_tke, TKE, ustar
    """
    if delta_sl is None:
        delta_sl = z_levels[1]/2
    if z_constant is None:
        z_constant = 2*delta_sl

    M = z_levels.shape[0] - 1
    simulator = Simu1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=8.,
            K_mol=1e-4, f=1.39e-4)
    u_0 = 8*np.ones(M)
    phi_0 = np.zeros(M+1)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    SST = np.concatenate(([265],
        [265 - 0.25*(dt*(n-1))/3600. for n in range(1, N+1)]))
    z_tke = np.copy(simulator.z_full)
    k = bisect.bisect_right(z_levels[1:], delta_sl)
    z_tke[k] = delta_sl #
    u_deltasl = 8. # first guess before the iterations
    if sf_scheme in {"FV1 free", "FV2 free", "FV free"}:
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
            u_kp1 = u_deltasl + (u_constant - u_deltasl) * \
                    ((zkp1 - delta_sl) / (z_constant - delta_sl))**3
            # For LES simulation, putting a linear profile between
            # the log law and the constant profile:
            z_linear_profile = zkp1+simulator.h_half[k+1:k_constant]
            u_0[k+1:k_constant] = u_kp1 + (u_constant-u_kp1) * \
                    ((z_linear_profile - zkp1) / (z_constant - zkp1))**3
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
            dz_tke, SL, sf_scheme)

    return z_fv, u_fv, theta_fv, z_tke, tke_fv, ustar, l_eps

def compute_with_sfNeutral(sf_scheme, z_levels, dt, N, delta_sl):
    """
    return z_fv, u_fv, theta_fv, z_tke, TKE, ustar
    """
    M = z_levels.shape[0] - 1
    simulator = Simu1dStratified(z_levels=z_levels,
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
    if sf_scheme in {"FV1 free", "FV2 free", "FV free"}:
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

    u, phi, TKE, dz_tke, ustar, temperature, dz_theta, l_m, SL = \
            simulator.FV(u_t0=u_0, phi_t0=phi_0, Neutral_case=True,
                    SST=SST, sf_scheme=sf_scheme, u_delta=u_deltasl,
                    forcing=forcing, delta_sl=delta_sl)

    z_fv, u_fv, theta_fv = simulator.reconstruct_FV(u, phi, temperature,
            dz_theta, SL=SL)
    return z_fv, u_fv, theta_fv, z_tke, TKE, ustar

def plot_FVStratified(axes, sf_scheme, dt=10., N=3240,
        z_levels=DEFAULT_z_levels_stratified, delta_sl=None,
        name=None, style={}):

    z_fv, u_fv, theta_fv, z_tke, TKE, ustar, l_eps = \
            compute_with_sfStratified(sf_scheme, z_levels, dt, N,
                    delta_sl)
    axes[0].semilogy(np.abs(u_fv), z_fv, **style)
    axes[1].semilogy(theta_fv, z_fv, **style)
    axes[2].semilogy(TKE, z_tke, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)

def plot_FDStratified(axes, sf_scheme, dt=10., N=3240,
        z_levels=DEFAULT_z_levels_stratified,
        name=None, style={}):
    if name is None:
        name = sf_scheme
    M = z_levels.shape[0] - 1
    simulator = Simu1dStratified(z_levels=z_levels,
            dt=dt, u_geostrophy=8.,
            K_mol=1e-4, f=1.39e-4)
    u_0 = 8*np.ones(M)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    SST = np.concatenate(([265],
        [265 - 0.25*(dt*(n-1))/3600. for n in range(1, N+1)]))
    u, TKE, ustar, temperature, l_m = simulator.FD(u_t0=u_0, SST=SST,
            sf_scheme=sf_scheme, forcing=forcing)
    z_tke = np.copy(simulator.z_full)
    z_tke[0] = z_levels[1]/2 if sf_scheme != "FD2" else 0.1

    axes[0].semilogy(np.abs(u), simulator.z_half[:-1], **style)
    axes[1].semilogy(temperature, simulator.z_half[:-1], **style)
    axes[2].semilogy(TKE, z_tke, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)
    # axes[3].semilogy(l_m, simulator.z_full, **style)

def plot_FD(axes, sf_scheme, dt=60., N=1680,
        z_levels=DEFAULT_z_levels, name=None, style={}):
    if name == None:
        name = sf_scheme
    M = z_levels.shape[0] - 1
    simulator = Simu1dStratified(z_levels=z_levels,
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

def fig_verify_FDFV():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels= np.linspace(0, 1500, 41)
    # for FV with FV interpretation of sf scheme,
    # the first grid level is divided by 2 so that
    # delta_{sl} is the same in all the schemes.
    dt = 60.
    N = 1680 # 28*60=1680

    fig, axes = plt.subplots(1,4, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)

    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":0.8, **kwargs}

    # plot_FD(axes, "FD pure", N=N, dt=dt, z_levels=z_levels,
    #         name="FD, M=40", style=style('r'))
    plot_FV(axes, "FV2", delta_sl=z_levels[1],
            N=N, dt=dt, z_levels=z_levels,
            name="FV2, M=40", style=style('b'))
    plot_FV(axes, "FV1 free", delta_sl=z_levels[1]*0.99,
            N=N, dt=dt, z_levels=z_levels,
            name=r"FV1 free, M=40, $\delta_{sl}=0.99z_1$", style=style('r', linestyle='dashed'))

    axes[0].set_ylim(top=1500., bottom=0.)
    axes[1].set_ylim(top=1500., bottom=0.)
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
    show_or_save("fig_verify_FDFV")

def fig_consistency_comparison():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels= np.linspace(0, 1500, 41)
    z_levels= IFS_z_levels
    z_levels_les= np.linspace(0, IFS_z_levels[-1], 401)
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
                "linewidth":0.8, **kwargs}

    plot_FD(axes, "FD pure", N=N, dt=dt, z_levels=z_levels,
            name="FD, M="+str(z_levels.shape[0] - 1),
            style=style(col_FDpure, "dotted"))
    plot_FD(axes, "FD pure", N=N, dt=dt, z_levels=z_levels_les,
            name="FD, M=400", style=style(col_FDpure))
    plot_FV(axes, "FV1", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FV1, "dotted"))
    plot_FV(axes, "FV1", delta_sl=z_levels_les[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV1, M=400", style=style(col_FV1))
    plot_FV(axes, "FV2", delta_sl=z_levels[1],
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style("c", "dotted"))
    plot_FV(axes, "FV2", delta_sl=z_levels_les[1],
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV2, M=400", style=style("c"))
    plot_FV(axes, "FV pure", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style("m", "dotted"))
    plot_FV(axes, "FV pure", delta_sl=z_levels_les[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV pure, M=400", style=style("m"))
    plot_FV(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name=None, style=style(col_FVfree, "dotted"))
    plot_FV(axes, "FV free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV free, M=400", style=style(col_FVfree))

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
