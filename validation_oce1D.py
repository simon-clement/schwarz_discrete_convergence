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
    N = 1
    time = dt * np.arange(N+1)
    rho0, cp, Qswmax = 1024., 3985., 0.
    srflx = np.maximum(np.cos(2.*np.pi*(time/86400. - 0.5)), 0. ) * \
            Qswmax / (rho0*cp)
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    heatloss = np.ones(N+1) * 100 # /!\ definition of Q0 is not the same as Florian
    # this heatloss will be divided by (rho0*cp)
    # Q0_{comodo} = -heatloss / (rho cp)
    wind_10m = 1.1*np.ones(N+1) + 0j
    temp_10m = np.ones(N+1) * 5
    # if temp_10m<T0, then __friction_scales does not converge.

    fig, axes = plt.subplots(1, 2)

    for sf_scheme in ("FV free",):
        if sf_scheme == "FV free":
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    simulator_oce.initialization(\
                    np.zeros(simulator_oce.M)+0j, # u_0
                    np.copy(theta_0), # theta_0
                    -.5, wind_10m[0], temp_10m[0], 10., sf_scheme)
        else:
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    u_0, phi_0, theta_0, dz_theta_0, 0., T0

        u_current, phi, tke, all_u_star, theta, \
                    dz_theta, l_eps, SL, viscosity =simulator_oce.FV(\
                u_t0=u_i, phi_t0=phi_i, theta_t0=theta_i,
                dz_theta_t0=dz_theta_i, solar_flux=srflx,
                u_delta=u_delta, t_delta=t_delta,
                heatloss=heatloss, wind_10m=wind_10m, TEST_CASE=0,
                temp_10m=temp_10m, sf_scheme=sf_scheme)
        zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
                phi, theta, dz_theta, SL, ignore_loglaw=False)

        axes[0].plot(thetaFV, zFV, "--",
                label="Temperature Python FV")
        axes[1].plot(viscosity, simulator_oce.z_full, "--",
                label="Diffusivity Python FV")

    u_currentFD, tke, all_u_star, thetaFD, \
                l_eps, viscosityFD = simulator_oce.FD(\
            u_t0=u_0, theta_t0=theta_0, TEST_CASE=0,
            solar_flux=srflx, wind_10m=wind_10m,
            temp_10m=temp_10m,
            heatloss=heatloss, sf_scheme="FD test")

    axes[0].plot(thetaFD, simulator_oce.z_half[:-1], "--",
            label="Temperature Python FD")
    axes[1].plot(viscosityFD, simulator_oce.z_full, "--",
            label="Diffusivity Python FD")

    axes[0].legend()
    axes[1].legend()
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[1].set_yscale("symlog", linthresh=0.1)
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
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    heatloss = None
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
                    -.5, wind_10m[0], temp_10m[0], 10., sf_scheme)
        else:
            u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                    u_0, phi_0, theta_0, dz_theta_0, 0., T0



        u_current, phi, tke, all_u_star, theta, \
                    dz_theta, l_eps, SL, viscosity = simulator_oce.FV(\
                u_t0=u_i, phi_t0=phi_i, theta_t0=theta_i,
                dz_theta_t0=dz_theta_i, solar_flux=srflx,
                TEST_CASE=0, u_delta=u_delta, t_delta=t_delta,
                heatloss=heatloss, wind_10m=wind_10m,
                temp_10m=temp_10m, sf_scheme=sf_scheme)
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
        u_currentFD, tke, all_u_star, thetaFD, \
                    l_eps, viscosityFD = simulator_oce.FD(\
                u_t0=u_0, theta_t0=theta_0, TEST_CASE=0,
                solar_flux=srflx, wind_10m=wind_10m, temp_10m=temp_10m,
                heatloss=heatloss, sf_scheme=sf_scheme)
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
    T0 = 16.
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
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    heatloss = np.ones(N+1) * 100 # /!\ definition of Q0 is not the same as Florian
    # this heatloss will be divided by (rho0*cp)
    # Q0_{comodo} = -heatloss / (rho cp)
    wind_10m = np.zeros(N+1) + 0j
    temp_10m = np.ones(N+1) * 240

    u_current, phi, tke, all_u_star, theta, \
                dz_theta, l_eps, SL, viscosity = simulator_oce.FV(\
            u_t0=u_0, phi_t0=phi_0, theta_t0=theta_0,
            u_delta=10., t_delta=15.,
            dz_theta_t0=dz_theta_0, solar_flux=srflx,
            heatloss=heatloss, wind_10m=wind_10m, TEST_CASE=2,
            temp_10m=temp_10m, sf_scheme="FV test")
    zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
            phi, theta, dz_theta, SL, ignore_loglaw=True)

    u_currentFD, tke, all_u_star, thetaFD, \
                l_eps, viscosityFD = simulator_oce.FD(\
            u_t0=u_0, theta_t0=theta_0, TEST_CASE=2,
            solar_flux=srflx, wind_10m=wind_10m,
            temp_10m=temp_10m, 
            heatloss=heatloss, sf_scheme="FD test")

    fig, axes = plt.subplots(1, 2)
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
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    theta_0 = T0 - N0**2 * np.abs(simulator_oce.z_half[:-1]) / alpha / 9.81
    dz_theta_0 = np.ones(simulator_oce.M+1) * N0**2 / alpha / 9.81
    heatloss = np.zeros(N+1)
    wind_10m = np.ones(N+1) * 11.6 + 0j
    temp_10m = np.ones(N+1) * T0

    u_current, phi, tke, all_u_star, theta, \
                dz_theta, l_eps, SL, viscosity = simulator_oce.FV(\
            u_t0=u_0, phi_t0=phi_0, theta_t0=theta_0,
            u_delta=0., t_delta=T0,
            dz_theta_t0=dz_theta_0, solar_flux=srflx,
            TEST_CASE=1,
            heatloss=heatloss, wind_10m=wind_10m, temp_10m=temp_10m,
            sf_scheme="FV test")
    zFV, uFV, thetaFV = simulator_oce.reconstruct_FV(u_current,
            phi, theta, dz_theta, SL, ignore_loglaw=True)

    u_currentFD, tke, all_u_star, thetaFD, \
                l_eps, viscosityFD = simulator_oce.FD(\
            u_t0=u_0, theta_t0=theta_0, TEST_CASE=1,
            solar_flux=srflx, wind_10m=wind_10m, temp_10m=temp_10m,
            heatloss=heatloss, sf_scheme="FD test")

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
