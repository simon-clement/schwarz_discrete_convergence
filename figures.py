#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import numpy as np
from memoisation import memoised
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = ':'
mpl.rcParams["grid.alpha"] = '0.7'
mpl.rcParams["grid.linewidth"] = '0.5'

def biggest_eig_linearized(theta = 1.):
    from nonlinear_simulator import bulk_schwarz_spinup
    builder = Builder()

    dt_spinup = builder.DT = 1e12 # almost infinite
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3
    nonlinear_steadystate = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=40, nonlinear=True, theta=1.5, C_D=C_D)

    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steadystate
    steady_jump_solution = u_atm_NL[0] - u_ocean_NL[-1]
    alpha_linear = C_D * np.abs(steady_jump_solution)

    builder.DT = 60. # 1 minute
    T = 1*24*60*60. # 1 day
    laplace_real_part = 0.

    w = get_discrete_freq(int(T/builder.DT), builder.DT, avoid_zero=False)
    z = np.exp((laplace_real_part+w*1j) * builder.DT)
    s = (z - 1) / z / builder.DT
    ratio_densities = 1e-3
    C_D = 1.2e-3
    h_a = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    h_o = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    #return explicit_part/((1+implicit_part))
    chi_o = (s+builder.R) * h_o**2/builder.D1
    chi_a = (s+builder.R) * h_a**2/builder.D2
    lam_a = (chi_a - np.sqrt(chi_a)*np.sqrt(chi_a+4.))/2
    lam_o = (chi_o - np.sqrt(chi_o)*np.sqrt(chi_o+4.))/2
    chi_o_conjs = (np.conj(s)+builder.R) * h_o**2/builder.D1
    chi_a_conjs = (np.conj(s)+builder.R) * h_a**2/builder.D2
    lam_a_conjs = (chi_a_conjs - np.sqrt(chi_a_conjs)*np.sqrt(chi_a_conjs+4.))/2
    lam_o_conjs = (chi_o_conjs - np.sqrt(chi_o_conjs)*np.sqrt(chi_o_conjs+4.))/2

    eps = ratio_densities * h_a / h_o 
    impl_part = (s+builder.R)*h_a/alpha_linear - lam_a * theta
    # now we have B_k = gamma B_{k-1} + mu CONJ(B_{k-1})
    orientation = steady_jump_solution / np.conj(steady_jump_solution) * (s+builder.R) / (s - builder.R)
    gamma_w = ((1.5 - theta) * lam_a + 1.5 * eps * lam_o) / impl_part
    mu_w = .5 * orientation * np.conj(lam_a_conjs + eps*lam_o_conjs) / impl_part

    M = [ np.array([[np.real(gamma + mu), -np.imag(gamma - mu)],
            [np.imag(gamma+mu), np.real(gamma-mu)]]) for gamma, mu in zip(gamma_w, mu_w)]
    _, sing_values, _ = np.linalg.svd(np.array(M))
    return np.max(sing_values)

def norm2_linearized(theta):
    return norm2_lastschwarz_iter(theta, 1)

def norm2_nonlinear(theta):
    return norm2_lastschwarz_iter(theta, 2)

def norm2_lastschwarz_iter(theta, order):
    from nonlinear_simulator import bulk_schwarz_spinup, bulk_frequency_simulation, linear_steadystate
    builder = Builder()
    dt_spinup = builder.DT = 6000000. # a lot of min
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3

    nonlinear_steadystate = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=40, nonlinear=True, theta=1.5, C_D=C_D)

    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steadystate
    steady_jump_solution = u_atm_NL[0] - u_ocean_NL[-1]
    alpha_linear = C_D * np.abs(steady_jump_solution)

    dt = builder.DT = 60. # 1 minute
    T = 1*24*60*60. # 1 day
    axis_freq = get_discrete_freq(int(T/dt), dt, avoid_zero=True)
    N = int(T/dt)
    B_k = memoised(bulk_frequency_simulation, builder, number_samples=1, T=T, NUMBER_IT=10,
            steady_state=nonlinear_steadystate, order=order, theta=theta,
            C_D=C_D, steady_jump_solution=steady_jump_solution, laplace_real_part=1e-3)[:,N//2:]
    return np.linalg.norm(B_k[-1])

def fig_eigenvalues_linearized():
    fig, ax = plt.subplots()
    from scipy.optimize import minimize_scalar

    print(minimize_scalar(biggest_eig_linearized))
    print(minimize_scalar(norm2_linearized))
    print(minimize_scalar(norm2_nonlinear))
    # B_k = memoised(bulk_frequency_simulation, builder, number_samples=1, T=T, NUMBER_IT=18,
    #         steady_state=nonlinear_steadystate, order=1, theta=theta,
    #         C_D=C_D, steady_jump_solution=steady_jump_solution, laplace_real_part=laplace_real_part)[0:]

    # ax.loglog(w, np.abs(B_k[2])/np.abs(B_k[1]))
    # ax.loglog(w, np.abs(B_k[3])/np.abs(B_k[2]))
    #show_or_save("fig_is_it_linear_enough")

def fig_validation_linearized():
    from nonlinear_simulator import bulk_schwarz_spinup
    builder = Builder()

    dt_spinup = builder.DT = 1e12 # almost infinite
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3
    nonlinear_steadystate = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=40, nonlinear=True, theta=1.5, C_D=C_D)

    max_gamma = []
    max_mu = []
    theta=1.5
    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steadystate
    steady_jump_solution = u_atm_NL[0] - u_ocean_NL[-1]
    alpha_linear = C_D * np.abs(steady_jump_solution)

    builder.DT = 60. # 1 minute
    T = 1*24*60*60. # 1 day
    laplace_real_part = 0.#1e-10

    w = get_discrete_freq(int(T/builder.DT), builder.DT, avoid_zero=False)
    z = np.exp((laplace_real_part+w*1j) * builder.DT)
    s = (z - 1) / z / builder.DT

    s = laplace_real_part + w*1j

    ratio_densities = 1e-3
    C_D = 1.2e-3
    h_a = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    h_o = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    #return explicit_part/((1+implicit_part))
    chi_o = (s+builder.R) * h_o**2/builder.D1
    chi_a = (s+builder.R) * h_a**2/builder.D2
    lam_a = (chi_a - np.sqrt(chi_a)*np.sqrt(chi_a+4.))/2
    lam_o = (chi_o - np.sqrt(chi_o)*np.sqrt(chi_o+4.))/2

    chi_o_conjs = (np.conj(s)+builder.R) * h_o**2/builder.D1
    chi_a_conjs = (np.conj(s)+builder.R) * h_a**2/builder.D2
    lam_a_conjs = (chi_a_conjs - np.sqrt(chi_a_conjs)*np.sqrt(chi_a_conjs+4.))/2
    lam_o_conjs = (chi_o_conjs - np.sqrt(chi_o_conjs)*np.sqrt(chi_o_conjs+4.))/2

    eps = ratio_densities * h_a / h_o
    impl_part = (s+builder.R) * h_a/alpha_linear - lam_a * theta
    # now we have B_k = gamma B_{k-1} + mu CONJ(B_{k-1})
    orientation = steady_jump_solution / np.conj(steady_jump_solution) * (s+builder.R) / (s - builder.R)
    gamma_w = ((1.5 - theta) * lam_a + 1.5 * eps * lam_o) / impl_part
    mu_w = .5 * orientation * np.conj(lam_a_conjs + eps*lam_o_conjs) / impl_part
    M = [ np.array([[np.real(gamma + mu), -np.imag(gamma - mu)],
            [np.imag(gamma+mu), np.real(gamma-mu)]]) for gamma, mu in zip(gamma_w, mu_w)]

    from nonlinear_simulator import bulk_frequency_simulation
    #N = int(T/builder.DT)
    B_k = memoised(bulk_frequency_simulation, builder, number_samples=10, T=T, NUMBER_IT=4,
            steady_state=nonlinear_steadystate, order=1, theta=theta,
            C_D=C_D, steady_jump_solution=steady_jump_solution, laplace_real_part=laplace_real_part, ignore_cached=False)#[:,N//2:]
    plt.semilogx(w, np.real(B_k[2]))

    B_kp1 = gamma_w * B_k[1] + mu_w * np.conj(B_k[1,::-1])

    B_kp1 = B_k[1]*((1-theta)*lam_a + lam_o*eps)/ ((s+builder.R)*h_a/alpha_linear -theta*lam_a)
    B_kp1 = []
    for M_w, B_w in zip(M, B_k[2]):
        B_w_mat = np.array((np.real(B_w), np.imag(B_w)))
        B_kp1 += [(M_w @ B_w_mat)[0] + 1j*(M_w @ B_w_mat)[1]]
    #plt.semilogx(w, np.real(B_k[2]), "r")
    #plt.semilogx(w, np.real(B_kp1), "r--")
    #plt.semilogx(w, np.imag(B_k[2]), "g")
    #plt.semilogx(w, np.imag(B_kp1), "g--")
    plt.semilogx(w, np.abs(B_k[3]/B_k[2]), "b")
    plt.semilogx(w, np.abs(B_k[2]/B_k[1]), "g")
    plt.semilogx(w, np.abs(B_k[4]/B_k[3]), "r--")
    plt.show()

def fig_profile_stationnaire_bug():
    from nonlinear_simulator import bulk_schwarz_spinup, nonlinear_steadystate
    builder = Builder()
    infinite_domain_assumption_factor = 1
    builder.SIZE_DOMAIN_1 = 400*infinite_domain_assumption_factor
    builder.SIZE_DOMAIN_2 = 1000*infinite_domain_assumption_factor
    builder.M2 = 1+50*infinite_domain_assumption_factor
    builder.M1 = 1+200*infinite_domain_assumption_factor
    builder.R = 5e-15j

    dt_spinup = builder.DT = 1e30 # a lot of min
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3
    fig, axes = plt.subplots(ncols=2, nrows=2)
    plt.subplots_adjust(wspace=0.3)

    nonlinear_steady_analytic = nonlinear_steadystate(builder, C_D)
    nonlinear_steady = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=10, nonlinear=True, theta=2., C_D=C_D)
    nonlinear_steady_analytic = nonlinear_steadystate(builder, C_D)
    h1 = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    h2 = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    depth_u1 = np.linspace(- builder.SIZE_DOMAIN_1 + h1/2, -h1/2, builder.M1 - 1)
    height_u2 = np.linspace(h2/2, builder.SIZE_DOMAIN_2 - h2/2, builder.M2 - 1)

    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steady
    u_atm_NL_analytic, _, u_ocean_NL_analytic, _ = nonlinear_steady_analytic

    fig.suptitle(r"Stationnary profile of u when $f=5\times 10^{-14}$")
    axes[0,0].plot(np.real(u_atm_NL), height_u2, label="Result of 10 Schwarz iterations")
    axes[0,0].plot(np.real(u_atm_NL_analytic), height_u2, "--", label="Theoretical")
    axes[0,0].set_yscale('symlog')
    axes[1,0].plot(np.real(u_ocean_NL), depth_u1)
    axes[1,0].plot(np.real(u_ocean_NL_analytic), depth_u1, "--")
    axes[1,0].set_yscale('symlog')
    axes[0,1].plot(np.imag(u_atm_NL), height_u2)
    axes[0,1].plot(np.imag(u_atm_NL_analytic), height_u2, "--")
    axes[0,1].set_yscale('symlog')
    axes[1,1].plot(np.imag(u_ocean_NL), depth_u1)
    axes[1,1].plot(np.imag(u_ocean_NL_analytic), depth_u1, "--")
    axes[1,1].set_yscale('symlog')

    axes[0,0].set_ylabel("z")
    axes[1,0].set_ylabel("z")

    axes[1,0].set_xlabel("Real part")
    axes[1,1].set_xlabel("Imaginary part")
    axes[0,0].legend(loc="upper left")

    plt.show()

def fig_profile_stationnaire():
    from nonlinear_simulator import bulk_schwarz_spinup, nonlinear_steadystate
    builder = Builder()
    h_a = 20
    h_o = 2
    builder.M2 = 100
    builder.M1 = 100
    builder.SIZE_DOMAIN_1 = (builder.M1 - 1) * h_o
    builder.SIZE_DOMAIN_2 = (builder.M2 - 1) * h_a

    dt_spinup = builder.DT = 1e30 # a lot of min
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(3.4,3.4))
    plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.3, top=0.93, bottom=0.12)

    nonlinear_steady_analytic = nonlinear_steadystate(builder, C_D)
    nonlinear_steady = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=5, nonlinear=True, theta=1.5, C_D=C_D)
    h1 = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    h2 = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    depth_u1 = np.linspace(- builder.SIZE_DOMAIN_1 + h1/2, -h1/2, builder.M1 - 1)
    height_u2 = np.linspace(h2/2, builder.SIZE_DOMAIN_2 - h2/2, builder.M2 - 1)

    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steady
    u_atm_NL_analytic, _, u_ocean_NL_analytic, _ = nonlinear_steady_analytic

    axes[0].plot(np.real(u_atm_NL), height_u2, "k", label="Real part (u)")
    axes[0].plot(np.imag(u_atm_NL), height_u2, "grey", label="Imaginary part (v)")
    axes[0].xaxis.tick_top()
    axes[1].plot(np.real(u_ocean_NL), depth_u1, "k")
    axes[1].plot(np.imag(u_ocean_NL), depth_u1, "grey")

    axes[0].grid(color='k', linestyle=':', linewidth=.15, which="minor")
    axes[0].grid(color='k', linestyle=':', linewidth=.2)
    axes[1].grid(color='k', linestyle=':', linewidth=.15, which="minor")
    axes[1].grid(color='k', linestyle=':', linewidth=.2)

    axes[0].set_yscale('symlog')
    axes[1].set_yscale('symlog')
    from matplotlib.ticker import LogLocator, AutoLocator
    axes[0].yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))

    def gen_tick_positions(scale_start=100, scale_max=10000):
        start, finish = np.floor(np.log10((scale_start, scale_max)))
        finish += 1
        majors = [10 ** x for x in np.arange(start, finish)]
        minors = []
        for idx, major in enumerate(majors[:-1]):
            minor_list = np.arange(majors[idx], majors[idx+1], major)
            minors.extend(minor_list[1:])
        return minors, majors

    axes[0].set_yticks([10, 100, 1000])
    axes[0].set_yticklabels([r'$\delta_{a}=10 \, {\rm m}$', r'$100 \, {\rm m}$',r'$1000 \, {\rm m}$'])
    axes[0].set_xticks([-1.25, 0., 1.25, 2.5, 3.75, 5., 6.25, 7.5, 8.75, 10.], minor=True)
    axes[0].set_xticks([0., 2.5, 5., 7.5, 10.], minor=False)
    axes[1].set_xticks([-0.05, 0., 0.05, 0.1, 0.15])
    axes[1].set_xticks([-0.05, -0.025, 0., 0.025, 0.05, 0.075, 0.1, 0.125, 0.15], minor=True)


    axes[1].set_yticks([-200, -100, -90, -80, -70, -60, -50, -40, -30,
        -20, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -0.9], minor=True)
    axes[1].set_yticks([-100,-10,-1])
    axes[1].set_yticklabels([r'$-100 \, {\rm m}$',r'$-10 \, {\rm m}$',r'$\delta_{o}=-1 \, {\rm m}$'])



    axes[0].set_ylabel(r"$z$")
    axes[1].set_ylabel(r"$z$")

    axes[1].set_xlabel(r"$u, v$")
    axes[0].legend(loc="upper center")
    show_or_save("fig_profile_stationnaire")

def fig_test():
    pass

def fig_is_it_linear_enough():
    from nonlinear_simulator import bulk_schwarz_spinup, bulk_frequency_simulation, linear_steadystate
    builder = Builder()
    dt_spinup = builder.DT = 6000000. # a lot of min
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3

    nonlinear_steadystate = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=40, nonlinear=True, theta=1.5, C_D=C_D)

    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steadystate
    steady_jump_solution = u_atm_NL[0] - u_ocean_NL[-1]
    alpha_linear = C_D * np.abs(steady_jump_solution)

    dt = builder.DT = 60. # 1 minute
    T = 1*24*60*60. # 1 day
    axis_freq = get_discrete_freq(int(T/dt), dt, avoid_zero=True)
    N = int(T/dt)
    fig, axes = plt.subplots(2,3)
    for order, axe, vmax in zip((0,2), axes, (0.15, 0.3)):
        for theta, ax in zip((0.5, 1., 2.), axe):
            ret = []
            X = axis_freq[N//2:]
            B_k = memoised(bulk_frequency_simulation, builder, number_samples=1, T=T, NUMBER_IT=15,
                    steady_state=nonlinear_steadystate, order=order, theta=theta,
                    C_D=C_D, steady_jump_solution=steady_jump_solution, laplace_real_part=1e-3)[:,N//2:]
            for i in range(8):
                ret += [np.abs(B_k[i+2]/B_k[i+1])]

            Y = list(range(8))
            Z = np.vstack(ret)
            CS = ax.pcolormesh(X, Y, Z, vmin=0., vmax=vmax, cmap='Greys')
            ax.set_xscale('log')
            #ax.grid(color='k', linestyle=':', linewidth=.2)

            ax.set_xticks([1e-4, 1e-3, 1e-2])
            ax.set_yticks([1, 4, 7])

            # if order == 2:
            #     ax.set_title('Local convergence rate: non-linear, $\\theta=$'+str(theta))
            #     ax.set_title('$\\theta=$'+str(theta))
            # elif order == 1:
            #     ax.set_title('Local convergence rate: linearized, $\\theta=$'+str(theta))
            if order == 0:
            #    ax.set_title('Local convergence rate: linear, $\\theta=$'+str(theta))
                ax.set_title('$\\theta=$'+str(theta))
        cbar = fig.colorbar(CS, ax=ax)

    axes[0,0].set_xticklabels([])
    axes[0,1].set_xticklabels([])
    axes[0,2].set_xticklabels([])

    axes[0,1].set_yticklabels([])
    axes[1,1].set_yticklabels([])

    axes[0,2].set_yticklabels([])
    axes[1,2].set_yticklabels([])


    axes[1,0].set_xlabel(r'$\omega$')
    axes[1,1].set_xlabel(r'$\omega$')
    axes[1,2].set_xlabel(r'$\omega$')
    axes[0,0].set_ylabel(r'Linear iteration')
    axes[1,0].set_ylabel(r'Nonlinear iteration')
    show_or_save("fig_is_it_linear_enough")

def fig_evolution_err_nonlinear():
    from nonlinear_simulator import bulk_schwarz_spinup, bulk_schwarz_simulator
    builder = Builder()
    dt_spinup = builder.DT = 1e12 # almost infinite
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3
    ratio_densities = 1e-3

    nonlinear_steadystate = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=40, nonlinear=True, theta=1.5, C_D=C_D)

    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steadystate
    steady_jump_solution = u_atm_NL[0] - u_ocean_NL[-1]

    builder.DT = 60. # 1 minute
    T = 1*24*60*60. # 1 day
    N = int(T / builder.DT)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.2,3.0))
    plt.subplots_adjust(left=0.11, right=0.69, bottom=0.15)
    order = 0
    theta = 1
    labels = (r"constant $\alpha$", "nonlinear", "linearized")

    colors_orders = ("#000000", "#000000")
    style_orders = ("", "", "d")
    style_thetas = ("--", "-", ":")
    label = [r"Constant $\alpha$, $\theta =$ ",r"Linearized, $\theta =$",r"Nonlinear, $\theta =$"]
    for order, col_order in zip((0,2), colors_orders):
        for theta, style_theta in zip((1., 1.5), style_thetas):
            B_k = memoised(bulk_schwarz_simulator, builder, T=T, NUMBER_IT=11,
                    steady_state=nonlinear_steadystate, order=order, theta=theta,
                    C_D=C_D, steady_jump_solution=steady_jump_solution)

            ax.semilogy(np.linalg.norm(B_k, axis=-1)/np.sqrt(N), style_orders[order] + style_theta, color=col_order, markersize=6, fillstyle='none', label=label[order]+str(theta))

            norme2_evol = np.linalg.norm(B_k, axis=-1)/np.sqrt(N)
            if order == 0:
                print("linear:", norme2_evol[4] / norme2_evol[3], np.abs((1 - theta + ratio_densities*np.sqrt(builder.D2/builder.D1))/theta))
                ax.semilogy(norme2_evol[1]*(np.abs((1 - theta + ratio_densities*np.sqrt(builder.D2/builder.D1))/theta))**np.array(range(-1, 11)), style_orders[order] + style_theta, color='grey', markersize=6, fillstyle='none')
            else:
                ax.semilogy(norme2_evol[1]*(np.abs((1.5 - theta + 1.5*ratio_densities*np.sqrt(builder.D2/builder.D1))/theta))**np.array(range(-1, 11)), \
                    style_orders[order] + style_theta, color='grey', markersize=6, fillstyle='none')


    ax.set_ylim(ymin=1e-6, ymax=1.)
    #ax2.set_ylim(ymin=4e-8, ymax=4e-2)
    ax.set_ylabel(r"Error $||\cdot||_2$")

    ax.set_xlabel("Iteration")
    #ax2.set_ylabel(r"Error $||\cdot||_\infty$")



    ax.set_xlim(xmin=1, xmax=10)
            

    import matplotlib.patches as mpatches
    grey_patch = mpatches.Patch(color='grey')
    h, l = ax.get_legend_handles_labels()
    fig.legend(h + [grey_patch], l + [r"Corresponding $\xi_0^k$"], loc='center right')
    ax.grid(color='k', linestyle=':')
    show_or_save("fig_evolution_err_nonlinear")

def fig_robustesse_evolution_err_nonlinear():
    from nonlinear_simulator import bulk_schwarz_spinup, bulk_schwarz_simulator
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.2,3.0))
    plt.subplots_adjust(left=0.11, right=0.69, bottom=0.15)
    builder = Builder()
    setups = [{}] # {} = first experiment with no changes
    setups += [{'D2' : .1 }] # instead of 1
    setups += [{'D1' : 1e-4}] # instead of 3e-3
    setups += [{'R'  : 1e-5j}]# instead of 1e-4j
    labels = [
            r'NL',
            r'L',
            r'NL, $\nu_2=0.1\; {\rm m^2}\;{\rm s}^{-1}$',
            r'L, $\nu_2=0.1\; {\rm m^2}\;{\rm s}^{-1}$',
            r'NL, $\nu_1=10^{-4}\; {\rm m^2}\;{\rm s}^{-1}$',
            r'L, $\nu_1=10^{-4}\; {\rm m^2}\;{\rm s}^{-1}$',
            r'NL, $f=10^{-5}\; {\rm s}^{-1}$',
            r'L, $f=10^{-5}\; {\rm s}^{-1}$',
            ]
    styles_setups_NL = ("-", "--", "-.", ":")
    styles_setups_L = ("+", "x", "p", "*")
    lines = [] # for style
    from matplotlib.lines import Line2D

    for changes, style_NL, style_L in zip(setups, styles_setups_NL, styles_setups_L):
        builder = Builder()
        C_D = 1.2e-3
        for key in changes:
            builder.__dict__[key] = changes[key]

        dt_spinup = builder.DT = 1e12 # almost infinite
        T_spinup = 10*dt_spinup
        nonlinear_steadystate = memoised(bulk_schwarz_spinup, builder,
                    T=T_spinup, NUMBER_IT=40, nonlinear=True, theta=1.5, C_D=C_D, geostrophy=(10,1.))

        u_atm_NL, _, u_ocean_NL, _ = nonlinear_steadystate
        steady_jump_solution = u_atm_NL[0] - u_ocean_NL[-1]

        builder.DT = 60. # 1 minute
        T = 1*24*60*60. # 1 day

        N = int(T / builder.DT)

        interface_values_L = memoised(bulk_schwarz_simulator, builder, T=T, NUMBER_IT=11,
                steady_state=nonlinear_steadystate, order=1, theta=1.5,
                C_D=C_D, steady_jump_solution=steady_jump_solution, init="white")

        ax.semilogy(np.linalg.norm(interface_values_L, axis=-1)/np.sqrt(N),
                style_L, fillstyle='none', color='k', markersize=7.2)

        interface_values_NL = memoised(bulk_schwarz_simulator, builder, T=T, NUMBER_IT=11,
                steady_state=nonlinear_steadystate, order=2, theta=1.5,
                C_D=C_D, steady_jump_solution=steady_jump_solution, init="white")
        ax.semilogy(np.linalg.norm(interface_values_NL, axis=-1)/np.sqrt(N),
                style_NL, color='k')

        interface_values_NL_theta1 = memoised(bulk_schwarz_simulator, builder, T=T, NUMBER_IT=11,
                steady_state=nonlinear_steadystate, order=2, theta=1.,
                C_D=C_D, steady_jump_solution=steady_jump_solution, init="white")
        ax.semilogy(np.linalg.norm(interface_values_NL_theta1, axis=-1)/np.sqrt(N),
                style_NL, color='grey', lw=0.8)
        interface_values_L_theta1 = memoised(bulk_schwarz_simulator, builder, T=T, NUMBER_IT=11,
                steady_state=nonlinear_steadystate, order=1, theta=1.,
                C_D=C_D, steady_jump_solution=steady_jump_solution, init="white")
        ax.semilogy(np.linalg.norm(interface_values_L_theta1, axis=-1)/np.sqrt(N),
                style_L, fillstyle='none', markersize=7.2, color='grey', lw=0.8)
        lines += [Line2D([0], [0], linestyle=style_NL, color="k")]
        lines += [Line2D([0], [0], marker=style_L, lw=0, color="k")]

    ax.set_ylabel(r"Error $||\cdot||_2$")

    ax.set_xlabel("Iteration")

    ax.set_ylim(ymin=1e-6, ymax=1.)
    ax.set_xlim(xmin=1, xmax=10)
    ax.grid(color='k', linestyle=':')
            
    import matplotlib.patches as mpatches
    h, l = ax.get_legend_handles_labels()

    grey_patch = mpatches.Patch(color='grey')
    black_patch = mpatches.Patch(color='k')

    labels += [r"$\theta = 1.5$"]
    labels += [r"$\theta = 1$"]
    fig.legend(h + lines + [black_patch, grey_patch], l + labels, loc="center right")
    show_or_save("fig_robustesse_evolution_err_nonlinear")

def fig_contour_linear_theta():
    """
        plots the linear convergence rate over frequencies and theta
    """
    levels = np.array((.005, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24))
    contour_theta(nonlinear=False, levels=levels)
    show_or_save("fig_contour_linear_theta")

def fig_contour_nonlinear_theta():
    """
        plots the non-linear convergence rate over frequencies and theta
    """
    levels = np.array((0.05, 0.1, .16, 0.2, 0.3, 0.45, 0.6))
    contour_theta(nonlinear=True, levels=levels)
    show_or_save("fig_contour_nonlinear_theta")

def contour_theta(nonlinear, levels):
    """
        plots the convergence rate over frequencies and theta
    """
    from nonlinear_simulator import bulk_schwarz_spinup, bulk_frequency_simulation, linear_steadystate
    builder = Builder()
    dt_spinup = builder.DT = 6000000. # a lot of min
    T_spinup = 10*dt_spinup
    C_D = 1.2e-3
    nonlinear_steadystate = memoised(bulk_schwarz_spinup, builder,
                T=T_spinup, NUMBER_IT=40, nonlinear=True, theta=2., C_D=C_D)
    if nonlinear:
        steady_state =  nonlinear_steadystate
    else:
        steady_state = linear_steadystate(builder, C_D)

    u_atm_NL, _, u_ocean_NL, _ = nonlinear_steadystate
    steady_jump_solution = u_atm_NL[0] - u_ocean_NL[-1]
    alpha_linear = C_D * np.abs(steady_jump_solution)

    dt = builder.DT = 600. # 10 minute
    T = 10*24*60*60. # 10 day
    axis_freq = get_discrete_freq(int(T/dt), dt, avoid_zero=True)
    N = int(T/dt)
    all_thetas = np.linspace(1., 2.5, 10)
    #linear case:
    if not nonlinear:
        all_thetas = np.linspace(0.5, 1.5, 10)

    ret = []
    X = axis_freq[N//2:]
    for theta in all_thetas:
        if not nonlinear:
            ret += [theory_cv_bulk(builder, w=X, theta=theta, alpha=alpha_linear)]
            # B_k = memoised(bulk_frequency_simulation, builder, number_samples=8, T=T, NUMBER_IT=2,
            #         steady_state=nonlinear_steadystate, order=1, theta=theta, C_D=C_D, steady_jump_solution=steady_jump_solution)[:,N//2:]
            # ret += [B_k[2]/B_k[1]]
        else:
            B_k = memoised(bulk_frequency_simulation, builder, number_samples=8, T=T, NUMBER_IT=2,
                    steady_state=steady_state, order=2, theta=theta, C_D=C_D, steady_jump_solution=steady_jump_solution)[:,N//2:]
            ret += [B_k[2]/B_k[1]]

    Y = all_thetas
    Z = np.vstack(ret)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)#, levels=levels)
    #manual_locations = [(1.1e-3, 1.35), (3.8e-4, 1.25), (2e-4, 1.2), (2e-4, 1.1), (1.6e-4, 1.0), (1e-2, 1.55)]
    ax.clabel(CS, inline=True, fontsize=9, manual=False, fmt='%1.2f', colors='k')
    # fig.colorbar(CS)

    if nonlinear:
        ax.set_title('Convergence rate: Bulk, non-linear case')
    else:
        ax.set_title('Convergence rate')#: Bulk, linear case')
    ax.set_xlabel(r'Frequency $\omega$ (${\rm s^{-1}}$)')
    ax.set_ylabel(r'Free parameter $\theta$')


def fig_first_test_bulk():
    from nonlinear_simulator import bulk_frequency_simulation
    builder = Builder()
    dt = builder.DT = 60. # 1 minute
    builder.R = 1e-4j
    builder.D1=3e-3
    builder.D2=1.
    builder.M1=1000
    builder.M2=1000
    builder.SIZE_DOMAIN_1=10000
    builder.SIZE_DOMAIN_2=1000
    T = 24*60*dt

    axis_freq = get_discrete_freq(int(T/dt), dt, avoid_zero=True)
    for theta in (1.3,):
        ret = memoised(bulk_frequency_simulation, builder, number_samples=1, T=T, NUMBER_IT=3,
                nonlinear=False, theta=theta, ignore_cached=True)
        plt.loglog(axis_freq, np.abs(ret[2]/ret[1]), label=r"$\theta="+str(theta)+"$")
        theoretical_cv = theory_cv_bulk(builder, w=axis_freq, theta=theta)
        plt.loglog(axis_freq, theoretical_cv, "--", label=r"validation with " + str(theta))
    plt.legend()
    show_or_save("fig_first_test_bulk")

def theory_cv_bulk_linearised(builder, w, theta):
    z = np.exp(w * 1j * builder.DT)
    s = (z - 1) / z / builder.DT
    ratio_densities = 1e-3
    C_D = 1.2e-3
    h_a = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    h_o = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    #return explicit_part/((1+implicit_part))
    chi_o = (s+builder.R) * h_o**2/builder.D1
    chi_a = (s+builder.R) * h_a**2/builder.D2
    lam_a = (chi_a - np.sqrt(chi_a)*np.sqrt(chi_a+4.))/2
    lam_o = (chi_o - np.sqrt(chi_o)*np.sqrt(chi_o+4.))/2
    assert (np.abs(1+lam_o) <= 1).all()
    assert (np.abs(1+lam_a) <= 1).all()
    rho_a = 1.
    ua_g = 10.
    uo_g = .1
    nu_a = builder.D2
    nu_o = builder.D1
    DIRECTION = (ua_g - uo_g) / np.conj(ua_g - uo_g)
    alpha = rho_a * C_D * (ua_g - uo_g)
    def m_expl():
        res = []
        for part in (1., 1j): # real part then imaginary part of B_{k-1}
            ua_km1 = nu_a*(lam_a - 1)/((s+builder.R)*h_a)*part
            uo_k = nu_o*(lam_o - 1)/((s+builder.R)*h_o)*ratio_densities * nu_a/nu_o * part
            res += [alpha*(ua_g + 3/2 * (ua_g - (1-theta)*ua_km1) + DIRECTION/2 * np.conj(ua_g - (1-theta)*ua_km1) \
                    - (uo_g + 3/2 * (uo_g - uo_k) + DIRECTION/2 * np.conj(uo_g - uo_k)))]
        return res
    def m_impl():
        res = []
        for part in (1., 1j): # real part then imaginary part of B_{k-1}
            ua_km1 = nu_a*(lam_a - 1)/((s+builder.R)*h_a)*part
            res += [part + 3*alpha*theta*ua_km1 / 2 + np.conj(ua_km1)*DIRECTION*alpha*theta/2]
        return res

    implicit = m_impl()
    explicit = m_expl()
    to_inverse = np.array(((np.real(implicit[0]), np.real(implicit[1])),(np.imag(implicit[0]), np.imag(implicit[1]))))
    mat_expl = np.array(((np.real(explicit[0]), np.real(explicit[1])),(np.imag(explicit[0]), np.imag(explicit[1]))))
    ret = [(np.linalg.solve(to_inverse[:,:,w], mat_expl[:,:,w])) for w in range(to_inverse.shape[2])]
    return np.array(ret)

def theory_cv_bulk(builder, w, theta, alpha=None):
    z = np.exp(w * 1j * builder.DT)
    s = (z - 1) / z / builder.DT
    ratio_densities = 1e-3
    h_a = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    h_o = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    #return explicit_part/((1+implicit_part))
    chi_o = (s+builder.R) * h_o**2/builder.D1
    chi_a = (s+builder.R) * h_a**2/builder.D2
    lam_a = (chi_a - np.sqrt(chi_a)*np.sqrt(chi_a+4.))/2
    lam_o = (chi_o - np.sqrt(chi_o)*np.sqrt(chi_o+4.))/2
    assert (np.abs(1+lam_o) <= 1).all()
    assert (np.abs(1+lam_a) <= 1).all()
    eps = ratio_densities * h_a/h_o
    if alpha is None:
        alpha = 1.2e-3 * (10 - .1)
    return (np.abs(1 - theta + eps * lam_o/lam_a)/np.abs((s+builder.R)*h_a/(alpha*lam_a) - theta))


def fig_compare_theory_asymptote():
    builder = Builder()
    ratio_densities = 1e-3
    dt = builder.DT = 600. # 10 minutes
    builder.LAMBDA_1 = 0.
    builder.LAMBDA_2 = 0.
    builder.R = 0#1e-4j
    builder.D1=5e-4
    builder.D2=5.
    builder.M1=1000
    builder.M2=1000
    builder.SIZE_DOMAIN_1=10000
    builder.SIZE_DOMAIN_2=1000
    T = 24*6000*dt
    w = get_discrete_freq(int(T/dt), dt, avoid_zero=True)
    h_a = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    h_o = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    eps = ratio_densities * h_a/h_o
    max_theoretical_cv = []
    asymptote = []
    all_theta = np.linspace(.1, 1.2, 100)
    for theta in all_theta:
        max_theoretical_cv += [np.max(theory_cv_bulk(builder, w, theta=theta))]
        asymptote += [np.abs(1 - theta + eps*np.sqrt(builder.D2/builder.D1)*h_o/h_a)/theta]
        if max_theoretical_cv[-1] > asymptote[-1]:
            plt.semilogx(w, theory_cv_bulk(builder, w, theta=theta))
            plt.show()
    plt.semilogx(all_theta, asymptote, "--")

        #asymptote +=[theory_cv_bulk(builder, w=0., theta=theta)]

    plt.semilogx(all_theta, max_theoretical_cv)
    plt.semilogx(all_theta, asymptote, "--")
    plt.show()

def fig_debug_stationnary():
    builder = Builder()
    builder.DT = 100000000.
    T_spinup = 50*builder.DT
    from nonlinear_simulator import bulk_schwarz_spinup, bulk_frequency_simulation

    u_atm, phi_atm, u_ocean, phi_ocean = memoised(bulk_schwarz_spinup, builder, T=T_spinup, NUMBER_IT=40, theta=2., nonlinear=True)
    # now we have our stationnary values:
    # First, verify our formulas:
    #print("errors:")

    ratio_densities = 1e-3
    C_D = 1.2e-3
    h_a = builder.SIZE_DOMAIN_2/(builder.M2-1)
    h_o = builder.SIZE_DOMAIN_1/(builder.M1-1)
    R = 1e-4j
    s=0
    nu_1 = builder.D1
    nu_2 = builder.D2
    #return explicit_part/((1+implicit_part))
    chi_o = (s+R) * h_o**2/nu_1
    chi_a = (s+R) * h_a**2/nu_2
    lam_a = (chi_a - np.sqrt(chi_a)*np.sqrt(chi_a+4.))/2
    lam_o = (chi_o - np.sqrt(chi_o)*np.sqrt(chi_o+4.))/2
    assert abs(nu_2*lam_a/ (R*h_a) * phi_atm[0] + 10. - u_atm[0]) < 1e-10
    assert abs(-nu_1*lam_o/ (R*h_o) * phi_ocean[-1] + 0.1 - u_ocean[-1]) < 1e-10

    assert abs((lam_o + 1) * phi_ocean[-1] - phi_ocean[-2]) < 1e-10
    assert abs((lam_a + 1) * phi_atm[0] - phi_atm[1]) < 1e-10

    assert abs(1e3*nu_1*phi_ocean[-1] - nu_2*phi_atm[0]) < 1e-10
    bulk_interface_k = 1.2e-3 * np.abs(u_atm[0] - u_ocean[-1])*(u_atm[0] - u_ocean[-1])

    assert abs(bulk_interface_k - nu_2*phi_atm[0]) < 1e-10
    # 1e3*nu_1*phi_ocean[-1] = 1.2e-3 * np.abs(u_atm[0] - u_ocean[-1])*(
    assert abs((u_atm[0] - u_ocean[-1]) -( 1e3 * nu_1*lam_a/ (R*h_a) * phi_ocean[-1] + 10. + nu_1*lam_o/ (R*h_o) * phi_ocean[-1] - 0.1)) < 1e-12

    tilde_d = (1e3 * nu_1*lam_a/ (R*h_a)  + nu_1*lam_o/ (R*h_o))
    g = 10 - 0.1
    ua_m_uo = ( tilde_d * phi_ocean[-1] + g)
    assert abs((np.abs(ua_m_uo) *ua_m_uo) - (phi_ocean[-1] / C_D * nu_1 * 1e3 )) < 1e-10
    # change of variable: we look for X = phi_ocean[-1] / C_D * nu_1 * 1e3
    X = tilde_d*phi_ocean[-1]+g
    c = 1e3*nu_1/C_D

    d = tilde_d / c
    g = 10 - .1
    assert abs((X-g) - d*np.abs(X)*X) < 1e-10
    tilde_a = np.real(d)
    b = np.imag(d)
    x, tilde_y = np.real(X), np.imag(X)
    t = b/g
    a = tilde_a * t
    y = np.cbrt(tilde_y/t)
    print(y)
    print(b**2*(1 - t**2 * y**4) - (t*y - a*y**2 )**2)


def fig_validate_initialisation():
    builder = Builder()
    builder.DT = 100000000.
    T_spinup = 50*builder.DT
    from nonlinear_simulator import bulk_schwarz_spinup, bulk_frequency_simulation

    steady_state = memoised(bulk_schwarz_spinup, builder, T=T_spinup, NUMBER_IT=40, theta=2., nonlinear=True, ignore_cached=True)

    T = 1000*builder.DT

    #alpha_w = memoised(bulk_frequency_simulation, builder, number_samples=10, steady_state=steady_state, T=T, NUMBER_IT=2, theta=2., C_D=1.2e-3, nonlinear=True)
    #plt.plot(alpha_w[2]/alpha_w[1])
    # plt.plot(np.array(u_a), label="u_a")
    # plt.plot(np.array(u_a0), "--", label="u_a0")
    #plt.plot(np.array(u_o0) - np.array(u_o), label="u_o err")
    #plt.plot(np.array(phi_a0) - np.array(phi_a), label="phi_a err")
    #plt.plot(np.array(phi_o0) - np.array(phi_o), label="phi_o err")
    #plt.show()

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
        self.R = 1e-4j
        self.D1=3e-3
        self.D2=1.
        self.M1=1000
        self.M2=100
        self.SIZE_DOMAIN_1=2000
        self.SIZE_DOMAIN_2=2000
        self.DT = self.COURANT_NUMBER * (self.SIZE_DOMAIN_1 / self.M1)**2 / self.D1

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
