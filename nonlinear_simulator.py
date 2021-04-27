#!/usr/bin/python3
"""
    This module provides functions to observe real convergence factors.
"""
import functools
import numpy as np

def bulk_frequency_simulation(builder, number_samples=100, steady_state=None, C_D=1.2e-3, laplace_real_part=0., **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.

        Every argument should be given in builder, except **kwargs
          kwargs = {T, NUMBER_IT, C_D, theta, nonlinear, initial_conditions}
        T: time window
        NUMBER_IT: number of Schwarz iterations (should be at least > 3)
        nonlinear: set to True if alpha = rho_a C_D|u_a-u_o|
        C_D: drag coef. of bulk formula.
        theta: the relaxation parameter-ish
        steady_state: very important to set the good nonlinear steady state !
        It is used because we solve on the difference with the steady state.

    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    # we put two times the same discretization, the code is evolutive
    to_map = functools.partial(bulk_schwarz_simulator, builder, C_D=C_D,
            steady_state=steady_state, **kwargs)

    _, phi_a, _, _ = steady_state if steady_state is not None else linear_steadystate(builder, C_D)

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        err_ret = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            err_ret += [result]
        errors = np.array(err_ret)

    errors *= np.exp(- laplace_real_part * builder.DT * np.array(range(errors.shape[-1])))
    freq_err = fftshift(fft(errors, axis=-1), axes=(-1, ))

    # for a given frequency:
    # convergence <=> the difference decrease
    # we compare the first three freq_u[1, f], freq_u[2, f], freq_u[3, f]
    # first_diff = np.mean(np.abs(freq_u[:,2] - freq_u[:,1]), axis=0)
    # second_diff = np.mean(np.abs(freq_u[:,3] - freq_u[:,2]), axis=0)

    # freq_err = np.array([u[:-1] - u[-1] for u in freq_u]) # where convergence
    # for sample in range(freq_err.shape[0]): # if divergence, no need for subtraction.
    #     np.copyto(freq_err[sample], freq_u[sample,:-1], where=(second_diff > first_diff))

    if freq_err.shape[0] == 1: # special case with only one sample
        return freq_err[0]

    return np.std(freq_err, axis=0)

def bulk_schwarz_simulator(builder, seed=9380, T=3600, NUMBER_IT=3, C_D=1.2e-3, theta=1., order=2, steady_state=None, steady_jump_solution=None):
    """
        Returns errors at interface from beginning (first guess) until the end.
        Coupling is made between AtmoFdBeFlux and OceanFdBeFlux models.
        builder: all the discretisation paramters.
        seed: random parameter for initializing first guess
        T: length of time window (s)
        NUMBER_IT: number of iterations, should be at least 3.
        C_D, theta: bulk parameters of interface
        order: if 2, alpha = C_D|u_a^{k-1} - u_o^{k-1}|
            (otherwise alpha = C_D |steady_jump_solution|
            if 0, cond = alpha (u_a^{k-1} - u_o^{k-1})
            if 1, cond = alpha (u_a^{k-1} - u_o^{k-1}) + alpha * linearized
            if 2, cond = alpha (u_a^{k-1} - u_o^{k-1})
        steady_jump_solution: at equilibrium, u_{1/2} - u{-1/2}
        We only look at the errors around the equilibrium "steady_state".
        the equations are then homogenous equations, and condition transforms to
        cond = alpha (u_a - u_o + u_a^e - u_o^e) (u_a - u_o + u_a^e - u_o^e) - rho_a nu_a phi^e
        for the classical schwarz method, see bulk_schwarz_spinup
    """
    assert order == 0 or order == 1 or order == 2
    from scipy.interpolate import interp1d
    from atmo_fd_be_flux import AtmoFdBeFlux
    from ocean_fd_be_flux import OceanFdBeFlux
    # atmosphere model: can have specific r, nu, M, size_domain, dt
    atmosphere = AtmoFdBeFlux(r=builder.R, nu=builder.D2,
            M=builder.M2, SIZE_DOMAIN=builder.SIZE_DOMAIN_2, DT=builder.DT)
    # ocean model: can have specific r, nu, M, size_domain, dt
    ocean = OceanFdBeFlux(r=builder.R, nu=builder.D1,
            M=builder.M1, SIZE_DOMAIN=builder.SIZE_DOMAIN_1, DT=builder.DT)

    # number of time steps in each model:
    N_ocean = int(np.round(T/ocean.dt))
    N_atm = int(np.round(T/atmosphere.dt)) # atm stands for atmosphere in the following
    rho_atm = 1.
    rho_ocean = 1e3 # ratio of the densities matters rho(atmosphere) / rho(ocean)

    def f_ocean(t): # Ocean is mainly forced by the interface with atmosphere but we use the (small) geostrophic speed
        return np.zeros(ocean.M - 1)
    def f_atm(t): # geostrophic forcing of atmosphere
        return np.zeros(atmosphere.M - 1)

    u_atm0, phi_atm0, u_ocean0, phi_ocean0 = steady_state if steady_state is not None else linear_steadystate(builder, C_D)

    # random initialization of the interface: we excite the high frequencies with white noise
    # to make it right, maybe we should look around the solution after spin-up instead of the
    # steady state of the linear problem ?
    np.random.seed(seed)
    u_ocean_interface = ((np.concatenate(([0], 2 * (np.random.rand(N_ocean) - 0.5)))) + 1j*(np.concatenate(([0], 2 * (np.random.rand(N_ocean) - 0.5)))))/40
    phi_ocean_interface = ((np.concatenate(([0], 2 * (np.random.rand(N_ocean) - 0.5)))) + 1j*(np.concatenate(([0], 2 * (np.random.rand(N_ocean) - 0.5)))))/40
    u_atm_interface = ((np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))) + 1j*(np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))))/40
    phi_atm_interface = ((np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))) + 1j*(np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))))/40

    all_u_atm_interface = [u_atm_interface]
    all_phi_atm_interface = [phi_atm_interface]
    all_u_ocean_interface = [u_ocean_interface]
    #all_phi_ocean_interface = [phi_ocean_interface]

    # Beginning of schwarz iterations:
    for k in range(NUMBER_IT+1):
        # geostrophic situation at time t=0:
        u_atm = np.zeros_like(u_atm0)
        phi_atm = np.zeros_like(phi_atm0)
        u_atm_interface = [0.]
        phi_atm_interface = [0.]
        reaction_effect = 1/(1. + atmosphere.r*atmosphere.dt)
        for n in range(1, N_atm+1):
            linearized = 0.

            if order == 1:
                diff_err = (all_u_atm_interface[-1][n] - all_u_ocean_interface[-1][n])
                orientation = steady_jump_solution / np.conj(steady_jump_solution)
                linearized = (diff_err + orientation * np.conj(diff_err))/2

            if order == 2:
                alpha = rho_atm * C_D * np.abs(all_u_atm_interface[-1][n] - all_u_ocean_interface[-1][n] + steady_jump_solution)
                #interface_info = alpha (theta (u^{n-1}+ if dt u_g)/(1+r dt) + (1-theta) u^{k-1} - u_ocean)
                interface_info = alpha*(theta * u_atm[0] * reaction_effect \
                        + (1.-theta)*all_u_atm_interface[-1][n] \
                        - all_u_ocean_interface[-1][n] + steady_jump_solution) \
                        + alpha*linearized - rho_atm * atmosphere.nu * phi_atm0[0]
            else:
                alpha = rho_atm * C_D * np.abs(steady_jump_solution)
                interface_info = alpha*(theta * u_atm[0] * reaction_effect \
                        + (1.-theta)*all_u_atm_interface[-1][n] \
                        - all_u_ocean_interface[-1][n] \
                        + linearized) \

            phi_atm, u_atm = atmosphere.integrate_in_time(phi_n=phi_atm, u_n=u_atm,
                    interface_info_next_time=interface_info,
                    forcing_next_time=f_atm(n*atmosphere.dt),
                    alpha=alpha, theta=theta, boundary=0)
            u_atm_interface += [u_atm[0]]
            phi_atm_interface += [phi_atm[0]]

        #geostrophic situation at time t=0:
        u_ocean = np.zeros_like(u_ocean0)
        phi_ocean = np.zeros_like(phi_ocean0)
        u_ocean_interface = [0]
        phi_ocean_interface = [0]

        for n in range(1, N_ocean+1):
            interface_flux = rho_atm / rho_ocean * atmosphere.nu * phi_atm_interface[n]
            phi_ocean, u_ocean = ocean.integrate_in_time(phi_n=phi_ocean, u_n=u_ocean,
                    interface_flux_next_time=interface_flux,
                    forcing_next_time=f_ocean(n*ocean.dt),
                    boundary=0)

            u_ocean_interface += [u_ocean[-1]]
            phi_ocean_interface += [phi_ocean[-1]]

        all_u_ocean_interface += [u_ocean_interface]
        #all_phi_ocean_interface += [phi_ocean_interface]
        all_u_atm_interface += [u_atm_interface]
        all_phi_atm_interface += [phi_atm_interface]

    return np.array(all_u_atm_interface)

def validate_result(ocean, phi, u, T):
    from numpy import sqrt
    from figures import get_discrete_freq
    from numpy.fft import fft, fftshift
    import matplotlib.pyplot as plt
    # on s'intéresse seulement au 2 premiers phi^n:
    # print(fft(phi[:2], axis=0).T[-4:-1])

    N = int(T/ocean.dt)
    N = phi.shape[0] - 1
    w = get_discrete_freq(N, ocean.dt, avoid_zero=False)
    # deja on verifie que phi^{n+1} - phi^{n} = nu / h^2 (phi{n+1} - 2phi + phi)

    C = ocean.dt * ocean.nu / ocean.h**2

    # sum_dx_part = 0
    # sum_dt_part = 0
    # np1_dt_part = 0
    # n_dt_part = 0
    for k in range(phi.shape[0]):
        zn_0 = np.exp(w * 1j * ocean.dt)
        zn = zn_0**k
        z = (zn_0[1]/zn_0[0])**k

        phi_hat = np.sum(zn * phi.T, axis=1)
        final = (1 - z + 2* C) * phi_hat[1:-1] - C * (phi_hat[2:] + phi_hat[:-2])

        ############# THEORY: #########
        s = (1 - z)/ocean.dt
        chi = ocean.h**2 * (s + ocean.r) / ocean.nu
        lambda_j = (chi - sqrt(chi) * sqrt(4+chi))/2

        print(np.linalg.norm(final + zn[0] * phi[-1, 1:-1]))

        depth = np.flipud(np.array(range(phi_hat.shape[0])))
        theory = (lambda_j+1)**depth

        final_theory = (1 - z + 2* C) * theory[1:-1] - C * (theory[2:] + theory[:-2])
        print(np.linalg.norm(final_theory))

        plt.semilogx(np.abs(phi_hat), depth)
        plt.semilogx(np.abs(phi_hat[-1]*(lambda_j+1)**depth), depth, "k--")
        plt.semilogx(np.abs(zn[0] * phi[-1]), depth, "--")

        plt.show()
        input()

    input()



    # phi = phi[:, -50:]
    # freq_err = fft(phi, axis=0)
    # w = get_discrete_freq(N, ocean.dt)
    # z = np.exp(w * 1j * ocean.dt)
    # s = (z - 1) / z / ocean.dt
    # chi = ocean.h**2 * (s + ocean.r) / ocean.nu
    # lambda_j = (chi - sqrt(chi) * sqrt(4+chi))/2

    # C = ocean.nu * ocean.dt / ocean.h**2
    # # plt.semilogy(np.log((((s + ocean.r + 2*C) * freq_err.T[1:-1] - C * (freq_err.T[2:] + freq_err.T[:-2]))).T / phi[-1][1:-1]).T/np.log(z))

    # plt.semilogy(np.abs((s + ocean.r + 2*C) * freq_err.T[1:-1] - C * (freq_err.T[2:] + freq_err.T[:-2])))

    # # plt.semilogy(np.abs(freq_err[2]), label="first freq")
    # # plt.semilogy(np.abs(freq_err[-1]), label="last freq")
    # plt.semilogy(np.abs(phi[-1][1:-1]), "k--")
    # # # plt.semilogy(np.abs(freq_err[-1]), label="last freq")
    # # m = np.array(range(freq_err.shape[1]))
    # # plt.semilogy(np.flipud(m), np.abs(freq_err[-1,-1] * (lambda_j[-1]+1)**m))
    # plt.legend()
    # plt.show()
    # input()


def bulk_schwarz_spinup(builder, T=3600, NUMBER_IT=5, C_D=1.2e-3, theta=1., nonlinear=True, initial_conditions=None):
    """
        Spin-up function, designed for the nonlinear model. 
        Coupling is made between AtmoFdBeFlux and OceanFdBeFlux models.
        builder: all the discretisation paramters.
        T: length of spinup time (s)
        NUMBER_IT: number of iterations, should be at least 2.
        theta: bulk parameters of interface
        nonlinear: if True, override alpha by C_D|u_a - u_o|
    """
    from scipy.interpolate import interp1d
    from atmo_fd_be_flux import AtmoFdBeFlux
    from ocean_fd_be_flux import OceanFdBeFlux
    # atmosphere model: can have specific r, nu, M, size_domain, dt
    atmosphere = AtmoFdBeFlux(r=builder.R, nu=builder.D2,
            M=builder.M2, SIZE_DOMAIN=builder.SIZE_DOMAIN_2, DT = builder.DT)
    # ocean model: can have specific r, nu, M, size_domain, dt
    ocean = OceanFdBeFlux(r=builder.R, nu=builder.D1,
            M=builder.M1, SIZE_DOMAIN=builder.SIZE_DOMAIN_1, DT = builder.DT)

    # large scale, geostrophic speeds are used to create a forcing
    atm_geostrophic_speed = 10.
    ocean_geostrophic_speed = 0.1
    u_atm_geostrophic = atm_geostrophic_speed * np.ones(atmosphere.M - 1) # 10 m.s^(-1)
    u_ocean_geostrophic = ocean_geostrophic_speed * np.ones(ocean.M - 1) # .1 m.s^(-1)

    # number of time steps in each model:
    N_ocean = int(np.round(T/ocean.dt))
    N_atm = int(np.round(T/atmosphere.dt)) # atm stands for atmosphere in the following
    rho_atm = 1.
    rho_ocean = 1e3 # ratio of the densities, rho(atmosphere) / rho(ocean)

    def f_ocean(t): # Ocean is mainly forced by the interface with atmosphere but we use the (small) geostrophic speed
        return ocean.r * u_ocean_geostrophic
    def f_atm(t): # geostrophic forcing of atmosphere
        return atmosphere.r * u_atm_geostrophic

    u_atm0, phi_atm0, u_ocean0, phi_ocean0 = initial_conditions if initial_conditions is not None else linear_steadystate(builder, C_D)

    # no random initialization of the interface.
    u_ocean_interface = u_ocean0[-1] + np.zeros(N_ocean+1)
    phi_ocean_interface = phi_ocean0[-1] + np.zeros(N_ocean+1)
    u_atm_interface = u_atm0[0] + np.zeros(N_atm+1)
    phi_atm_interface = phi_atm0[0] + np.zeros(N_atm+1)

    bulk_interface_k = []
    phi_interface_k = []

    # Beginning of schwarz iterations:
    for k in range(NUMBER_IT+1):
        interpolated_u_ocean = interp1d(x=np.array(range(N_ocean+1))*ocean.dt,
                y=np.array(u_ocean_interface), kind='cubic', bounds_error=False, fill_value=(0., 0.))
        interpolated_u_atm   = interp1d(x=np.array(range(N_atm+1))*atmosphere.dt,
                y=np.array(u_atm_interface),   kind='cubic', bounds_error=False, fill_value=(0., 0.))
        interpolated_phi_atm = interp1d(x=np.array(range(N_atm+1))*atmosphere.dt,
                y=np.array(phi_atm_interface), kind='cubic', bounds_error=False, fill_value=(0., 0.))

        # geostrophic situation at time t=0:
        u_atm = np.copy(u_atm0)
        phi_atm = np.copy(phi_atm0)
        u_atm_interface = [u_atm[0]]
        phi_atm_interface = [phi_atm[0]]
        for n in range(1, N_atm+1):
            if nonlinear:
                alpha = rho_atm * C_D * np.abs(interpolated_u_atm(n*atmosphere.dt) - interpolated_u_ocean(n*atmosphere.dt))
            else:
                alpha = rho_atm * C_D * np.abs(u_atm_geostrophic[0] - u_ocean_geostrophic[-1])

            #interface_info = alpha (theta u^{n-1}/(1+r dt) + (1-theta) u^{k-1} - u_ocean)
            reaction_effect = 1/(1. + atmosphere.r*atmosphere.dt)
            interface_info = alpha*(theta * (u_atm[0] \
                    + atmosphere.r*atmosphere.dt*u_atm_geostrophic[0]) * reaction_effect + \
                 (1.-theta)*interpolated_u_atm(n*atmosphere.dt) - interpolated_u_ocean(n*atmosphere.dt))

            phi_atm, u_atm = atmosphere.integrate_in_time(phi_n=phi_atm, u_n=u_atm,
                    interface_info_next_time=interface_info,
                    forcing_next_time=f_atm(n*atmosphere.dt),
                    alpha=alpha, theta=theta, boundary=atm_geostrophic_speed)
            u_atm_interface += [u_atm[0]]
            phi_atm_interface += [phi_atm[0]]

        #geostrophic situation at time t=0:
        u_ocean = np.copy(u_ocean0)
        phi_ocean = np.copy(phi_ocean0)
        u_ocean_interface = [u_ocean[-1]]
        phi_ocean_interface = [phi_ocean[-1]]

        interpolated_phi_atm = interp1d(x=np.array(range(N_atm+1))*atmosphere.dt,
                y=np.array(phi_atm_interface), kind='cubic', bounds_error=False, fill_value=(0., 0.))
        for n in range(1, N_ocean+1):
            interface_flux = rho_atm / rho_ocean * atmosphere.nu * interpolated_phi_atm(n*ocean.dt)
            phi_ocean, u_ocean = ocean.integrate_in_time(phi_n=phi_ocean, u_n=u_ocean,
                    interface_flux_next_time=interface_flux,
                    forcing_next_time=f_ocean(n*ocean.dt),
                    boundary=ocean_geostrophic_speed)

            u_ocean_interface += [u_ocean[-1]]
            phi_ocean_interface += [phi_ocean[-1]]
        bulk_interface_k += [rho_atm * C_D * np.abs(u_atm[0] - u_ocean[-1])*(u_atm[0] - u_ocean[-1])]
        phi_interface_k += [phi_atm[0]]

    #converting to tuple so that they can be stored in dictionnary and we can use memoised(...)
    return tuple(u_atm), tuple(phi_atm), tuple(u_ocean), tuple(phi_ocean)

def linear_steadystate(builder, C_D):
    from atmo_fd_be_flux import AtmoFdBeFlux
    from ocean_fd_be_flux import OceanFdBeFlux
    # atmosphere model: can have specific r, nu, M, size_domain, dt
    atmosphere = AtmoFdBeFlux(r=builder.R, nu=builder.D2,
            M=builder.M2, SIZE_DOMAIN=builder.SIZE_DOMAIN_2, DT = builder.DT)
    # ocean model: can have specific r, nu, M, size_domain, dt
    ocean = OceanFdBeFlux(r=builder.R, nu=builder.D1,
            M=builder.M1, SIZE_DOMAIN=builder.SIZE_DOMAIN_1, DT = builder.DT)

    # large scale, geostrophic speeds are used to create a forcing
    atm_geostrophic_speed = 10.
    ocean_geostrophic_speed = 0.1
    u_atm_geostrophic = atm_geostrophic_speed * np.ones(atmosphere.M - 1) # 10 m.s^(-1)
    u_ocean_geostrophic = ocean_geostrophic_speed * np.ones(ocean.M - 1) # .1 m.s^(-1)

    rho_atm = 1.
    rho_ocean = 1e3 # ratio of the densities, rho(atmosphere) / rho(ocean)

    def f_ocean(t): # Ocean is mainly forced by the interface with atmosphere but we use the (small) geostrophic speed
        return ocean.r * u_ocean_geostrophic
    def f_atm(t): # geostrophic forcing of atmosphere
        return atmosphere.r * u_atm_geostrophic


    chi_a = atmosphere.r * atmosphere.h**2 / atmosphere.nu
    chi_o = ocean.r * ocean.h**2 / ocean.nu
    lambda_a = (chi_a - np.sqrt(chi_a)*np.sqrt(chi_a+4))/2.
    lambda_o = (chi_o - np.sqrt(chi_o)*np.sqrt(chi_o+4))/2.
    alpha = rho_atm * C_D * (atm_geostrophic_speed - ocean_geostrophic_speed)

    denominator = atmosphere.nu - alpha * (atmosphere.h * lambda_a / chi_a - \
            - rho_atm/rho_ocean * atmosphere.nu / ocean.nu * ocean.h * lambda_o / chi_o)
    C_a = alpha * (atm_geostrophic_speed - ocean_geostrophic_speed) / denominator
    C_o = C_a * rho_atm * atmosphere.nu / rho_ocean / ocean.nu
    phi_a = C_a * (1+lambda_a)**np.array(range(atmosphere.M))
    u_a = np.diff(phi_a) * atmosphere.h / chi_a + u_atm_geostrophic
    phi_o = np.flipud(C_o * (1+lambda_o)**np.array(range(ocean.M)))
    u_o = np.diff(phi_o) * ocean.h / chi_o + u_ocean_geostrophic
    return tuple(u_a), tuple(phi_a), tuple(u_o), tuple(phi_o)

def nonlinear_steadystate(builder, C_D):
    from numpy import sqrt, cbrt
    R = builder.R
    h_a = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    h_o = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    nu_1 = builder.D1
    nu_2  = builder.D2
    #return explicit_part/((1+implicit_part))
    chi_o = R * h_o**2/nu_1
    chi_a = R * h_a**2/nu_2
    lam_a = (chi_a - np.sqrt(chi_a)*np.sqrt(chi_a+4.))/2
    lam_aplus = (chi_a + np.sqrt(chi_a)*np.sqrt(chi_a+4.))/2
    lam_o = (chi_o - np.sqrt(chi_o)*np.sqrt(chi_o+4.))/2
    lam_oplus = (chi_o + np.sqrt(chi_o)*np.sqrt(chi_o+4.))/2
    eps_a = ((lam_a+1) / (lam_aplus + 1))**(builder.M2 - 2)
    eps_o = ((lam_o+1) / (lam_oplus + 1))**(builder.M1 - 2)
    q_a = 1-eps_a*lam_a/lam_aplus
    q_o = 1-eps_o*lam_o/lam_oplus
    tilde_d = (1e3*q_o/q_a * nu_1*lam_a/ (R*h_a)*(1-eps_a)  + nu_1*lam_o/ (R*h_o)*(1-eps_o))
    g = 10 - 0.1
    # change of variable: we look for X = phi_ocean[-1] / C_D * nu_1 * 1e3
    c = 1e3*nu_1/C_D * q_o

    d = tilde_d / c
    g = 10 - .1
    tilde_a = np.real(d)
    b = np.imag(d)
    t = b/g
    a = tilde_a * t

    j = a**2 + b**2 * t**2
    s = 2 * t**6 + 72 * b**4 * t**4 - 36. * a**2 * b**2 * t**2
    beta = t**4 - 12*b**2*j
    alpha = s**2 - 4*beta**3
    gamma = cbrt(s+sqrt(alpha))/(3*cbrt(2)*j) + cbrt(2)*beta / (3*j*cbrt(s+sqrt(alpha)))
    zeta = -2*t**2/(3*j) + a**2*t**2/j**2
    r = np.sqrt(zeta+gamma+0j)

    y = a*t/(2*j) - r/2 + .5*sqrt(2*zeta-gamma - (2*a**3*t**3 - 2*a*j*t**3)/(j**3*r))
    print("y should be real:", y)
    y = a*t/(2*j) - r/2 - .5*sqrt(2*zeta-gamma - (2*a**3*t**3 - 2*a*j*t**3)/(j**3*r))
    print("y should be real:", y)
    y = a*t/(2*j) + r/2 - .5*sqrt(2*zeta-gamma + (2*a**3*t**3 - 2*a*j*t**3)/(j**3*r))
    print("y should be real:", y)
    y = a*t/(2*j) + r/2 + .5*sqrt(2*zeta-gamma + (2*a**3*t**3 - 2*a*j*t**3)/(j**3*r))
    print("y should be real:", y)

    x = (g - b*t*y**4)/(1 - a*y/t)
    A_1 = (x + t*y**3*1j - g)/tilde_d
    A_2 = 1e3 * nu_1 / nu_2 * q_o / q_a * A_1
    # now U_1 = U_1^g - A_1 * nu_1 * ((lam_o+1)**(m+1) - (lam_o+1)**m) / (if h_o) 
    # now U_2 = U_2^g - A_2 * nu_2 * ((lam_a+1)**(m+1) - (lam_a+1)**m) / (if h_a) 
    u_1 = tuple(0.1 - A_1 * nu_1 * ((lam_o+1)**m-eps_o*(lam_oplus+1)**m) * lam_o / (R*h_o) for m in range(builder.M1-2, -1, -1))
    u_2 = tuple(10. + A_2 * nu_2 * ((lam_a+1)**m-eps_a*(lam_aplus+1)**m) * lam_a / (R*h_a) for m in range(builder.M2-1))
    phi_1 = tuple(A_1 * ((lam_o+1)**m-eps_o*lam_o/lam_oplus*(lam_oplus+1)**m) for m in range(builder.M1-1, -1, -1))
    phi_2 = tuple(A_2 * ((lam_a+1)**m-eps_a*lam_a/lam_aplus*(lam_aplus+1)**m) for m in range(builder.M2))
    return tuple(u_2), tuple(phi_2), tuple(u_1), tuple(phi_1)
