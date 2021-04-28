#!/usr/bin/python3
"""
    This module provides functions to observe real convergence factors.
"""
import functools
import numpy as np

def frequency_simulation(atmosphere, ocean, number_samples=100, laplace_real_part=0., **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        The errors are of the form ocean.Lambda * u_2(x=0) + nu phi_2(x=0)

        Every argument should be given in models atmosphere and ocean, except **kwargs
          kwargs = {T, NUMBER_IT}
        T: time window
        NUMBER_IT: number of Schwarz iterations (should be at least > 3)
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    # we put two times the same discretization, the code is evolutive
    to_map = functools.partial(schwarz_simulator, atmosphere, ocean, **kwargs)

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        err_ret = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            err_ret += [result]
        errors = np.array(err_ret)

    errors *= np.exp(- laplace_real_part * atmosphere.dt * np.array(range(errors.shape[-1])))
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

def schwarz_simulator(atmosphere, ocean, seed=9380, T=3600, NUMBER_IT=3, init="white", overlap=0):
    """
        Returns errors at interface from beginning (first guess) until the end.
        Coupling is made between "atmosphere" (instanciated from atmosphere_models)
        and "ocean" (instanciated from ocean_models)
        seed: random parameter for initializing first guess
        T: length of time window (s)
        NUMBER_IT: number of iterations, should be at least 3.
        We only look at the errors around the solution.
        the equations are then homogenous equation.
        init must be in { "white", "dirac", "GP" }
        overlap is the (non-negative integer) number of grid points that overlap
    """
    # number of time steps in each model:
    N_ocean = int(np.round(T/ocean.dt))
    N_atm = int(np.round(T/atmosphere.dt)) # atm stands for atmosphere in the following
    assert N_ocean == N_atm # different time steps are not supported yet
    assert ocean.M == atmosphere.M and abs(ocean.size_domain - atmosphere.size_domain)<1e-10
    p1, p2 = ocean.Lambda, atmosphere.Lambda

    def f_ocean(t): # Size of u, whether it is diagnostic or prognostic variable
        return np.zeros(ocean.size_u())
    def f_atm(t): # Size of u, whether it is diagnostic or prognostic variable
        return np.zeros(atmosphere.size_u())

    def get_star(val_n, val_np1):
        return val_np1 + (1+1/np.sqrt(2))*(val_n - val_np1)
    def give_tuple_star(val_n, val_np1):
        return val_n, get_star(val_n, val_np1), val_np1

    # random initialization of the interface: we excite the high frequencies with white noise
    np.random.seed(seed)
    if init == "white":
        interface_ocean = ((np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))) + 1j*(np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))))
        interface_atm = ((np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))) + 1j*(np.concatenate(([0], 2 * (np.random.rand(N_atm) - 0.5)))))
    elif init == "GP":
        cov = np.array([[ np.exp(-.1*np.abs(i-j)) for i in range(N_atm)] for j in range(N_atm)])
        rand1, rand2 = np.random.default_rng().multivariate_normal(np.zeros(N_atm), cov, 2)
        interface_atm = np.concatenate(([0], rand1))
        interface_ocean = np.concatenate(([0], rand2))
    elif init == "dirac":
        interface_ocean = np.concatenate(([0, np.random.rand(1)[0]], np.zeros(N_ocean-1)))
        interface_atm = np.concatenate(([0, np.random.rand(1)[0]], np.zeros(N_atm-1)))
    else:
        raise



    all_interface_atm = [interface_atm]

    # Beginning of schwarz iterations:
    for k in range(NUMBER_IT+1):
        # integration in time of the atmosphere model:
        diagnosed = np.zeros(atmosphere.M - 1)
        prognosed = np.zeros(atmosphere.M)

        interface_atm = [0]
        for n in range(1, N_atm+1):
            robin = give_tuple_star(interface_ocean[n-1], interface_ocean[n])
            forcing = give_tuple_star(f_atm((n-1)*atmosphere.dt), f_atm(n*atmosphere.dt))
            prognosed, diagnosed = atmosphere.integrate_in_time(prognosed=prognosed, diagnosed=diagnosed,
                    interface_robin=robin, forcing=forcing, boundary=(0,0,0))

            u_interface, phi_interface = atmosphere.interface_values(prognosed, diagnosed, overlap)
            interface_atm += [p1*u_interface + atmosphere.nu*phi_interface]

        # integration in time of the ocean model:
        diagnosed = np.zeros(ocean.M - 1)
        prognosed = np.zeros(ocean.M)

        interface_ocean = [0]

        for n in range(1, N_ocean+1):
            robin = give_tuple_star(interface_atm[n-1], interface_atm[n])
            forcing = give_tuple_star(f_ocean((n-1)*atmosphere.dt), f_ocean(n*atmosphere.dt))
            prognosed, diagnosed = ocean.integrate_in_time(prognosed=prognosed, diagnosed=diagnosed,
                    interface_robin=robin, forcing=forcing, boundary=(0., 0., 0.))
            u_interface, phi_interface = ocean.interface_values(prognosed, diagnosed, overlap)
            interface_ocean += [p2*u_interface + ocean.nu * phi_interface]

        all_interface_atm += [interface_atm]

    return np.array(all_interface_atm)

