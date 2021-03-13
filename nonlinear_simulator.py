#!/usr/bin/python3
"""
    This module provides functions to observe real convergence factors.
"""
import time
import functools
import numpy as np
from numpy import pi

def frequency_simulation(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors, like NUMBER_IT
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    T = N * discretization.DT
    # we put two times the same discretization, the code is evolutive
    to_map = functools.partial(interface_errors, discretization, discretization, T, **kwargs)

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        u_ret = []
        for result in progressbar(executor.map(to_map, range(number_samples))): #TODO enlever executor quand c'est vraiment nonlinéaire
            # No parallelism with non-linear, because alpha_nonlinear becomes part of dis
            u_ret += [result]
        u_ret = np.array(u_ret)

    errors = np.array([[u_k - 0 for u_k in u_it[:-1]] for u_it in u_ret])

    freq_err = fftshift(fft(errors, axis=-1), axes=(-1, ))
    return np.std(freq_err, axis=0)


def interface_errors(discretization_1,
                     discretization_2,
                     T,
                     seed=9380,
                     NUMBER_IT=2):
    """
        returns errors at interface from beginning (first guess) until the end.
    """
    period = 24 * 60 * 60.
    def f1(t):
        return np.zeros(discretization_1.size_f(upper_domain=False))
    def f2(t):
        return np.zeros_like(np.linspace(1.2,0.3, discretization_2.size_f(upper_domain=True)) * np.cos(period * t / 2*np.pi))
    def neumann(t):
        return 0.
    def dirichlet(t):
        return 0.

    precomputed_Y1 = discretization_1.precompute_Y(upper_domain=False)
    precomputed_Y2 = discretization_2.precompute_Y(upper_domain=True)

    u1_0 = np.zeros(discretization_1.size_prognostic(upper_domain=False))
    u2_0 = np.zeros(discretization_2.size_prognostic(upper_domain=True))
    # random false initialization:
    np.random.seed(seed)
    N1 = int(np.round(T/discretization_1.DT))
    N2 = int(np.round(T/discretization_2.DT))
    dt1 = discretization_1.DT
    dt2 = discretization_2.DT
    all_u1_interface = [np.concatenate(([0], 2 * (np.random.rand(N1) - 0.5)))]
    all_phi1_interface = [np.concatenate(([0], 2 * (np.random.rand(N1) - 0.5)))]
    all_u2_interface = [np.concatenate(([0], 2 * (np.random.rand(N2) - 0.5)))]
    all_phi2_interface = [np.concatenate(([0], 2 * (np.random.rand(N2) - 0.5)))]

    from scipy.interpolate import interp1d
    interpolators_u1 = [interp1d(x=np.array(range(N1+1))*dt1, y=np.zeros_like(all_u1_interface[0]),
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]
    interpolators_u2 = [interp1d(x=np.array(range(N2+1))*dt2, y=np.zeros_like(all_u2_interface[0]),
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]

    interpolators_phi1 = [interp1d(x=np.array(range(N1+1))*dt1, y=np.zeros_like(all_phi1_interface[0]),
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]
    interpolators_phi2 = [interp1d(x=np.array(range(N2+1))*dt2, y=np.zeros_like(all_phi2_interface[0]),
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]

    # Beginning of schwarz iterations:
    for k in range(NUMBER_IT):
        u2_interface = [0]
        phi2_interface = [0]
        last_u2 = u2_0
        additional = []
        # Time iteration:
        interpolators_u1 += [interp1d(x=np.array(range(N1+1))*dt1, y=all_u1_interface[-1],
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]
        interpolators_phi1 += [interp1d(x=np.array(range(N1+1))*dt1, y=all_phi1_interface[-1],
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]

        for i in range(1, N2+1):
            u_interface_ti = [lambda t : interpolator((i-1+t)*dt2) for interpolator in interpolators_u1]
            phi_interface_ti = [lambda t : interpolator((i-1+t)*dt2) for interpolator in interpolators_phi1]
            u2_interface_ti = [lambda t : interpolator((i-1+t)*dt2) for interpolator in interpolators_u2]
            phi2_interface_ti = [lambda t : interpolator((i-1+t)*dt2) for interpolator in interpolators_phi2]
            current_f2 = lambda t : f2((i-1+t)*dt2)

            u2_ret, u_interface, phi_interface, *additional = \
                    discretization_2.integrate_one_step(
                f=current_f2,
                bd_cond=neumann,
                u_nm1=last_u2,
                u_interface=u_interface_ti,
                phi_interface=phi_interface_ti,
                additional=additional,
                upper_domain=True,
                Y=precomputed_Y2,
                selfu_interface=u2_interface_ti,
                selfphi_interface=phi2_interface_ti)
            last_u2 = u2_ret
            u2_interface += [u_interface]
            phi2_interface += [phi_interface]

        all_u2_interface += [u2_interface]
        all_phi2_interface += [phi2_interface]

        u1_interface = [0]
        phi1_interface = [0]
        last_u1 = u1_0
        additional = []

        interpolators_u2 += [interp1d(x=np.array(range(N2+1))*dt2, y=all_u2_interface[-1],
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]
        interpolators_phi2 += [interp1d(x=np.array(range(N2+1))*dt2, y=all_phi2_interface[-1],
                kind='cubic', bounds_error=False, fill_value=(0., 0.))]
        for i in range(1, N1+1):
            u_interface_ti = [lambda t : interpolator((i-1+t)*dt1) for interpolator in interpolators_u2]
            phi_interface_ti = [lambda t : interpolator((i-1+t)*dt1) for interpolator in interpolators_phi2]
            u1_interface_ti = [lambda t : interpolator((i-1+t)*dt1) for interpolator in interpolators_u1]
            phi1_interface_ti = [lambda t : interpolator((i-1+t)*dt1) for interpolator in interpolators_phi1]
            f1_interface_ti = [lambda t : interpolator((i-1+t)*dt1) for interpolator in interpolators_phi1]
            current_f1 = lambda t : f1((i-1+t)*dt1)

            u1_ret, u_interface, phi_interface, *additional = \
                    discretization_1.integrate_one_step(
                f=current_f1,
                bd_cond=dirichlet,
                u_nm1=last_u1,
                u_interface=u_interface_ti,
                phi_interface=phi_interface_ti,
                additional=additional,
                upper_domain=False,
                Y=precomputed_Y1,
                selfu_interface=u1_interface_ti,
                selfphi_interface=phi1_interface_ti,
                )
            last_u1 = u1_ret
            u1_interface += [u_interface]
            phi1_interface += [phi_interface]

        all_u1_interface += [u1_interface]
        all_phi1_interface += [phi1_interface]
        #input()
    ret = discretization_2.LAMBDA_2 * np.array(all_u1_interface) + np.array(all_phi1_interface)

    return ret


if __name__ == "__main__":
    import main
    main.main()
