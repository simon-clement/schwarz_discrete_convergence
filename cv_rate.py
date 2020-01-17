#!/usr/bin/python3
"""
    This module computes theoric convergence rates and
    provide functions to observe real convergence rate.
"""
import time
import functools
import numpy as np
from numpy import pi
from discretizations import finite_difference_naive_neumann
from discretizations import rk4_finite_differences
from discretizations import rk4_finite_volumes
from discretizations import rk2_finite_differences
from discretizations import finite_difference
from discretizations import finite_difference_no_corrective_term
from discretizations import finite_volumes

#########################################################################
# THEORIC PART : RETURN RATES YIELDED BY ANALYSIS IN FREQUENTIAL DOMAIN #
#########################################################################

def continuous_analytic_rate_robin_neumann(discretization, Lambda_1, w):
    """
        Returns the convergence rate predicted by continuous analysis.
        The interface condition is Robin-Neumann.
        This is equivalent to call continuous_analytic_rate_robin_robin
        with the parameter Lambda_2=0.
    """
    return continuous_analytic_rate_robin_robin(discretization, Lambda_1, 0., w)


def continuous_analytic_rate_robin_robin(discretization, Lambda_1, Lambda_2,
                                         w):
    """
        Returns the convergence rate predicted by continuous analysis.
        The equation is diffusion-reaction with constant coefficients.
        The interface condition is Robin-Robin.
        w is the frequency;
        Lambda_{1,2} are the Robin condition free parameters.
        discretization must have the following attributes:
        discretization.D1_DEFAULT : diffusivity in \\Omega_1
        discretization.D2_DEFAULT : diffusivity in \\Omega_2
        discretization.SIZE_DOMAIN_1 : Size of \\Omega_1
        discretization.SIZE_DOMAIN_2 : Size of \\Omega_2
        discretization.C_DEFAULT : reaction coefficient (may be complex or real)
    """

    D1 = discretization.D1_DEFAULT
    D2 = discretization.D2_DEFAULT
    H1 = - discretization.SIZE_DOMAIN_1
    H2 = discretization.SIZE_DOMAIN_2
    c = discretization.C_DEFAULT

    # sig1 is \sigma^1_{-}
    sig1 = np.sqrt((w*1j + c) / D1)
    sig2 = np.sqrt((w*1j + c) / D2)

    #This line is necessary because we have sig1=\sigma^{-} and sig2=\sigma^{+}
    #Lambda_2 = - Lambda_2 OR sig1 = -sig1
    sig2 = -sig2

    first_term = (D2*sig2+Lambda_1)/ \
            (D1*sig1 + Lambda_1)
    second = (D1*sig1 + Lambda_2)/ \
            (D2*sig2 + Lambda_2)
    return np.abs(first_term * second)


def continuous_analytic_rate_robin_robin_modified_only_eq(discretization, Lambda_1, Lambda_2,
                                         w):
    """
        Returns the convergence rate predicted by continuous analysis with modified equations..
    """

    D1 = discretization.D1_DEFAULT
    D2 = discretization.D2_DEFAULT
    assert(D1==D2)
    H1 = - discretization.SIZE_DOMAIN_1
    H2 = discretization.SIZE_DOMAIN_2
    c = discretization.C_DEFAULT
    assert(c==0)
    h1 = H1/(discretization.M1_DEFAULT - 1)
    if discretization.name() == finite_volumes.FiniteVolumes().name():
        h1 = H1/discretization.M1_DEFAULT
    dt = discretization.DT_DEFAULT

    c_mod = h1**2 / (12*D1**2) + dt / 2

    # sig1 is \sigma^1_{-}
    sig1 = np.sqrt((w*1j + c_mod*w**2 + c) / D1)
    sig2 = np.sqrt((w*1j + c_mod*w**2 + c) / D2)

    #This line is necessary because we have sig1=\sigma^{-} and sig2=\sigma^{+}
    #Lambda_2 = - Lambda_2 OR sig1 = -sig1
    sig2 = -sig2

    first_term = (D2*sig2+Lambda_1)/ \
            (D1*sig1 + Lambda_1)
    second = (D1*sig1 + Lambda_2)/ \
            (D2*sig2 + Lambda_2)
    return np.abs(first_term * second)


def continuous_analytic_rate_robin_robin_modified_only_eq_simple_formula(discretization, Lambda_1, Lambda_2,
                                         w):
    """
        Returns the convergence rate predicted by continuous analysis with modified equations.
        It is called simple formula because there is no complex number involved in the computations.
    """

    D1 = discretization.D1_DEFAULT
    D2 = discretization.D2_DEFAULT
    assert D1 == D2
    H1 = - discretization.SIZE_DOMAIN_1
    H2 = discretization.SIZE_DOMAIN_2
    c = discretization.C_DEFAULT
    assert c == 0
    h1 = H1/(discretization.M1_DEFAULT - 1)
    if discretization.name() == finite_volumes.FiniteVolumes().name():
        h1 = H1/discretization.M1_DEFAULT
    dt = discretization.DT_DEFAULT

    c_mod = h1**2 / (12*D1**2) + dt / 2
    assert c_mod > 0

    Lambda_2 /= np.sqrt(D1)
    Lambda_1 /= np.sqrt(D1) # we remove all the D from the convergence rate thanks to this
    # Now we have Lambda_{1,2}+sqrt(iw+cw^2)
    #let's multiply everywhere by sqrt(c_mod) :
    Lambda_1 *= np.sqrt(c_mod)
    Lambda_2 *= np.sqrt(c_mod)
    w *= c_mod
    w = w**2

    # We now have Lambda_{1,2} + sqrt(iw + w^2)
    def f(pm, l1_or_2):
        return np.sqrt(2)*pm*l1_or_2*np.sqrt(w+np.sqrt(w*(1+w)))+np.sqrt(w*(1+w)) + l1_or_2**2

    # This should be equal to the output of continuous_analytic_rate_robin_robin_modified_only_eq.
    return np.sqrt(f(-1, Lambda_1)*f(1, Lambda_2) / (f(1, Lambda_1) * f(-1, Lambda_2)))


def continuous_best_lam_robin_neumann(discretization, N):
    """
        Returns the optimal Robin free parameter according to
        continuous analysis of the convergence rate.
        The equation is pure diffusion.
        N is the number of time steps of the window.
        discretization must have the following attributes:
        discretization.D1_DEFAULT : diffusivity in \\Omega_1
        discretization.D2_DEFAULT : diffusivity in \\Omega_2
        discretization.DT_DEFAULT : time step
        It is assumed that the size of the domains infinite.
    """
    sqD1 = np.sqrt(discretization.D1_DEFAULT)
    sqD2 = np.sqrt(discretization.D2_DEFAULT)
    dt = discretization.DT_DEFAULT
    T = dt * N
    sqw1 = np.sqrt(pi / T)
    sqw2 = np.sqrt(pi / dt)
    return 1 / (2 * np.sqrt(2)) * ((sqD2 - sqD1) * (sqw1 + sqw2) + np.sqrt(
        (sqD2 - sqD1)**2 * (sqw1 + sqw2)**2 + 8 * sqD1 * sqD2 * sqw1 * sqw2))


def continuous_best_lam_robin_onesided_modif_vol(discretization, dt, courant_number, wmin, wmax):
    from figures import figProjectionComplexPlan

    assert discretization.name() == finite_volumes.FiniteVolumes().name()
    assert discretization.D1_DEFAULT == discretization.D2_DEFAULT
    def rho(lam, a, b):
        return ((lam-a)**2 + b**2)/((lam+a)**2 + b**2)

    D = discretization.D1_DEFAULT
    facteur_transfo = dt *(1/(12*D * courant_number) + 1/2)
    xi_min = wmin * facteur_transfo
    xi_max = wmax * facteur_transfo
    amin = np.real(np.sqrt(xi_min*1j + xi_min**2))
    amax = np.real(np.sqrt(xi_max*1j + xi_max**2))
    bmax = np.imag(np.sqrt(xi_max*1j + xi_max**2))
    lam_1 = np.sqrt(amin**2 + bmax**2)
    lam_2 = np.sqrt(amax**2 + bmax**2)

    if amin*amax < bmax**2:
        print("cas amin*amax < bmax")
        return np.sqrt(D)*lam_1/np.sqrt(facteur_transfo), rho(lam_1, amin, bmax)

    lam_l = np.sqrt(amin*amax - bmax**2)

    if lam_l < lam_1:
        assert rho(lam_2, amax, bmax) < rho(lam_2, amin, bmax)
        print("cas lam_l < lam_1 -> on prend amin, bmax (cas le plus courant)")
        ret = lam_1
    elif lam_l < lam_2:
        print("cas lam_1 < lam_l < lam_2 -> on prend amin ou amax")
        print("lam_2 =", lam_2, "LAM_l =", lam_l, "lam_1 = ", lam_1)
        assert rho(lam_2, amax, bmax) < rho(lam_2, amin, bmax)
        assert rho(lam_1, amax, bmax) > rho(lam_1, amin, bmax)
        ret = lam_l
    else: # lam_l > lam2:
        print("cas lam_l > lam_2 -> on prend amax, bmax (cas le moins courant)")
        assert rho(lam_1, amax, bmax) > rho(lam_1, amin, bmax)
        ret = lam_2

    #figProjectionComplexPlan(ret, xi_min, xi_max)
    return np.sqrt(D)*ret / np.sqrt(facteur_transfo), max(rho(ret, amax, bmax), rho(ret, amin, bmax))


def rate_by_z_transform(discretization, Lambda_1, NUMBER_SAMPLES):
    """
        This is an attempt to find the convergence rate in time domain
        without making a simulation. The inverse Z transform is very
        hard to do and the parameter r is hard to find.
        Problem : if r is 1, the function is not inside the convergence ray.
        if r > 1, there are numeric instabilities and the signal is false.
        This problem is explained by Ehrhardt, M. in his paper
        "Discrete transparent boundary conditions for parabolic equations"
    """
    all_points = np.linspace(0, 2 * pi, NUMBER_SAMPLES, endpoint=False)
    dt = discretization.DT_DEFAULT
    def z_transformed(z): return discretization.analytic_robin_robin(
        s=1. / dt * (z - 1) / z, Lambda_1=Lambda_1)
    r = 1.001
    samples = [z_transformed(r * np.exp(p * 1j)) for p in all_points]
    ret_ifft = np.fft.ifft(np.array(samples))
    rate_each_time = [r**n * l for n, l in enumerate(ret_ifft)]
    return np.max(np.abs(rate_each_time))


def analytic_robin_robin(discretization,
                         w=None,
                         Lambda_1=None,
                         Lambda_2=None,
                         a=None,
                         c=None,
                         dt=None,
                         M1=None,
                         M2=None,
                         D1=None,
                         D2=None,
                         verbose=False,
                         semi_discrete=False,
                         modified_time=0,
                         N=None):
    """
        returns the theoric discrete/semi-discrete convergence rate.
        The equation is advection-diffusion-reaction with constant coefficients.
        w is the frequency: if w is None, then the local-in-time rate is returned.
        Lambda_{1,2} are the free robin parameters.
        a is the advection coefficient.
        c is the reaction coefficient.
        dt is the time step.
        M_j is the number of points in \\Omega_j
        The size of the domains must be given in discretization:
        discretization.SIZE_DOMAIN_{1,2} are the size of the domains \\omega{1,2}
        D_j is the diffusivity in \\Omega_j
        if w is not None and semi-discrete is True, returns semi-discrete analysis
        if w is not None and semi-discrete is False, returns discrete analysis (with Z transform)
        N is the number of time steps in the time window.

        modified time is the order of approximation in the modified equation (only in dt)

        Main theoric function of the module. It is just a call
        to the good discretization.
    """
    if dt is None:
        dt = discretization.DT_DEFAULT
    if w is None:
        s = 1. / dt
    else:
        if semi_discrete:
            s = discretization.s_time_modif(w, dt, modified_time)
        else:
            # Note : in full discrete case, the case N odd / even must be separated
            # if N % 2 == 0: # even
            #     all_k = np.linspace(-N/2, N/2 - 1, N)
            # else: #odd
            #     all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
            # w = 2 pi k / (N)
            z = 1 * np.exp(w * 1j * dt)
            #   raise
            #k = w
            #z = 1.0 * np.exp(2*k * 1j*np.pi / N)

            s = 1. / dt * (z - 1) / z

    return discretization.analytic_robin_robin(s=s,
                                               Lambda_1=Lambda_1,
                                               Lambda_2=Lambda_2,
                                               a=a,
                                               c=c,
                                               dt=dt,
                                               M1=M1,
                                               M2=M2,
                                               D1=D1,
                                               D2=D2,
                                               verbose=verbose)

#########################################################################
# SIMULATION PART : SOLVE THE SYSTEM OF ERROR AND RETURN RATES          #
#########################################################################


def rate_fast(discretization,
              N,
              Lambda_1=None,
              Lambda_2=None,
              a=None,
              c=None,
              dt=None,
              M1=None,
              M2=None,
              function_to_use=lambda x: max(np.abs(x)),
              seeds=range(100)):
    """
        Makes a simulation and gives the convergence rate.
        N is the number of time steps.

        Lambda_{1,2} are the free robin parameters.
        a is the advection coefficient.
        c is the reaction coefficient.
        dt is the time step.
        M_j is the number of points in \\Omega_j

        The size of the domains must be given in discretization:
        discretization.SIZE_DOMAIN_{1,2} are the size of the domains \\Omega{1,2}
        The diffusivities of the domains must be given in discretization:
        discretization.D{1,2}_DEFAULT are the diffusivities of the domains

        Note that it would be easy to extend this function to variable D,
        by giving to rust_mod.errors the arguments function_D{1,2}.

        function_to_use can be max for L^\\infty or np.linalg.norm for L^2
        This function use a lot of different simulations with random
        first guess to get a good convergence rate.
        uses the rust module to be faster than python
    """
    try:
        import rust_mod
        errors = rust_mod.errors(discretization,
                                 N,
                                 Lambda_1,
                                 Lambda_2,
                                 a,
                                 c,
                                 dt,
                                 M1,
                                 M2,
                                 number_seeds=len(list(seeds)),
                                 function_D1=None,
                                 function_D2=None)
    except BaseException:
        PARALLEL = False
        print("Cannot use rate_fast. Did you compile rust module ?" +
              "Using pure python...")
        errors = None
        to_map = functools.partial(rate_one_seed,
                                   discretization,
                                   N,
                                   function_to_use=function_to_use,
                                   Lambda_1=Lambda_1,
                                   Lambda_2=Lambda_2,
                                   a=a,
                                   c=c,
                                   dt=dt,
                                   M1=M1,
                                   M2=M2)
        if PARALLEL:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                errors = np.mean(np.array(list(executor.map(to_map, seeds))),
                                 axis=0)
        else:
            errors = np.mean(np.abs(np.array(list(map(to_map, seeds)))),
                             axis=0)

    return function_to_use(errors[2]) / function_to_use(errors[1])

def rate_slow(discretization,
              N,
              Lambda_1=None,
              Lambda_2=None,
              a=None,
              c=None,
              dt=None,
              M1=None,
              M2=None,
              function_to_use=lambda x: max(np.abs(x)),
              seeds=range(100)):
    """
        see @rate_fast.
        This function is the same but without any call to rust.
        It is therefore slower, but it works without a doubt.
    """
    PARALLEL = False
    print("Using rate_slow.")
    errors = None
    to_map = functools.partial(rate_one_seed,
                               discretization,
                               N,
                               function_to_use=function_to_use,
                               Lambda_1=Lambda_1,
                               Lambda_2=Lambda_2,
                               a=a,
                               c=c,
                               dt=dt,
                               M1=M1,
                               M2=M2)
    if PARALLEL:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            errors = np.mean(np.array(list(executor.map(to_map, seeds))),
                             axis=0)
    else:
        errors = np.mean(np.abs(np.array(list(map(to_map, seeds)))),
                         axis=0)

    return function_to_use(errors[2]) / function_to_use(errors[1])


def rate_freq(discretization,
              N,
              Lambda_1=None,
              Lambda_2=None,
              a=None,
              c=None,
              dt=None,
              M1=None,
              M2=None,
              function_to_use=lambda x: max(np.abs(x)),
              seeds=range(100)):
    """
        See @rate_fast.
        It is the same but the rate is computed from the frequencial errors:
        a fft is done to consider errors in frequencial domain.
    """
    try:
        raise
        import rust_mod
        errors = rust_mod.errors_raw(discretization,
                                 N,
                                 Lambda_1,
                                 Lambda_2,
                                 a,
                                 c,
                                 dt,
                                 M1,
                                 M2,
                                 number_seeds=len(list(seeds)),
                                 function_D1=None,
                                 function_D2=None)
        from numpy.fft import fft, fftshift
        freq_err = fftshift(fft(errors, norm="ortho", axis=-1), axes=(-1, ))

        errors = np.mean(np.abs(freq_err), axis=0)
        return function_to_use(errors[2]) / function_to_use(errors[1])
    except BaseException:
        return rate_freq_slow(discretization,
                              N,
                              Lambda_1,
                              Lambda_2,
                              a,
                              c,
                              dt,
                              M1,
                              M2,
                              function_to_use,
                              seeds)

def rate_freq_slow(discretization,
                   N,
                   Lambda_1=None,
                   Lambda_2=None,
                   a=None,
                   c=None,
                   dt=None,
                   M1=None,
                   M2=None,
                   function_to_use=lambda x: max(np.abs(x)),
                   seeds=range(100)):
    """
        See @rate_slow.
        It is the same but the rate is computed from the frequencial errors:
        a fft is done to consider errors in frequencial domain.
    """
    PARALLEL = False
    print("Using rate_freq_slow...")
    errors = None
    to_map = functools.partial(rate_one_seed,
                               discretization,
                               N,
                               function_to_use=function_to_use,
                               Lambda_1=Lambda_1,
                               Lambda_2=Lambda_2,
                               a=a,
                               c=c,
                               dt=dt,
                               M1=M1,
                               M2=M2)
    if PARALLEL:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            errors = list(executor.map(to_map, seeds))
    else:
        errors = list(map(to_map, seeds))

    from numpy.fft import fft, fftshift
    freq_err = fftshift(fft(errors, norm="ortho", axis=-1), axes=(-1, ))

    errors = np.mean(np.abs(freq_err), axis=0)
    return function_to_use(errors[2]) / function_to_use(errors[1])

def rate_one_seed(discretization, N, seed, function_to_use=max, **kwargs):
    """
        Warning: function_to_use won't be used.
        Simulation with a single first guess created with the seed.
        Do not use it directly, prefer using rate_slow with seeds=range(1).
        The name is not explicit since it just calls interface_errors...
        Maybe it would be good to replace rate_one_seed by interface_errors?
    """
    return interface_errors(discretization, N, seed=seed, **kwargs)


def raw_simulation(discretization, N, number_samples=1000, **kwargs):
    """
        Simulate and returns directly errors in time domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors:
        Lambda_1, Lambda_2, a, c, dt, M1, M2,
    """
    try:
        import rust_mod
        time_start = time.time()
        errors = rust_mod.errors_raw(discretization,
                                     N,
                                     number_seeds=number_samples,
                                     **kwargs)
        print("took", time.time() - time_start, "seconds")
        return np.mean(np.abs(errors), axis=0)
    except BaseException:
        print("cannot make a fast raw simulation... Going to pure python " +
              "(but it will take some time)")
        import concurrent.futures
        to_map = functools.partial(interface_errors, discretization, N,
                                   **kwargs)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            errors = np.array(list(executor.map(to_map,
                                                range(number_samples))))
        return np.mean(np.abs(errors), axis=0)


def frequency_simulation(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors:
        Lambda_1, Lambda_2, a, c, dt, M1, M2,
    """
    try:
        raise
        import rust_mod
        errors = rust_mod.errors_raw(discretization,
                                     N,
                                     number_seeds=number_samples,
                                     **kwargs)
        freq_err = np.fft.fftshift(np.fft.fft(errors, axis=-1), axes=(-1, ))
        return np.mean(np.real(freq_err), axis=0)
    except:
        print( "Cannot make a fast frequency simulation..." +
               "Going to pure python (but it will take some time)")
        return frequency_simulation_slow(discretization, N, number_samples, **kwargs)

def frequency_simulation_slow(discretization, N, number_samples=100, **kwargs):
    """
        See @frequency_simulation.
        This function can be used if you are not sure of the results of the rust module.
        kwargs can contain any argument of interface_errors:
        Lambda_1, Lambda_2, a, c, dt, M1, M2,
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    to_map = functools.partial(interface_errors, discretization, N,
                               **kwargs)
    print(number_samples)

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        errors = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            errors += [result]
    freq_err = fftshift(fft(np.array(errors), axis=-1), axes=(-1, ))
    return np.std(freq_err, axis=0)


def interface_errors(discretization,
                     time_window_len,
                     seed=9380,
                     Lambda_1=None,
                     Lambda_2=None,
                     a=None,
                     c=None,
                     dt=None,
                     M1=None,
                     M2=None,
                     NUMBER_IT=2):
    """
        returns errors at interface from beginning (first guess) until the end.
        to get rate, just use the following code:
        def rate(*args, function_to_use=max):
            errors = interface_errors(*args)
            errors = [function_to_use([abs(e) for e in err]) for err in errors]
            return errors[2]/errors[1]
        for details on the arguments, you can see for instance @rate_fast
    """
    if M1 is None:
        M1 = discretization.M1_DEFAULT
    if M2 is None:
        M2 = discretization.M2_DEFAULT
    if Lambda_1 is None:
        Lambda_1 = discretization.LAMBDA_1_DEFAULT
    if Lambda_2 is None:
        Lambda_2 = discretization.LAMBDA_2_DEFAULT
    h1, h2 = discretization.get_h(discretization.SIZE_DOMAIN_1,
                                  discretization.SIZE_DOMAIN_2, M1, M2)
    D1, D2 = discretization.get_D(h1, h2)

    f1 = np.zeros(M1)
    f2 = np.zeros(M2)
    neumann = 0
    dirichlet = 0

    precomputed_Y1 = discretization.precompute_Y(M=M1,
                                                 h=h1,
                                                 D=D1,
                                                 a=a,
                                                 c=c,
                                                 dt=dt,
                                                 f=f1,
                                                 bd_cond=dirichlet,
                                                 Lambda=Lambda_1,
                                                 upper_domain=False)

    precomputed_Y2 = discretization.precompute_Y(M=M2,
                                                 h=h2,
                                                 D=D2,
                                                 a=a,
                                                 c=c,
                                                 dt=dt,
                                                 f=f2,
                                                 bd_cond=neumann,
                                                 Lambda=Lambda_2,
                                                 upper_domain=True)

    u1_0 = np.zeros(M1)
    u2_0 = np.zeros(M2)
    phi1_0_fvol = np.zeros(M1 + 1)
    phi2_0_fvol = np.zeros(M2 + 1)
    # random false initialization:
    np.random.seed(seed)
    all_u1_interface = np.concatenate(([0], 2 * (np.random.rand(time_window_len) - 0.5)))
    all_phi1_interface = np.concatenate(([0], 2 * (np.random.rand(time_window_len) - 0.5)))
    #all_u1_interface[-1] /= 1000
    #all_phi1_interface[-1] /= 1000
    ret = [all_u1_interface[1:]]
    # Beginning of schwarz iterations:
    from scipy.interpolate import interp1d
    for k in range(NUMBER_IT):
        all_u2_interface = [0]
        all_phi2_interface = [0]
        all_u2 = [u2_0]
        phi_for_FV = [phi2_0_fvol]
        # Time iteration:
        interpolator_u1 = interp1d(x=np.array(range(time_window_len+1)), y=all_u1_interface, kind='cubic')
        interpolator_phi1 = interp1d(x=np.array(range(time_window_len+1)), y=all_phi1_interface, kind='cubic')

        for i in range(1, time_window_len+1):
            u_nm1_interface = all_u1_interface[i-1]
            phi_nm1_interface = all_phi1_interface[i-1]
            u_nm1_2_interface = interpolator_u1(i-1/2)
            phi_nm1_2_interface = interpolator_phi1(i-1/2)
            u_interface = all_u1_interface[i]
            phi_interface = all_phi1_interface[i]

            u2_ret, u_interface, phi_interface, *phi_for_FV = \
                    discretization.integrate_one_step(
                M=M2,
                h=h2,
                D=D2,
                a=a,
                c=c,
                dt=dt,
                f=f2,
                f_nm1=f2,
                f_nm1_2=f2,
                bd_cond=neumann,
                bd_cond_nm1_2=neumann,
                bd_cond_nm1=neumann,
                Lambda=Lambda_2,
                u_nm1=all_u2[-1],
                u_interface=u_interface,
                phi_interface=phi_interface,
                u_nm1_2_interface=u_nm1_2_interface,
                phi_nm1_2_interface=phi_nm1_2_interface,
                u_nm1_interface=u_nm1_interface,
                phi_nm1_interface=phi_nm1_interface,
                phi_for_FV=phi_for_FV,
                upper_domain=True,
                Y=precomputed_Y2)
            all_u2 += [u2_ret]
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = [0]
        all_phi1_interface = [0]
        all_u1 = [u1_0]
        phi_for_FV = [phi1_0_fvol]

        interpolator_u2 = interp1d(x=np.array(range(time_window_len+1)), y=all_u2_interface, kind='cubic')
        interpolator_phi2 = interp1d(x=np.array(range(time_window_len+1)), y=all_phi2_interface, kind='cubic')
        for i in range(1, time_window_len+1):

            u_interface = all_u2_interface[i]
            phi_interface = all_phi2_interface[i]
            u_nm1_2_interface = interpolator_u2(i-1/2)
            phi_nm1_2_interface = interpolator_phi2(i-1/2)
            u_nm1_interface = all_u2_interface[i-1]
            phi_nm1_interface = all_phi2_interface[i-1]

            u1_ret, u_interface, phi_interface, *phi_for_FV = \
                    discretization.integrate_one_step(
                M=M1,
                h=h1,
                D=D1,
                a=a,
                c=c,
                dt=dt,
                f=f1,
                f_nm1_2=f1, #0 anywayy
                f_nm1=f1,#0 anywayy
                bd_cond=dirichlet,
                bd_cond_nm1_2=dirichlet,
                bd_cond_nm1=dirichlet,
                Lambda=Lambda_1,
                u_nm1=all_u1[-1],
                u_interface=u_interface,
                phi_interface=phi_interface,
                u_nm1_interface=u_nm1_interface,
                phi_nm1_interface=phi_nm1_interface,
                u_nm1_2_interface=u_nm1_2_interface,
                phi_nm1_2_interface=phi_nm1_2_interface,
                phi_for_FV=phi_for_FV,
                upper_domain=False,
                Y=precomputed_Y1)
            all_u1 += [u1_ret]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]

        ret += [all_u1_interface[1:]]

    return ret


if __name__ == "__main__":
    import main
    main.main()
