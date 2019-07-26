#!/usr/bin/python3
"""
    This module computes theoric convergence rates and
    provide functions to observe real convergence rate.
"""
import time
import numpy as np
from numpy import pi
import functools

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

    # sig1 is \sigma^1_{+} : should it be \sigma^1_{-} ?
    sig1 = np.sqrt((w*1j + c) / D1)
    sig2 = np.sqrt((w*1j + c) / D2)

    #TODO comprendre pourquoi cette ligne est nécessaire
    #Lambda_2 = - Lambda_2 OR sig1 = -sig1
    sig1 = -sig1

    first_term = (D2*sig2-Lambda_1*np.tanh(H2*sig2))/ \
            (D1*sig1 - Lambda_1*np.tanh(H1*sig1))
    second = (D1*sig1 - Lambda_2*np.tanh(H2*sig2))/ \
            (D2*sig2 - Lambda_2*np.tanh(H1*sig1))
    return np.abs(first_term * second)


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

        Main theoric function of the module. It is just a call
        to the good discretization.
    """
    if dt is None:
        dt = discretization.DT_DEFAULT
    if w is None:
        s = 1. / dt
    else:
        if semi_discrete:
            s = w * 1j
        else:
            # Note : in full discrete case, a fftshift MAY BE needed
            # I don't really know when it's accurate,
            # it seems that with a small N it is necessary to perform
            # a fftshift. That's a little bit some dark magic...
            z = 1.0 * np.exp(-w * 1j * dt * N)
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

def one_schwarz_iteration_matrix(dis, N, Lambda_1, Lambda_2):
    """
        Gives the matrices:
        - Z such that:
            Z(Lambda_1*e^k + phi^k) = Lambda_1*e^{k+1} + phi^{k+1}
        - Z_fin such that:
            Z_fin(Lambda_1*e^k + phi^k) = e^{k+1}
        returns Z, Z_fin
    """
    a, c, dt = dis.get_a_c_dt(None, None, None)
    M1 = dis.M1_DEFAULT
    M2 = dis.M2_DEFAULT
    h1 = -dis.SIZE_DOMAIN_1 / (M1-1)
    h2 = dis.SIZE_DOMAIN_2 / (M2-1)
    D1 = dis.D1_DEFAULT
    D2 = dis.D2_DEFAULT

    R1 = np.diag(np.concatenate(((0,), np.ones(M1-2), (0,))))
    R2 = np.diag(np.concatenate(((0,), np.ones(M2-2), (0,))))

    I_0_1 = np.array([1] + [0]*(M1-1))
    I_0_2 = np.array([1] + [0]*(M2-1))
    I_1_1 = dis.give_robin_projector(M1, h1, D1, a, c, dt, 0, Lambda_2)
    I_1_2 = dis.give_robin_projector(M2, h2, D2, a, c, dt, 0, Lambda_1)

    Y1 = dis.give_Y_for_analysis(M=M1, h=h1, D=D1, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_1, upper_domain=False)
    Y2 = dis.give_Y_for_analysis(M=M2, h=h2, D=D2, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_2, upper_domain=True)

    Y_inv_2 = np.linalg.inv(Y2)
    Y_inv_1 = np.linalg.inv(Y1)
    Z1_fin = []
    Z2_fin = []
    Z1 = []
    Z2 = []
    for n in range(1, N+1):
        bloc1 = []
        bloc1_fin = []
        bloc2 = []
        bloc2_fin = []
        for i in range(N):
            if i < n:
                bloc1_fin += [I_0_1.T @ np.linalg.matrix_power(Y_inv_1 @ R1, n - i-1) @Y_inv_1 @ I_0_1]
                bloc1 += [I_1_1.T @ np.linalg.matrix_power(Y_inv_1 @ R1, n - i-1) @Y_inv_1 @ I_0_1]
                bloc2_fin += [np.reshape(np.linalg.matrix_power(Y_inv_2 @ R2, n - i-1) @Y_inv_2 @ I_0_2, (-1, 1))]
                bloc2 += [I_1_2.T @ np.linalg.matrix_power(Y_inv_2 @ R2, n - i-1) @Y_inv_2 @ I_0_2]
            else:
                bloc1_fin += [np.zeros(1)]
                bloc2_fin += [np.zeros_like(np.reshape(I_0_2, (-1,1)))]
                bloc1 += [np.zeros(1)]
                bloc2 += [np.zeros(1)]
        Z1_fin += [np.hstack(bloc1_fin)]
        Z2_fin += [np.hstack(bloc2_fin)]
        Z1 += [np.hstack(bloc1)]
        Z2 += [np.hstack(bloc2)]
    Z1_fin = np.vstack(Z1_fin)
    Z2_fin = np.vstack(Z2_fin)
    Z1 = np.vstack(Z1)
    Z2 = np.vstack(Z2)
    
    Z = Z1@Z2
    Z_fin = Z1_fin@Z2
    return Z, Z_fin

def norm_matrix_for_performances(dis, N, Lambda_1, Lambda_2, norm='fro'):
    """
        possible norms:
        - 'fro': Frobenius norm
        - 'nuc': Nuclear norm
        - 2: Largest singular value
        - 1: max(sum(abs(x), axis=0))
    """
    a, c, dt = dis.get_a_c_dt(None, None, None)
    M1 = dis.M1_DEFAULT
    M2 = dis.M2_DEFAULT
    h1 = -dis.SIZE_DOMAIN_1 / (M1-1)
    h2 = dis.SIZE_DOMAIN_2 / (M2-1)
    D1 = dis.D1_DEFAULT
    D2 = dis.D2_DEFAULT

    R1 = np.diag(np.concatenate(((0,), np.ones(M1-2), (0,))))
    R2 = np.diag(np.concatenate(((0,), np.ones(M2-2), (0,))))

    I_0_1 = np.array([1] + [0]*(M1-1))
    I_0_2 = np.array([1] + [0]*(M2-1))
    I_1_1 = dis.give_robin_projector(M1, h1, D1, a, c, dt, 0, Lambda_2)
    I_1_2 = dis.give_robin_projector(M2, h2, D2, a, c, dt, 0, Lambda_1)

    Y1 = dis.give_Y_for_analysis(M=M1, h=h1, D=D1, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_1, upper_domain=False)
    Y2 = dis.give_Y_for_analysis(M=M2, h=h2, D=D2, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_2, upper_domain=True)

    Y_inv_2 = np.linalg.inv(Y2)
    Y_inv_1 = np.linalg.inv(Y1)
    Z1 = []
    Z2 = []
    for n in range(1, N+1):
        bloc1 = []
        bloc2 = []
        for i in range(N):
            if i < n:
                bloc1 += [I_1_1.T @ np.linalg.matrix_power(Y_inv_1 @ R1, n - i-1) @Y_inv_1 @ I_0_1]
                bloc2 += [I_1_2.T @ np.linalg.matrix_power(Y_inv_2 @ R2, n - i-1) @Y_inv_2 @ I_0_2]
            else:
                bloc1 += [np.zeros(1)]
                bloc2 += [np.zeros(1)]
        Z1 += [np.hstack(bloc1)]
        Z2 += [np.hstack(bloc2)]
    Z1 = np.vstack(Z1)
    Z2 = np.vstack(Z2)
    Z = Z1@Z2

    return np.linalg.norm(Z, norm)


def fast_simulation_by_matrix(dis, N, Lambda_1, Lambda_2, number_samples=1000, NUMBER_IT=3):

    Z, Z_fin = one_schwarz_iteration_matrix(dis, N, Lambda_1, Lambda_2)
    e_simu = []
    for k in range(number_samples):
        np.random.seed(k)
        all_u1_interface = 2 * (np.random.rand(N) - 0.5)
        all_phi1_interface = 2 * (np.random.rand(N) - 0.5)
        first_guess = (Lambda_2 * all_u1_interface + all_phi1_interface)
        e_simu += [[first_guess]]
        for i in range(NUMBER_IT):
            #e_simu[-1] += [np.fft.fftshift(np.fft.fft(Z_fin @ np.linalg.matrix_power(Z, i) @ first_guess, axis=-1), axes=(-1, ))]
            e_simu[-1] += [np.fft.fftshift(np.fft.fft(np.linalg.matrix_power(Z, i) @ first_guess, axis=-1), axes=(-1, ))]
    
    return np.std(np.array(e_simu), axis=0)


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
        return np.mean(np.abs(freq_err), axis=0)
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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        errors = np.array(list(executor.map(to_map,
                                            range(number_samples))))
    freq_err = fftshift(fft(errors, axis=-1), axes=(-1, ))
    if number_samples != 1:
        return np.std(freq_err, axis=0)
    else:
        return np.abs(freq_err)[0]


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
                     NUMBER_IT=20):
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

    # random false initialization:
    u1_0 = np.zeros(M1)
    u2_0 = np.zeros(M2)
    np.random.seed(seed)
    all_u1_interface = 2 * (np.random.rand(time_window_len) - 0.5)
    all_phi1_interface = 2 * (np.random.rand(time_window_len) - 0.5)
    #all_u1_interface[-1] /= 1000
    #all_phi1_interface[-1] /= 1000
    ret = [all_u1_interface]
    # Beginning of schwarz iterations:
    for k in range(NUMBER_IT):
        all_u2_interface = []
        all_phi2_interface = []
        all_u2 = [u2_0]
        # Time iteration:
        for i in range(time_window_len):
            u_interface = all_u1_interface[i]
            phi_interface = all_phi1_interface[i]

            u2_ret, u_interface, phi_interface = \
                    discretization.integrate_one_step(
                M=M2,
                h=h2,
                D=D2,
                a=a,
                c=c,
                dt=dt,
                f=f2,
                bd_cond=neumann,
                Lambda=Lambda_2,
                u_nm1=all_u2[-1],
                u_interface=u_interface,
                phi_interface=phi_interface,
                upper_domain=True,
                Y=precomputed_Y2)
            all_u2 += [u2_ret]
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = []
        all_phi1_interface = []
        all_u1 = [u1_0]

        for i in range(time_window_len):

            u_interface = all_u2_interface[i]
            phi_interface = all_phi2_interface[i]

            u1_ret, u_interface, phi_interface = \
                    discretization.integrate_one_step(
                M=M1,
                h=h1,
                D=D1,
                a=a,
                c=c,
                dt=dt,
                f=f1,
                bd_cond=dirichlet,
                Lambda=Lambda_1,
                u_nm1=all_u1[-1],
                u_interface=u_interface,
                phi_interface=phi_interface,
                upper_domain=False,
                Y=precomputed_Y1)
            all_u1 += [u1_ret]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]
        """
        if k==0:
            print("first simu:")
            print(all_u1[1])
            print("second simu:")
            print(all_u1[2])
        """

        ret += [all_u1_interface]

    return ret


if __name__ == "__main__":
    import main
    main.main()
