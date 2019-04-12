#!/usr/bin/python3
import time
import numpy as np
from numpy import pi, cos, sin
from discretizations.finite_difference import FiniteDifferences
from discretizations.finite_volumes import FiniteVolumes
import functools

def continuous_analytic_rate_robin_neumann(discretization, Lambda_1, w):
    D1 = discretization.D1_DEFAULT
    D2 = discretization.D2_DEFAULT
    # sig1 is \sigma^1_{+}
    sig1 = np.sqrt(np.abs(w)/(2*D1)) * (1 + np.abs(w)/w * 1j)
    # sig2 is \sigma^2_{-}
    sig2 = -np.sqrt(np.abs(w)/(2*D2)) * (1 + np.abs(w)/w * 1j)
    return np.abs(D1*sig1*(D2*sig2+Lambda_1) / (D2*sig2*(D1*sig1+Lambda_1)))

def continuous_analytic_rate_robin_robin(discretization, Lambda_1, Lambda_2, w):
    D1 = discretization.D1_DEFAULT
    D2 = discretization.D2_DEFAULT
    # sig1 is \sigma^1_{+}
    sig1 = np.sqrt(np.abs(w)/(2*D1)) * (1 + np.abs(w)/w * 1j)
    # sig2 is \sigma^2_{-}
    sig2 = -np.sqrt(np.abs(w)/(2*D2)) * (1 + np.abs(w)/w * 1j)
    first_term = np.abs((D2*sig2+Lambda_1) / (D1*sig1+Lambda_1))
    #TODO why is there here a "+" whereas in the paper it's 'D2*sig2-Lambda_2'
    second = np.abs((D1*sig1-Lambda_2) / (D2*sig2-Lambda_2))
    # TODO put back a "+" ?
    return first_term*second

def continuous_best_lam_robin_neumann(discretization, N):
    sqD1 = np.sqrt(discretization.D1_DEFAULT)
    sqD2 = np.sqrt(discretization.D2_DEFAULT)
    dt = discretization.DT_DEFAULT
    T = dt*N
    sqw1 = np.sqrt(pi/T)
    sqw2 = np.sqrt(pi/dt)
    return 1/(2*np.sqrt(2)) * ((sqD2-sqD1)*(sqw1+sqw2) + np.sqrt((sqD2-sqD1)**2 * (sqw1 + sqw2)**2 + 8*sqD1*sqD2*sqw1*sqw2))


def rate_by_z_transform(discretization, Lambda_1, NUMBER_SAMPLES):
    all_points = np.linspace(0, 2*pi, NUMBER_SAMPLES, endpoint=False)
    dt=DT_DEFAULT
    z_transformed = lambda z:discretization.analytic_robin_robin(s=1./dt*(z-1)/z,
            Lambda_1=Lambda_1)
    r = 1.001
    samples = [z_transformed(r*np.exp(p*1j)) for p in all_points]
    ret_ifft = np.fft.ifft(np.array(samples))
    rate_each_time = [r**n * l for n, l in enumerate(ret_ifft)]
    return np.max(np.abs(rate_each_time))


def analytic_robin_robin(discretization, w=None, Lambda_1=None,
        Lambda_2=None, a=None, 
        c=None, dt=None, M1=None, M2=None,
        D1=None, D2=None, verbose=False):
    dt = discretization.DT_DEFAULT
    if w is None:
        s = 1./dt
    else:
        z = 1*np.exp(w*1j)
        s = 1./dt * (z-1) / z
        s = w*1j
    return discretization.analytic_robin_robin(s=s, Lambda_1=Lambda_1,
            Lambda_2=Lambda_2, a=a, c=c, dt=dt, M1=M1, M2=M2,
            D1=D1, D2=D2, verbose=verbose)


"""
    Makes a simulation and gives the convergence rate.
    uses the rust module to be faster than python
    For details of args and kwargs, see @interface_errors
    function_to_use can be max for L^\infty or np.linalg.norm for L^2
    This particular function use a lot of different simulations with random
    first guess to get a good convergence rate.
"""
PARALLEL = True
def rate_fast(discretization, N, Lambda_1=None, Lambda_2=None,
        a=None, c=None, dt=None, M1=None, M2=None,
        function_to_use=lambda x:max(np.abs(x)),
        seeds=range(10)):
    try:
        import rust_mod
        errors = rust_mod.errors(discretization, N, Lambda_1, Lambda_2,
                a, c, dt, M1, M2,
                number_seeds=len(list(seeds)),
                function_D1=None, function_D2=None)
    except:
        print("Cannot use rate_fast. Did you compile rust module ? Using pure python...")
        errors = None
        to_map = functools.partial(rate_one_seed, discretization, N, function_to_use=function_to_use,
                Lambda_1=Lambda_1, Lambda_2=Lambda_2, a=a, c=c, dt=dt, M1=M1, M2=M2) 
        if PARALLEL:
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor() as executor:
                errors = np.mean(np.array(list(executor.map(to_map, seeds))), axis=0)
        else:
            errors = np.mean(np.abs(np.array(list(map(to_map, seeds)))), axis=0)

    return function_to_use(errors[2])/function_to_use(errors[1])

"""
    Makes a simulation and gives the convergence rate.
    For details of args and kwargs, see @interface_errors
    function_to_use can be max for L^\infty or np.linalg.norm for L^2
    This particular function use a lot of different simulations with random
    first guess to get a good convergence rate.
"""
def rate(discretization, N, Lambda_1=None, Lambda_2=None,
        a=None, c=None, dt=None, M1=None, M2=None,
        function_to_use=lambda x:max(np.abs(x)),
        seeds=range(10)):
    errors = None
    to_map = functools.partial(rate_one_seed, discretization, N, function_to_use=function_to_use,
            Lambda_1=Lambda_1, Lambda_2=Lambda_2, a=a, c=c, dt=dt, M1=M1, M2=M2) 
    if PARALLEL:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            errors = np.mean(np.array(list(executor.map(to_map, seeds))), axis=0)
    else:
        errors = np.mean(np.abs(np.array(list(map(to_map, seeds)))), axis=0)

    import matplotlib.pyplot as plt
    plt.plot(errors[0], label="py")
    plt.plot(errors_rust[0], label="rust")
    plt.plot(errors[1], "--", label="py")
    plt.plot(errors_rust[1], "--", label="rust")
    plt.legend()
    plt.show()

    return function_to_use(errors[2])/function_to_use(errors[1])

def rate_one_seed(discretization, N, seed, function_to_use=max, **kwargs):
    errors = interface_errors(discretization, N, seed=seed, **kwargs)
    return errors
    return np.array([function_to_use([abs(e) for e in err]) for err in errors])

def rate_old(discretization, time_window_len, Lambda_1=None, Lambda_2=None,
        a=None, c=None, dt=None, M1=None, M2=None, function_to_use=max):

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
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                    bd_cond=dirichlet, Lambda=Lambda_1, upper_domain=False)

    precomputed_Y2 = discretization.precompute_Y(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                    bd_cond=neumann, Lambda=Lambda_2, upper_domain=True)

    # random false initialization:
    u1_0 = np.zeros(M1)
    u2_0 = np.zeros(M2)
    error = []
    np.random.seed(9380)
    all_u1_interface = 2*(np.random.rand(time_window_len) - 0.5)
    all_phi1_interface = 2*(np.random.rand(time_window_len) - 0.5)
    all_u1_interface[-1] /= 1000.
    all_phi1_interface[-1] /= 1000.
    # Beginning of schwarz iterations:
    for k in range(2):
        all_u2_interface = []
        all_phi2_interface = []
        all_u2 =  [u2_0]
        # Time iteration:
        for i in range(time_window_len):
            u_interface = all_u1_interface[i]
            phi_interface = all_phi1_interface[i]

            u2_ret, u_interface, phi_interface = discretization.integrate_one_step(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                    bd_cond=neumann, Lambda=Lambda_2, u_nm1=all_u2[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=True, Y=precomputed_Y2)
            all_u2 += [u2_ret]
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = []
        all_phi1_interface = []
        all_u1 = [u1_0]

        for i in range(time_window_len):

            u_interface = all_u2_interface[i]
            phi_interface = all_phi2_interface[i]

            u1_ret, u_interface, phi_interface = discretization.integrate_one_step(M=M1,
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                    bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=all_u1[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=False, Y=precomputed_Y1)
            all_u1 += [u1_ret]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]

        error += [function_to_use([abs(e) for e in all_phi1_interface])]

    return error[1] / error[0]

def raw_simulation(discretization, N, number_samples=1000, **kwargs):
    try:
        import rust_mod
        time_start = time.time()
        errors = rust_mod.errors_raw(discretization, N, 
                number_seeds=number_samples, **kwargs)
        print("took", time.time() - time_start, "seconds")
        return np.mean(np.abs(errors), axis=0)
    except:
        print("cannot make a fast raw simulation... Going to pure python " + \
                "(but it will take some time)")
        import concurrent.futures
        from numpy.fft import fft, fftshift
        to_map = functools.partial(interface_errors, discretization, N, **kwargs)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            errors = np.array(list(executor.map(to_map, range(number_samples))))
        return np.mean(np.abs(errors), axis=0)


def frequency_simulation(discretization, N, number_samples=1000, **kwargs):
    try:
        import rust_mod
        errors = rust_mod.errors_raw(discretization, N, 
                number_seeds=number_samples, **kwargs)
        freq_err = np.fft.fftshift(np.fft.fft(errors, axis=-1), axes=(-1,))
        return np.mean(np.abs(freq_err), axis=0)
    except:
        raise
        print("cannot make a fast frequency simulation... Going to pure python " + \
                "(but it will take some time)")
        import concurrent.futures
        from numpy.fft import fft, fftshift
        to_map = functools.partial(interface_errors, discretization, N, **kwargs)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            errors = np.array(list(executor.map(to_map, range(number_samples))))
        freq_err = np.fft.fftshift(np.fft.fft(errors, axis=-1), axes=(-1,))
        return np.mean(np.abs(freq_err), axis=0)


"""
    returns errors at interface from beginning (first guess) until the end.
    to get rate, just use the following code:
    def rate(*args, function_to_use=max):
        errors = interface_errors(*args)
        errors = [function_to_use([abs(e) for e in err]) for err in errors]
        return errors[2]/errors[1]
"""
def interface_errors(discretization, time_window_len, seed=9380, Lambda_1=None, Lambda_2=None,
        a=None, c=None, dt=None, M1=None, M2=None):

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
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                    bd_cond=dirichlet, Lambda=Lambda_1, upper_domain=False)

    precomputed_Y2 = discretization.precompute_Y(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                    bd_cond=neumann, Lambda=Lambda_2, upper_domain=True)

    # random false initialization:
    u1_0 = np.zeros(M1)
    u2_0 = np.zeros(M2)
    error = []
    np.random.seed(seed)
    all_u1_interface = 2*(np.random.rand(time_window_len) - 0.5)
    all_phi1_interface = 2*(np.random.rand(time_window_len) - 0.5)
    all_u1_interface[-1] /= 1000;
    all_phi1_interface[-1] /= 1000;
    ret = [all_u1_interface]
    # Beginning of schwarz iterations:
    for k in range(3):
        all_u2_interface = []
        all_phi2_interface = []
        all_u2 =  [u2_0]
        # Time iteration:
        for i in range(time_window_len):
            u_interface = all_u1_interface[i]
            phi_interface = all_phi1_interface[i]

            u2_ret, u_interface, phi_interface = discretization.integrate_one_step(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                    bd_cond=neumann, Lambda=Lambda_2, u_nm1=all_u2[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=True, Y=precomputed_Y2)
            all_u2 += [u2_ret]
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        ret += [all_u2_interface]
        all_u1_interface = []
        all_phi1_interface = []
        all_u1 = [u1_0]

        for i in range(time_window_len):

            u_interface = all_u2_interface[i]
            phi_interface = all_phi2_interface[i]

            u1_ret, u_interface, phi_interface = discretization.integrate_one_step(M=M1,
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                    bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=all_u1[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=False, Y=precomputed_Y1)
            all_u1 += [u1_ret]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]

        ret += [all_u1_interface]

    return ret


if __name__ == "__main__":
    import main
    main.main()
