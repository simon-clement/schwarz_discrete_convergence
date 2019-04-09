#!/usr/bin/python3
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
    if w is None:
        s = 1./dt
    else:
        s = w*1j
    return discretization.analytic_robin_robin(s=s, Lambda_1=Lambda_1,
            Lambda_2=Lambda_2, a=a, c=c, dt=dt, M1=M1, M2=M2,
            D1=D1, D2=D2, verbose=verbose)


def rate(discretization, time_window_len, Lambda_1=None, Lambda_2=None,
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


def frequency_simulation(discretization, N, **kwargs):
    ret = None
    for seed in range(100, 103):
        errors = np.array(interface_errors(discretization, N, seed=seed, **kwargs))
        freq_err = np.fft.fft(errors, axis=-1)
        if ret is None:
            ret = freq_err
        else:
            ret += freq_err
    return ret


"""
    returns errors at interface from beginning (first guess) until the end.
    to get rate, just use the following code:
    def rate(*args, function_to_use=max):
        errors = interface_errors(*args)
        errors = [function_to_use([abs(e) for e in err]) for err in errors]
        return errors[2]/errors[1]
"""
def interface_errors(discretization, time_window_len, Lambda_1=None, Lambda_2=None,
        a=None, c=None, dt=None, M1=None, M2=None, seed=9380):

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
    ret = [all_u1_interface]
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

        ret += [all_u1_interface]

    return ret


if __name__ == "__main__":
    import main
    main.main()
