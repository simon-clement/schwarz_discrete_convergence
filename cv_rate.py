#!/usr/bin/python3

import numpy as np
from numpy import pi, cos, sin
import finite_difference
import finite_volumes

LAMBDA_1_DEFAULT=0.0
LAMBDA_2_DEFAULT=0.0

A_DEFAULT=1.0
C_DEFAULT=0.3
D1_DEFAULT=1.12
D2_DEFAULT=2.22 #TODO was 2.2


TIME_WINDOW_LEN_DEFAULT=1
DT_DEFAULT=0.05

M1_DEFAULT= 200
M2_DEFAULT= 200

def main():
    import sys
    if len(sys.argv) == 1:
        print("to launch tests, use \"python3 cv_rate.py test\"")
        print("Usage: python3 cv_rate {test, graph, optimize, debug}")
    else:
        if sys.argv[1] == "test":
            import tests.test_linear_sys
            import tests.test_schwarz
            import tests.test_finite_volumes
            tests.test_linear_sys.launch_all_tests()
            tests.test_schwarz.launch_all_tests()
            tests.test_finite_volumes.launch_all_tests()
        elif sys.argv[1] == "graph":
            beauty_graph()
        elif sys.argv[1] == "optimize":
            from scipy.optimize import minimize_scalar
            print("rate finite volumes:", minimize_scalar(rate_finite_volumes))
            print("rate finite differences:", minimize_scalar(rate_finite_differences))

            D1 = D1_DEFAULT
            D2 = D2_DEFAULT
            def theory_star(wmin, wmax):
                a = (np.sqrt(D1) - np.sqrt(D2)) * (np.sqrt(wmin)+np.sqrt(wmax))
                return 1/(2*np.sqrt(2))*(a + np.sqrt(a*a + 8*np.sqrt(D1*D2)*np.sqrt(wmin*wmax)))
            print("theory:", theory_star(pi/DT_DEFAULT, pi/(DT_DEFAULT*TIME_WINDOW_LEN_DEFAULT)))
        elif  sys.argv[1] == "debug":
            rate_finite_differences(15.)
        elif sys.argv[1] == "analytic":
            import matplotlib.pyplot as plt
            analytic_robin_robin_finite_differences(Lambda_1=0., verbose=True)
            lambda_1 = np.linspace(-10, 10, 10000)
            plt.plot(lambda_1, 
                   [analytic_robin_robin_finite_differences(Lambda_1=i) \
                           for i in lambda_1], "g")
            beauty_graph()


def rate_finite_volumes2(l): 
    return rate_finite_volumes(Lambda_1=l)
def rate_finite_differences2(l):
    return rate_finite_differences(Lambda_1=l)

def beauty_graph():
    import matplotlib.pyplot as plt
    x = np.linspace(-10, 10, 1000)
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        rate_fvolumes = executor.map(rate_finite_volumes2, x)
        rate_fdifferences = executor.map(rate_finite_differences2, x)

    plt.semilogy(x, list(rate_fvolumes), "r")
    plt.semilogy(x, list(rate_fdifferences), "b")
    plt.xlabel("$\\Lambda^1$")
    plt.ylabel("$\\rho$")
    plt.title("")
    plt.show()


def analytic_robin_robin_finite_differences(w=None, Lambda_1=LAMBDA_1_DEFAULT,
        Lambda_2=LAMBDA_2_DEFAULT, a=A_DEFAULT, 
        c=C_DEFAULT, dt=DT_DEFAULT, M1=M1_DEFAULT, M2=M2_DEFAULT,
        D1=D1_DEFAULT, D2=D2_DEFAULT, verbose=False):

    h1 = -1/(M1-1)
    h2 = 1/(M2-1)

    if w is None:
        s = 1./dt
    else:
        s = w*1j
    eta1_0 = D1/h1 + h1 / 2 * (s + c) - a/2
    eta2_0 = D2/h2 + h2 / 2 * (s + c) - a/2
    y2_0 = D2/h2 - a/2
    y1_0 = D1/h1 - a/2

    Y1_0 = - D1 / (h1*h1) - .5 * a / h1
    Y1_1 = 2*D1 / (h1*h1) + c
    Y1_2 = - D1 / (h1*h1) + .5 * a / h1

    Y2_0 = - D2 / (h2*h2) - .5 * a / h2
    Y2_1 = 2*D2 / (h2*h2) + c
    Y2_2 = - D2 / (h2*h2) + .5 * a / h2
    lambda2_plus = (- Y2_1 - s + np.sqrt((Y2_1+s)**2 - 4*Y2_0*Y2_2))/(2*Y2_2)
    lambda2_moins = (- Y2_1 - s - np.sqrt((Y2_1+s)**2 - 4*Y2_0*Y2_2))/(2*Y2_2)
    lambda1_plus = (- Y1_1 - s + np.sqrt((Y1_1+s)**2 - 4*Y1_0*Y1_2))/(2*Y1_2)
    lambda1_moins = (- Y1_1 - s - np.sqrt((Y1_1+s)**2 - 4*Y1_0*Y1_2))/(2*Y1_2)
    # Properties of lambda:
    assert abs(lambda1_moins*lambda1_plus - Y1_0/Y1_2) < 1e-12
    assert abs(lambda2_moins*lambda2_plus - Y2_0/Y2_2) < 1e-12
    #D constant continuous: assert abs(lambda1_moins - 1./lambda2_plus) < 1e-12
    #D constant continuous: assert abs(lambda2_moins - 1./lambda1_plus) < 1e-12

    teta1_0 = eta1_0 - y1_0 * lambda1_plus # warning : it is lambda_+ in the document
    teta2_0 = eta2_0 - y2_0 * lambda2_plus
    if verbose:
        print("eta^1_0:", teta1_0)
        print("eta^2_0:", teta2_0)
    rho_numerator = (Lambda_2 - teta2_0) * (Lambda_1 - teta1_0)
    rho_denominator = (Lambda_2 - teta1_0) * (Lambda_1 - teta2_0)

    return np.abs(rho_denominator / rho_numerator)

def rate_finite_volumes(Lambda_1=LAMBDA_1_DEFAULT, Lambda_2=LAMBDA_2_DEFAULT,
        a=A_DEFAULT, c=C_DEFAULT, time_window_len=TIME_WINDOW_LEN_DEFAULT,
                        dt=DT_DEFAULT, M1=M1_DEFAULT, M2=M2_DEFAULT):
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2
    integrate_one_step_star = finite_volumes.integrate_one_step_star
    integrate_one_step = finite_volumes.integrate_one_step

    T = 5.
    d = 8.
    t0 = 3.
    h1, h2 = 1/M1, 1/M2
    h1 = 1/M1 + np.zeros(M1)
    h2 = 1/M2 + np.zeros(M2)
    h1 = np.diff(np.cumsum(np.concatenate(([0],h1)))**1)
    h2 = np.diff(np.cumsum(np.concatenate(([0],h2)))**1)
    h = np.concatenate((h1[::-1], h2))

    # Center of the volumes are x, sizes are h
    x1 = np.cumsum(np.concatenate(([h1[0]/2],(h1[1:] + h1[:-1])/2)))
    x2 = np.cumsum(np.concatenate(([h2[0]/2],(h2[1:] + h2[:-1])/2)))
    x1 = np.flipud(x1)

    # coordinates at half-points:
    x1_1_2 = np.cumsum(np.concatenate(([0],h1)))
    x2_1_2 = np.cumsum(np.concatenate(([0],h2)))
    x_1_2 = np.concatenate((np.flipud(x1_1_2[:-1]), x2_1_2))

    x = np.concatenate((-x1, x2))
    two_if_not_constant = 0.

    D1 = D1_DEFAULT + x1_1_2 **two_if_not_constant
    D2 = D2_DEFAULT + x2_1_2 **two_if_not_constant

    ratio_D = D1[0] / D2[0]

    u0 = np.concatenate((np.diff(-cos(-d*x1_1_2[::-1])/d - T*t0*x1_1_2[::-1]),
        np.diff(-ratio_D*cos(d*x2_1_2)/d + T*t0*x2_1_2))) / h

    all_ui = [u0]
    all_ui_interface = []

    def get_f2(t):
        # Note: f is an average and not a local approximation !
        f2 = T * (x2_1_2[1:] - x2_1_2[:-1]) \
                + ratio_D*a*(sin(d*x2_1_2[1:]) - sin(d*x2_1_2[:-1])) \
                + c*(-ratio_D/d*(cos(d*x2_1_2[1:]) - cos(d*x2_1_2[:-1])) \
                                + T*t*(x2_1_2[1:] - x2_1_2[:-1])) \
                - d*ratio_D*(D2[1:]*cos(d*x2_1_2[1:]) - D2[:-1]*cos(d*x2_1_2[:-1]))
        f2 /= h2
        return f2

    def neumann(t):
        return ratio_D * d*cos(d*1)

    def dirichlet(t):
        return sin(-d) + T*t

    def get_f1(t):
        # {inf, sup} bounds of the interval ([x-h/2, x+h/2]):
        x1_sup = -x1_1_2[:-1]
        x1_inf = -x1_1_2[1:]

        f1 = T * (x1_sup - x1_inf) + a*(sin(d*x1_sup) - sin(d*x1_inf)) \
                + c*(-cos(d*x1_sup)/d + cos(d*x1_inf)/d + T*t*(x1_sup - x1_inf)) \
                - d*(D1[:-1]*cos(d*x1_sup) - D1[1:]*cos(d*x1_inf))

        f1 /= h1
        return f1

    t = t0
    for i in range(time_window_len):
        t += dt
        f1 = get_f1(t)
        f2 = get_f2(t)

        ui = np.concatenate((np.diff(-cos(-d*x1_1_2[::-1])/d - T*t*x1_1_2[::-1]),
            np.diff(-ratio_D*cos(d*x2_1_2)/d + T*t*x2_1_2))) / h

        u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
                M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                neumann=neumann(t), dirichlet=dirichlet(t), u_nm1=all_ui[-1])

        all_ui += [u_np1]
        all_ui_interface += [real_u_interface]


    # random fixed false initialization:
    u_interface=0.0
    phi_interface= d * D2[0]

    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1:]
    ecart = []
    all_u1_interface = [u_interface for _ in range(time_window_len)]
    all_phi1_interface = [phi_interface for _ in range(time_window_len)]
    # Beginning of schwarz iterations:
    for _ in range(3):
        all_u2_interface = []
        all_phi2_interface = []
        all_u2 =  [u2_0]
        # Time iteration:
        for i in range(time_window_len):
            t = t0 + (i+1)*dt

            u_interface = all_u1_interface[i]
            phi_interface = all_phi1_interface[i]

            u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=get_f2(t),
                    bd_cond=neumann(t), Lambda=Lambda_2, u_nm1=all_u2[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=True)
            all_u2 += [u2_ret]
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = []
        all_phi1_interface = []
        all_u1 = [u1_0]

        for i in range(time_window_len):
            t = t0 + (i+1)*dt

            u_interface = all_u2_interface[i]
            phi_interface = all_phi2_interface[i]

            u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                    h=h1, D=D1, a=a, c=c, dt=dt, f=get_f1(t),
                    bd_cond=dirichlet(t), Lambda=Lambda_1, u_nm1=all_u1[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=False)
            all_u1 += [u1_ret]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]

        ecart += [max([abs(u - real) for u, real in zip(all_u1_interface, all_ui_interface)])]

    return ecart[2] / ecart[1]


def rate_finite_differences(Lambda_1=LAMBDA_1_DEFAULT, Lambda_2=LAMBDA_2_DEFAULT,
        a=A_DEFAULT, c=C_DEFAULT, time_window_len=TIME_WINDOW_LEN_DEFAULT,
                        dt=DT_DEFAULT, M1=M1_DEFAULT, M2=M2_DEFAULT):
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2
    integrate_one_step_star = finite_difference.integrate_one_step_star
    integrate_one_step = finite_difference.integrate_one_step

    T = 5.
    d = 8.
    t0 = 3.

    x1 = -np.linspace(0,1,M1)**1
    x2 = np.linspace(0,1,M2)**1

    h1 = np.diff(x1)
    h2 = np.diff(x2)

    h = np.concatenate((-h1[::-1], h2))

    # coordinates at half-points:
    x1_1_2 = x1[:-1] + h1 / 2
    x2_1_2 = x2[:-1] + h2 / 2
    x_1_2 = np.concatenate((np.flipud(x1_1_2), x2_1_2))

    x = np.concatenate((np.flipud(x1[:-1]), x2))
    two_if_not_constant = 0.

    D1 = D1_DEFAULT + x1_1_2 ** two_if_not_constant
    D2 = D2_DEFAULT + x2_1_2 ** two_if_not_constant

    D1_x = D1_DEFAULT + x1** two_if_not_constant
    D2_x = D2_DEFAULT + x2** two_if_not_constant
    D1_prime = two_if_not_constant*x1
    D2_prime = two_if_not_constant*x2

    #TODO see if it is important to keep this ugly first term
    D1 = np.concatenate(([D1_x[0]], D1[:-1]))
    D2 = np.concatenate(([D2_x[0]], D2[:-1]))

    ratio_D = D1_x[0] / D2_x[0]

    def neumann(t):
        return ratio_D * d*cos(d*x2_1_2[-1])
    def dirichlet(t):
        return sin(d*x1[-1]) + T*t

    def f2(t):
        # Note: f is a local approximation !
        return T*(1+c*t) + ratio_D * (d*a*cos(d*x2) + c*sin(d*x2) \
                + D2_x * d*d *sin(d*x2) - D2_prime * d * cos(d*x2))

    def f1(t):
        return T*(1+c*t) + d*a*cos(d*x1) + c*sin(d*x1) \
            + D1_x * d*d *sin(d*x1) - D1_prime * d * cos(d*x1)


    u0 = np.concatenate((sin(d*x1[-1:0:-1]) + T*t0, ratio_D * sin(d*x2) + T*t0))
    all_ui = [u0]
    all_ui_interface = []
    for i in range(time_window_len):
        t = t0 + (i+1)*dt
        u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
                M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f1=f1(t), f2=f2(t),
                neumann=neumann(t), dirichlet=dirichlet(t), u_nm1=all_ui[-1])

        all_ui += [u_np1]
        all_ui_interface += [real_u_interface]


    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1-1:]

    # random fixed false initialization:
    u_interface=real_u_interface
    phi_interface= real_phi_interface

    ecart = []
    all_u1_interface = [u_interface for _ in range(time_window_len)]
    all_phi1_interface = [phi_interface for _ in range(time_window_len)]
    # Beginning of schwarz iterations:
    for i in range(3):
        all_u2_interface = []
        all_phi2_interface = []
        all_u2 =  [u2_0]
        # Time iteration:
        for i in range(time_window_len):
            t = t0 + (i+1)*dt

            u_interface = all_u1_interface[i]
            phi_interface = all_phi1_interface[i]

            u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2(t),
                    bd_cond=neumann(t), Lambda=Lambda_2, u_nm1=all_u2[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=True)
            all_u2 += [u2_ret]
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = []
        all_phi1_interface = []
        all_u1 =  [u1_0]
        # Time iteration:
        for i in range(time_window_len):
            t = t0 + (i+1)*dt

            u_interface = all_u2_interface[i]
            phi_interface = all_phi2_interface[i]
            u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1(t),
                    bd_cond=dirichlet(t), Lambda=Lambda_1, u_nm1=all_u1[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=False)
            all_u1 += [u1_ret]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]
        ecart += [max([abs(u - real) for u, real in zip(all_u1_interface, all_ui_interface)])]

    return (ecart[2] / ecart[1])


if __name__ == "__main__":
    main()
