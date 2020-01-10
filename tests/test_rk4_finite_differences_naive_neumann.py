import numpy as np
from numpy import cos, sin
from numpy.random import random
from discretizations.rk4_finite_differences \
        import Rk4FiniteDifferences
fdifference = Rk4FiniteDifferences()
integrate_one_step = fdifference.integrate_one_step
integrate_one_step_star = fdifference.integrate_one_step_star


def launch_all_tests():
    print("Test integration finite differences with naive neumann:", time_test())
    return "ok"

def time_test():
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2

    a = 3.
    c = 0.3

    T = -10.
    d = 8.
    t_initial = 3.
    t_final = 4.
    N = 3000
    M1, M2 = 100, 30
    Lambda = -1.
    print("COURANT NUMBER:", (t_final-t_initial)/N*M2*M2)

    x1 = -np.linspace(0, 1, M1)**1
    x2 = np.linspace(0, 1, M2)**1

    h1 = np.diff(x1)
    h2 = np.diff(x2)

    h = np.concatenate((-h1[::-1], h2))

    # coordinates at half-points:
    x1_1_2 = x1[:-1] + h1 / 2
    x2_1_2 = x2[:-1] + h2 / 2
    x_1_2 = np.concatenate((np.flipud(x1_1_2), x2_1_2))

    x = np.concatenate((np.flipud(x1[:-1]), x2))

    D1 = 1.2 + x1_1_2**2
    D2 = 1.2 + x2_1_2**2

    D1_x = 1.2 + x1**2
    D2_x = 1.2 + x2**2
    D1_prime = 2 * x1
    D2_prime = 2 * x2

    ratio_D = D1_x[0] / D2_x[0]

    def u_interface_cond(t):
        return ratio_D * sin(d * x2[0]) + T * t
    def phi_interface_cond(t):
        return D2[0] * ratio_D* d * cos(d * x2_1_2[0])
    def dirichlet_cond(t):
        return sin(d * x1[-1]) + T * t
    def neumann_cond(t):
        return ratio_D * d * cos(d * x2_1_2[-1])

    def f2(t):
        return T * (1 + c * t) + ratio_D * (d * a * cos(d * x2) + c *
                                      sin(d * x2) + D2_x * d * d * sin(d * x2) - D2_prime * d * cos(d * x2))

    def u2(t):
        return ratio_D * sin(d * x2) + T * t

    """
    # Note: f is a local approximation !
    f2 = T * (1 + c * t) + ratio_D * (d * a * cos(d * x2) + c *
                                      sin(d * x2) + D2_x * d * d * sin(d * x2) - D2_prime * d * cos(d * x2)) - D2_x * ratio_D * d**4 * sin(d*x2)

    f1 = T * (1 + c * t) + d * a * cos(d * x1) + c * sin(d * x1) \
        + D1_x * d * d * sin(d * x1) - D1_prime * d * cos(d * x1)

    u0 = np.concatenate(
        (sin(d * x1[-1:0:-1]) + T * t_n, ratio_D * sin(d * x2) + T * t_n))
    u1 = np.concatenate(
        (sin(d * x1[-1:0:-1]) + T * t, ratio_D * sin(d * x2) + T * t))

    """

    D1 = np.concatenate(([D1_x[0]], D1[:-1]))
    D2 = np.concatenate(([D2_x[0]], D2[:-1]))

    ret = []
    u2_i = u2(t_initial)
    dt = (t_final - t_initial) / N
    for t_i in np.linspace(t_initial, t_final, N, endpoint=False):
        u2_i, comp_u_interface, comp_phi_interface = integrate_one_step(M=M2,
                                                                       h=h2,
                                                                       D=D2,
                                                                       a=a, c=c,
                                                                       dt=dt,
                                                                       f=f2(t_i+dt),
                                                                       f_nm1_2=f2(t_i+dt/2),
                                                                       f_nm1=f2(t_i),
                                                                       bd_cond=neumann_cond(t_i+dt),
                                                                       bd_cond_nm1_2=neumann_cond(t_i+dt/2),
                                                                       bd_cond_nm1=neumann_cond(t_i),
                                                                       Lambda=Lambda,
                                                                       u_nm1=u2_i,
                                                                       u_interface=u_interface_cond(t_i+dt),
                                                                       u_nm1_2_interface=u_interface_cond(t_i+dt/2),
                                                                       u_nm1_interface=u_interface_cond(t_i),
                                                                       phi_interface=phi_interface_cond(t_i+dt),
                                                                       phi_nm1_2_interface=phi_interface_cond(t_i+dt/2),
                                                                       phi_nm1_interface=phi_interface_cond(t_i),
                                                                       upper_domain=True)
        ret += [u2_i]

        
        nb_plots = 10
        if int(N * (t_i - t_initial) / (t_final - t_initial)) % int(N/nb_plots) == 0:
            import matplotlib.pyplot as plt
            plt.plot(x2, u2_i, "b")
            plt.plot(x2, u2(t_i+dt), "r")
            #plt.plot(x, u1, "k--")
            plt.show()


    assert np.linalg.norm(u1 - u_np1) < 1e-2

    return "ok"

