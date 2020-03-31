import numpy as np
from numpy import cos, sin
from numpy.random import random
from discretizations.finite_difference_naive_neumann \
        import FiniteDifferencesNaiveNeumann
from figures import DEFAULT
fdifference = DEFAULT.new(FiniteDifferencesNaiveNeumann)
integrate_one_step = fdifference.integrate_one_step
integrate_one_step_star = fdifference.integrate_one_step_star


def launch_all_tests():
    print("Test complete finite differences with naive neumann:",  complete_test_schwarz())
    print("Test integration finite differences with naive neumann:", time_test_domain2())
    print("Test integration finite differences with naive neumann:", time_test())
    return "ok"

def complete_test_schwarz():
    from tests.test_schwarz import schwarz_convergence
    ecart = schwarz_convergence(fdifference)
    assert ecart[-1] < 2e-4
    return "ok"

def time_test():
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2

    a = 0.0
    c = 0.

    T = 5.
    d = 8.
    t = 3.
    dt = 1.
    M1 = 1500

    x1 = -np.linspace(0, 1, M1)**1

    h1 = np.diff(x1)

    # coordinates at half-points:
    x1_1_2 = x1[:-1] + h1 / 2
    D1 = 1. + np.zeros_like(x1_1_2)
    D1_x = 1. + np.zeros_like(x1)

    t_n, t = t, t+dt
    dirichlet = sin(d * x1[-1]) + T * t

    u_interface = sin(d * x1[0]) + T * t
    phi_interface = D1_x[0]*d*cos(d * x1[0])

    f1 = T + D1_x * d * d * sin(d * x1)

    u0 = sin(d * x1) + T * t_n
    u1 = sin(d * x1) + T * t

    fdifference.D1 = D1
    fdifference.M1 = M1
    fdifference.SIZE_DOMAIN_1 = 1.
    fdifference.LAMBDA_1 = 0.
    fdifference.A = a
    fdifference.c = c
    fdifference.DT = dt

    u_np1, real_u_interface, real_phi_interface = fdifference.integrate_one_step(f=f1, bd_cond=dirichlet, phi_interface=phi_interface, u_interface=u_interface, u_nm1=u0, upper_domain=False)

    print(np.linalg.norm(u1 - u_np1))
    assert np.linalg.norm(u1 - u_np1) < 1e-3

    return "ok"


def time_test_domain2():
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2

    a = 0.0
    c = 0.

    T = 100.
    d = 8.
    t = 3.
    dt = 1.
    M2 = 1500

    x2 = np.linspace(0., 1., M2)

    h2 = np.diff(x2)

    # coordinates at half-points:
    x2_1_2 = x2[:-1] + h2 / 2

    D2 = 2.2 + np.zeros_like(x2_1_2)

    D2_x = 2.2 + np.zeros_like(x2)
    D2_prime = np.zeros_like(2 * x2)

    ratio_D = 1. / D2_x[0]

    t_n, t = t, t+dt
    neumann = ratio_D * d * cos(d * x2_1_2[-1])

    u_interface = sin(d * x2[0]) + T * t
    phi_interface = 1.*d*cos(d * x2[0])

    f2 = T * (1 + c * t) + ratio_D * (d * a * cos(d * x2) + c *
                                      sin(d * x2) + D2_x * d * d * sin(d * x2) - D2_prime * d * cos(d * x2))

    u0 = ratio_D * sin(d * x2) + T * t_n
    u1 = ratio_D * sin(d * x2) + T * t

    D2 = np.concatenate(([D2_x[0]], D2[:-1]))
    fdifference.D2 = D2
    fdifference.M2 = M2
    fdifference.LAMBDA_2 = .1
    fdifference.SIZE_DOMAIN_2 = 1.
    fdifference.A = a
    fdifference.c = c
    fdifference.DT = dt

    u_np1, real_u_interface, real_phi_interface = fdifference.integrate_one_step(f=f2, bd_cond=neumann, phi_interface=phi_interface, u_interface=u_interface, u_nm1=u0, upper_domain=True)

    print(np.linalg.norm(u1 - u_np1))

    assert np.linalg.norm(u1 - u_np1) < 1e-3

    return "ok"

if __name__ == "__main__":
    launch_all_tests()
