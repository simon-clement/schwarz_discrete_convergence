import numpy as np
from numpy import cos, sin
from numpy.random import random
from discretizations.finite_difference import FiniteDifferences
fdifference = FiniteDifferences()
integrate_one_step = fdifference.integrate_one_step
integrate_one_step_star = fdifference.integrate_one_step_star


def launch_all_tests():
    print("Test integration finite differences:", time_test_star())
    print("Test complete finite differences:", complete_test_schwarz())
    return "ok"


def complete_test_schwarz():
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2

    a = 1.
    c = 0.3

    T = 5.
    d = 8.
    t = 3.
    dt = 0.05
    M1, M2 = 10, 10

    x1 = -np.linspace(0, 1, M1)**3
    x2 = np.linspace(0, 1, M2)**4

    h1 = np.diff(x1)
    h2 = np.diff(x2)

    h = np.concatenate((-h1[::-1], h2))

    # coordinates at half-points:
    x1_1_2 = x1[:-1] + h1 / 2
    x2_1_2 = x2[:-1] + h2 / 2
    x_1_2 = np.concatenate((np.flipud(x1_1_2), x2_1_2))

    x = np.concatenate((np.flipud(x1[:-1]), x2))

    D1 = 1.2 + x1_1_2**2
    D2 = 2.2 + x2_1_2**2

    D1_x = 1.2 + x1**2
    D2_x = 2.2 + x2**2
    D1_prime = 2 * x1
    D2_prime = 2 * x2

    ratio_D = D1_x[0] / D2_x[0]

    t_n, t = t, t + dt
    neumann = ratio_D * d * cos(d * x2_1_2[-1])
    dirichlet = sin(d * x1[-1]) + T * t

    # Note: f is a local approximation !
    f2 = T*(1+c*t) + ratio_D * (d*a*cos(d*x2) + c*sin(d*x2) \
            + D2_x * d*d *sin(d*x2) - D2_prime * d * cos(d*x2))

    f1 = T*(1+c*t) + d*a*cos(d*x1) + c*sin(d*x1) \
            + D1_x * d*d *sin(d*x1) - D1_prime * d * cos(d*x1)

    u0 = np.concatenate(
        (sin(d * x1[-1:0:-1]) + T * t_n, ratio_D * sin(d * x2) + T * t_n))
    u1 = np.concatenate(
        (sin(d * x1[-1:0:-1]) + T * t, ratio_D * sin(d * x2) + T * t))

    # TODO study the impact of theses lines to remove them
    D1 = np.concatenate(([D1_x[0]], D1[:-1]))
    D2 = np.concatenate(([D2_x[0]], D2[:-1]))

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    # Schwarz parameters:
    Lambda_1 = 15.
    Lambda_2 = 0.3

    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1 - 1:]
    u1_1 = np.flipud(u_np1[:M1])
    u2_1 = u_np1[M1 - 1:]

    # random fixed false initialization:
    u_interface = real_u_interface
    phi_interface = real_phi_interface

    ecart = []
    # Beginning of iterations:
    for i in range(30):
        u2_ret, u_interface, phi_interface = integrate_one_step(
            M=M2,
            h=h2,
            D=D2,
            a=a,
            c=c,
            dt=dt,
            f=f2,
            bd_cond=neumann,
            Lambda=Lambda_2,
            u_nm1=u2_0,
            u_interface=u_interface,
            phi_interface=phi_interface,
            upper_domain=True)

        u1_ret, u_interface, phi_interface = integrate_one_step(
            M=M1,
            h=h1,
            D=D1,
            a=a,
            c=c,
            dt=dt,
            f=f1,
            bd_cond=dirichlet,
            Lambda=Lambda_1,
            u_nm1=u1_0,
            u_interface=u_interface,
            phi_interface=phi_interface,
            upper_domain=False)
        u_np1_schwarz = np.concatenate((u1_ret[-1:0:-1], u2_ret))
        ecart += [np.linalg.norm(u_np1 - u_np1_schwarz)]

    assert ecart[-1] < 1 * 1e-10

    return "ok"


def time_test_star():
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2

    a = 1.2
    c = 0.3

    T = 5.
    d = 8.
    t = 3.
    dt = 0.05
    M1, M2 = 3000, 3000

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
    D2 = 2.2 + x2_1_2**2

    D1_x = 1.2 + x1**2
    D2_x = 2.2 + x2**2
    D1_prime = 2 * x1
    D2_prime = 2 * x2

    ratio_D = D1_x[0] / D2_x[0]

    t_n, t = t, t + dt
    neumann = ratio_D * d * cos(d * x2_1_2[-1])
    dirichlet = sin(d * x1[-1]) + T * t

    # Note: f is a local approximation !
    f2 = T*(1+c*t) + ratio_D * (d*a*cos(d*x2) + c*sin(d*x2) \
            + D2_x * d*d *sin(d*x2) - D2_prime * d * cos(d*x2))

    f1 = T*(1+c*t) + d*a*cos(d*x1) + c*sin(d*x1) \
            + D1_x * d*d *sin(d*x1) - D1_prime * d * cos(d*x1)

    u0 = np.concatenate(
        (sin(d * x1[-1:0:-1]) + T * t_n, ratio_D * sin(d * x2) + T * t_n))
    u1 = np.concatenate(
        (sin(d * x1[-1:0:-1]) + T * t, ratio_D * sin(d * x2) + T * t))

    D1 = np.concatenate(([D1_x[0]], D1[:-1]))
    D2 = np.concatenate(([D2_x[0]], D2[:-1]))

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u0)
    """
    import matplotlib.pyplot as plt
    plt.plot(x, u0, "b")
    plt.plot(x, u_np1, "r")
    plt.plot(x, u1, "k--")
    plt.show()
    """
    assert np.linalg.norm(u1 - u_np1) < 1e-2

    return "ok"
