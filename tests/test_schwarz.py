import numpy as np
from numpy import cos, sin
from numpy.random import random
from tests.solve_full_domain import solve_u_time_domain
from tests.utils_numeric import integration
from utils_linalg import solve_linear
from discretizations.finite_difference import FiniteDifferences
fdifference = FiniteDifferences()
get_Y = fdifference.get_Y
get_Y_star = fdifference.get_Y_star


def schwarz_convergence_global(discretisation):
    integrate_one_step_star = discretisation.integrate_one_step_star
    integrate_one_step = discretisation.integrate_one_step
    # Our domain is [-H,H]
    # we define u as u(x, t) = sin(dx+1) + Tt in \Omega_1,
    # u(x, t) = sin(dx+1) + Tt + rx      in \Omega_2
    # where r = dcos(1) * (D1/D2 - 1)

    H = 200
    a = 0.0
    c = 0.3

    T = 5.
    d = .5
    t = 3.
    dt = 0.2
    M1, M2 = 2000, 2000
    LENGTH = .5

    x1 = -H*np.linspace(0, 1, M1)**1.5
    x2 = H*np.linspace(0, 1, M2)**1.5

    h1 = np.diff(x1)
    h2 = np.diff(x2)

    h = np.concatenate((-h1[::-1], h2))

    # coordinates at half-points:
    x1_1_2 = x1[:-1] + h1 / 2
    x2_1_2 = x2[:-1] + h2 / 2
    x_1_2 = np.concatenate((np.flipud(x1_1_2), x2_1_2))

    x = np.concatenate((np.flipud(x1[:-1]), x2))

    D_CONSTANT = True
    if D_CONSTANT:
        D1 = 1.2 + np.zeros_like(x1_1_2**2)
        D2 = 2.2 + np.zeros_like(x2_1_2**2)

        D1_x = 1.2 + np.zeros_like(x1**2)
        D2_x = 2.2 + np.zeros_like(x2**2)
        D1_prime = np.zeros_like(2 * x1)
        D2_prime = np.zeros_like(2 * x2)
    else:
        D1 = 1.2 + x1_1_2**2
        D2 = 2.2 + x2_1_2**2

        D1_x = 1.2 + x1**2
        D2_x = 2.2 + x2**2
        D1_prime = 2 * x1
        D2_prime = 2 * x2


    ratio_D = D1_x[0] / D2_x[0]
    r = d*np.cos(1) * (ratio_D - 1)

    t_n, t = t, t + dt
    all_times = np.arange(t+dt, t+LENGTH, dt)
    neumann = [d * cos(d * x2_1_2[-1]+1) + r for t in all_times]
    dirichlet = [sin(d * x1[-1]+1) + T * t for t in all_times]

    # Note: f is a local approximation !
    f2 = [T * (1 + c * t) - a * (d*cos(d * x2+1) + r) + c * (sin(d * x2+1) + r*x2)
        + D2_x * d * d * sin(d * x2+1) - D2_prime * (d * cos(d * x2+1) + r) for t in all_times]


    f1 = [T * (1 + c * t) + d * a * cos(d * x1+1) + c * sin(d * x1+1) \
        + D1_x * d * d * sin(d * x1+1) - D1_prime * d * cos(d * x1+1) for t in all_times]

    u0 = np.concatenate(
        (sin(d * x1[-1:0:-1]+1) + T * t_n, sin(d * x2+1) + T * t_n + r*x2))

    u_solution = [u0]
    ustar_interface = np.zeros_like(all_times)
    phistar_interface = np.zeros_like(all_times)
    for i,t in enumerate(all_times):
        u_np1, ustar_interface[i], phistar_interface[i] = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                                              D2=D2, a=a, c=c, dt=dt, f1=f1[i], f2=f2[i],
                                              neumann=neumann[i], dirichlet=dirichlet[i],
                                              u_nm1=u_solution[i])
        u_solution += [u_np1]


    # Schwarz parameters:
    Lambda_1 = 1e12
    Lambda_2 = 0.

    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1 - 1:]
    to_plot = []
    ecart = []
    np.random.seed(1)

    for _ in range(40):
        # random fixed false initialization:
        u1_interface = ustar_interface + np.array([np.random.random() for _ in all_times])-.5
        phi1_interface = phistar_interface + np.array([np.random.random() for _ in all_times])-.5
        u2_interface = np.zeros_like(u1_interface)
        phi2_interface = np.zeros_like(phi1_interface)
        all_u1_interfaces = [np.copy(u1_interface)]

        ecart += [[]]
        # Beginning of iterations:
        for _ in range(100):
            u2_nm1 = [u2_0]
            for i, t in enumerate(all_times):
                u2_ret, u2_interface[i], phi2_interface[i] = integrate_one_step(
                    M=M2,
                    h=h2,
                    D=D2,
                    a=a,
                    c=c,
                    dt=dt,
                    f=f2[i],
                    bd_cond=neumann[i],
                    Lambda=Lambda_2,
                    u_nm1=u2_nm1[i],
                    u_interface=u1_interface[i],
                    phi_interface=phi1_interface[i],
                    upper_domain=True)
                u2_nm1 += [u2_ret]

            u1_nm1 = [u1_0]
            for i, t in enumerate(all_times):
                u1_ret, u1_interface[i], phi1_interface[i] = integrate_one_step(
                    M=M1,
                    h=h1,
                    D=D1,
                    a=a,
                    c=c,
                    dt=dt,
                    f=f1[i],
                    bd_cond=dirichlet[i],
                    Lambda=Lambda_1,
                    u_nm1=u1_nm1[i],
                    u_interface=u2_interface[i],
                    phi_interface=phi2_interface[i],
                    upper_domain=False)
                u1_nm1 += [u1_ret]

            # frequentiel :
            #interface_err = max(np.abs(np.fft.fft(u1_interface - ustar_interface, norm="ortho")))
            interface_err = max(np.abs(u1_interface - ustar_interface))
            all_u1_interfaces += [np.copy(u1_interface)]
            ecart[-1] += [interface_err]
            """
        import matplotlib.pyplot as plt
        plt.plot(np.abs(np.fft.fftshift(np.fft.fft(all_u1_interfaces[0] - ustar_interface, norm="ortho"))), "k--")
        plt.plot(np.abs(np.fft.fftshift(np.fft.fft(all_u1_interfaces[1] - ustar_interface, norm="ortho"))), "r")
        plt.plot(np.abs(np.fft.fftshift(np.fft.fft(all_u1_interfaces[2] - ustar_interface, norm="ortho"))), "g")
        plt.plot(np.abs(np.fft.fftshift(np.fft.fft(all_u1_interfaces[3] - ustar_interface, norm="ortho"))), "b")
        plt.title(discretisation.name())
        plt.show()
        to_plot += [np.abs(np.fft.fftshift(np.fft.fft(all_u1_interfaces[2]-ustar_interface, norm="ortho"))/np.fft.fftshift(np.fft.fft(all_u1_interfaces[1]-ustar_interface, norm="ortho")))]
    import matplotlib.pyplot as plt
    plt.plot(np.mean(np.array(to_plot), axis=0), "r--")
    plt.title(discretisation.name())
    plt.show()
            """
    return np.mean(np.array(ecart), axis=0)


def schwarz_convergence(discretisation):
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx+1) + Tt in \Omega_1,
    # u(x, t) = sin(dx+1) + Tt + rx      in \Omega_2
    # where r = dcos(1) * (D1/D2 - 1)

    integrate_one_step_star = discretisation.integrate_one_step_star
    integrate_one_step = discretisation.integrate_one_step

    H = 100
    a = 0.
    c = 0.0

    T = 5.
    d = .8
    t = 3.
    dt = 0.01
    M1, M2 = 500, 500
    # h[0] is 10^-6, so the error at interface should be 10^-12 xD

    x1 = -H*np.linspace(0, 1, M1)**2
    x2 = H*np.linspace(0, 1, M2)**2

    h1 = np.diff(x1)
    h2 = np.diff(x2)

    h = np.concatenate((-h1[::-1], h2))

    # coordinates at half-points:
    x1_1_2 = x1[:-1] + h1 / 2
    x2_1_2 = x2[:-1] + h2 / 2
    x_1_2 = np.concatenate((np.flipud(x1_1_2), x2_1_2))

    x = np.concatenate((np.flipud(x1[:-1]), x2))

    D1 = 1.2 + np.zeros_like(x1_1_2**2)
    D2 = 2.5 + np.zeros_like(x2_1_2**2)

    D1_x = 1.2 + np.zeros_like(x1**2)
    D2_x = 2.5 + np.zeros_like(x2**2)
    D1_prime = 2 * np.zeros_like(x1)
    D2_prime = 2 * np.zeros_like(x2)

    ratio_D = D1_x[0] / D2_x[0]
    r = d*np.cos(1) * (ratio_D - 1)

    t_n, t = t, t + dt
    neumann = d * cos(d * x2_1_2[-1]+1) + r
    dirichlet = sin(d * x1[-1]+1) + T * t

    # Note: f is a local approximation !

    f2 = T * (1 + c * t) - a * (d*cos(d * x2+1) + r) + c * (sin(d * x2+1) + r*x2)   \
        + D2_x * d * d * sin(d * x2+1) - D2_prime * (d * cos(d * x2+1) + r)

    f1 = T * (1 + c * t) - d * a * cos(d * x1+1) + c * sin(d * x1+1)   \
        + D1_x * d * d * sin(d * x1+1) - D1_prime * d * cos(d * x1+1)

    u0 = np.concatenate(
        (sin(d * x1[-2::-1]+1) + T * t_n, sin(d * x2+1) + T * t_n + r*x2))
    u1 = np.concatenate(
        (sin(d * x1[-1:0:-1]+1) + T * t, sin(d * x2+1) + T * t + r*x2))

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1,
                                                                          M2=M2, h1=h1, h2=h2, D1=D1,
                                                                          D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                                                                          neumann=neumann, dirichlet=dirichlet, u_nm1=u0)
    """
    import matplotlib.pyplot as plt
    D_tot = np.concatenate((D1[::-1], D2))
    u1_np1 = u_np1[:f1.shape[0]]
    u2_np1 = u_np1[f1.shape[0]-1:]
    u1_np1 = u1_np1[::-1]
    #plt.plot(x1_1_2, D1*np.diff(u1_np1)/np.diff(x1), "g")
    #plt.plot(x2_1_2, D2*np.diff(u2_np1)/np.diff(x2), "g")
    #plt.plot(x1, D1_x * d*cos(d * x1+1), "r--")
    #plt.plot(x2, D2_x*(d*cos(d * x2+1) + r), "r--")
    plt.plot(x, u1, "g")
    plt.plot(x, u_np1, "r--")
    plt.show()
    """

    # Schwarz parameters:
    Lambda_1 = 1e12
    Lambda_2 = 0.

    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1 - 1:]
    u1_1 = np.flipud(u_np1[:M1])
    u2_1 = u_np1[M1 - 1:]

    # random fixed false initialization:
    u_interface = np.random.random()
    phi_interface = np.random.random()

    ecart = []
    # Beginning of iterations:
    for i in range(300):
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
        ecart += [abs(u_interface - real_u_interface)]#np.linalg.norm(u_np1 - u_np1_schwarz)]
    return ecart

def test_integration():

    # Our domain is [0,1]
    M = 401
    x = np.linspace(0, 1, M)**6
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2]) / 2
    D = 1.0
    a = 0.0
    c = 0.0
    Y = get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)

    # Simplest case: f=0, u* = x
    f = np.zeros(M)
    f[1:M - 1] = 0 * (hm1 + hm)
    f[-1] = np.pi
    assert np.linalg.norm(np.pi * x - solve_linear(Y, f)) < 1e-10

    dt = 0.1
    number_time_steps = 5000
    f_x0 = np.zeros(number_time_steps + 1)
    u0 = np.zeros_like(f)
    u = integration(u0, Y, f, f_x0, dt=dt, theta=1)  # implicit scheme
    assert (np.linalg.norm(np.pi * x - u[-1])) < 1e-9

    M = 10001
    a = 1.2
    c = 0.3
    d = 1.2

    x = np.linspace(0, 1, M)**3
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2]) / 2

    def D(x): return x
    def D_prime(x): return 1
    D_star = D((x[1:] + x[:-1]) / 2)

    def function_f(x): return d * d * D(x) * sin(d * x) - d * \
        D_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)
    f = np.zeros(M)
    f[1:M - 1] = function_f(x[1:M - 1]) * (hm1 + hm)
    f[-1] = d * cos(d * middle_grid_last)

    Y = get_Y_star(M_star=M, h_star=h, D_star=D_star, a=a, c=c)
    assert np.linalg.norm(sin(d * x) - solve_linear(Y, f)) < 1e-7

    dt = 100
    number_time_steps = 5000
    f_x0 = np.zeros(number_time_steps + 1)
    u0 = np.zeros_like(f)
    u = integration(u0, Y, f, f_x0, dt=dt, theta=1)  # Implicit scheme
    assert np.linalg.norm(sin(d * x) - u[-1]) < 1e-7

    dt = 100
    number_time_steps = 5000
    f_x0 = np.zeros(number_time_steps + 1)
    u0 = np.zeros_like(f)
    u = integration(u0, Y, f, f_x0, dt=dt, theta=0.5)  # Crank-Nicholson scheme
    assert np.linalg.norm(sin(d * x) - u[-1]) < 1e-3

    return "ok"


def test_time_domain():
    # Our domain is [-1,1]
    # Omega_2 is [0,1]
    # Omega_1 is [-1,0]
    M = 11
    a = 1.2
    c = 0.3
    d = 1.2

    Lambda1 = 1.0
    Lambda2 = 2.0

    x2 = np.linspace(0, 1, M)**3
    h2 = np.diff(x2)
    h2m1 = h2[:-1]
    h2m = h2[1:]
    middle_grid_last2 = (x2[-1] + x2[-2]) / 2
    def D2(x): return 1.1 - x * x
    def D2_prime(x): return -2 * x
    D2_send = D2((x2[1:] + x2[:-1]) / 2)
    def function_f2(x): return d * d * D2(x) * sin(d * x) - d * \
        D2_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)

    # Omega_1 values:
    x1 = np.linspace(0, -1, M)**3
    h1 = np.diff(x1)
    h1m1 = h1[:-1]
    h1m = h1[1:]
    def D1(x): return 1.1 - x * x
    def D1_prime(x): return -2 * x
    D1_send = D1((x1[1:] + x1[:-1]) / 2)
    def function_f1(x): return d * d * D1(x) * sin(d * x) - d * \
        D1_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)

    f1 = np.zeros(M)
    f1[1:M - 1] = function_f1(x1[1:M - 1]) * (h1m1 + h1m)
    f1[-1] = sin(d * x1[-1])  # Dirichlet bd condition

    f2 = np.zeros(M)
    f2[1:M - 1] = function_f2(x2[1:M - 1]) * (h2m1 + h2m)
    f2[-1] = d * cos(d * middle_grid_last2)  # Neumann bd condition
    f2[0] = d * cos(d * x2[1] / 2) * D2_send[0]  # Robin bd condition, u(0) = 0

    f1[0] = D1_send[0] * d * cos(d * x1[1] / 2)

    u1_init = np.zeros_like(f1)
    u2_init = np.zeros_like(f2)

    nb_time_steps = 2000
    u = solve_u_time_domain(u1_init,
                            u2_init,
                            f_star_0=function_f1(0) * (x2[1] - x1[1]),
                            f1=f1,
                            f2=f2,
                            Lambda_1=Lambda1,
                            Lambda_2=Lambda2,
                            D1=D1_send,
                            D2=D2_send,
                            h1=h1,
                            h2=h2,
                            a=a,
                            c=c,
                            dt=1,
                            number_time_steps=nb_time_steps)

    M = 2 * M - 1
    x = np.linspace(-1, 1, M)**3
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2]) / 2

    def D(x): return 1.1 - x * x
    def D_prime(x): return -2 * x
    D_star = D((x[1:] + x[:-1]) / 2)

    def function_f(x): return d * d * D(x) * sin(d * x) - d * \
        D_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)
    f = np.zeros(M)
    f[1:M - 1] = function_f(x[1:M - 1]) * (hm1 + hm)
    f[-1] = d * cos(d * middle_grid_last)
    f[0] = sin(d * x[0])  # Dirichlet bd condition
    f0_on_time = np.ones(nb_time_steps + 1) * f[0]
    Y = get_Y_star(M_star=M, h_star=h, D_star=D_star, a=a, c=c)
    ureal = solve_linear(Y, f)

    assert np.linalg.norm(ureal - u[-1]) < 1e-10
    return "ok"


"""
    Simple test of the Schwarz method
"""


def test_schwarz():
    # Our domain is [-1,1]
    # Omega_2 is [0,1]
    # Omega_1 is [-1,0]
    M = 10001
    a = 1.2
    c = 0.3
    d = 1.2
    dt = 1000.0
    # TODO ajouter terme correctif

    Lambda1 = 1.0
    Lambda2 = 2.0

    x2 = np.linspace(0, 1, M)**3
    h2 = np.diff(x2)
    h2m1 = h2[:-1]
    h2m = h2[1:]
    middle_grid_last2 = (x2[-1] + x2[-2]) / 2
    def D2(x): return x
    def D2_prime(x): return 1
    D2_send = D2((x2[1:] + x2[:-1]) / 2)
    def function_f2(x): return d * d * D2(x) * sin(d * x) - d * \
        D2_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)
    Y2 = get_Y(M=M,
               Lambda=Lambda2,
               h=h2,
               D=D2_send,
               dt=dt,
               a=a,
               c=c,
               upper_domain=True)

    # Omega_1 values:
    x1 = np.linspace(0, -1, M)**3
    h1 = np.diff(x1)
    h1m1 = h1[:-1]
    h1m = h1[1:]
    def D1(x): return -x
    def D1_prime(x): return -1
    D1_send = D1((x1[1:] + x1[:-1]) / 2)
    def function_f1(x): return d * d * D1(x) * sin(d * x) - d * \
        D1_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)

    Y1 = get_Y(M=M,
               Lambda=Lambda1,
               h=h1,
               D=D1_send,
               dt=dt,
               a=a,
               c=c,
               upper_domain=False)

    f1 = np.zeros(M)
    f1[1:M - 1] = function_f1(x1[1:M - 1]) * (h1m1 + h1m)
    f1[-1] = sin(d * x1[-1])  # Dirichlet bd condition

    f2 = np.zeros(M)
    f2[1:M - 1] = function_f2(x2[1:M - 1]) * (h2m1 + h2m)
    f2[-1] = d * cos(d * middle_grid_last2)  # Neumann bd condition

    f2[0] = d * cos(d * x2[1] / 2) * D2_send[0]  # Robin bd condition, u(0) = 0
    assert np.linalg.norm(sin(d * x2) - solve_linear(Y2, f2)) < 1e-7

    f1[0] = D1_send[0] * d * cos(d * x1[1] / 2)
    assert np.linalg.norm(sin(d * x1) - solve_linear(Y1, f1)) < 2e-6

    initial_error1 = np.linalg.norm(sin(d * x1) - solve_linear(Y1, f1))
    initial_error2 = np.linalg.norm(sin(d * x2) - solve_linear(Y2, f2))

    f1[0] = f2[0] = -3
    # TODO ajuster terme correctif
    ur2 = sin(d * x2)  # ureal 2
    ur1 = sin(d * x1)  # ureal 1
    """
    print("real flux:", D1_send[0]*d) # note: it's not even real, D[0] is D(1/2)
    print("approximated by:", D1_send[0] * (ur1[1] - ur1[0]) / h1[0])
    print("or by:", D1_send[0] * (ur1[1] - ur1[0]) / h1[0] \
            - h1[0] / 2 * ((ur1[0] + ur1[1])/(2*dt) + a*(ur1[1] - ur1[0]) / h1[0] \
                           + c * (ur1[0] + ur1[1])/2 - function_f1(x1[1]/2)))
    print("or by:", D2_send[0] * (ur2[1] - ur2[0]) / h2[0] \
            - h2[0] / 2 * ((ur2[0] + ur2[1])/(2*dt) + a*(ur2[1] - ur2[0]) / h2[0] \
                           + c * (ur2[0] + ur2[1])/2 - function_f2(x2[1]/2)) )
    """

    for i in range(1, 100):
        u1 = solve_linear(Y1, f1)

        f2[0] = D1_send[0] * (u1[1] - u1[0]) / h1[0] + Lambda2 * u1[0] \
            - h1[0] / 2 * ((u1[0] + u1[1]) / (2 * dt) + a * (u1[1] - u1[0]) / h1[0]
                           + c * (u1[0] + u1[1]) / 2 - function_f1(x1[1] / 2)) \
            - h2[0] / 2 * function_f2(x2[1] / 2)
        u2 = solve_linear(Y2, f2)

        f1[0] = D2_send[0] * (u2[1] - u2[0]) / h2[0] + Lambda1 * u2[0] \
            - h2[0] / 2 * ((u2[0] + u2[1]) / (2 * dt) + a * (u2[1] - u2[0]) / h2[0]
                           + c * (u2[0] + u2[1]) / 2 - function_f2(x2[1] / 2)) \
            - h1[0] / 2 * function_f1(x1[1] / 2)

    # we tolerate 1% additional error (before it was 5%)
    tol_err = 1.01 * max(initial_error1, initial_error2)
    assert np.linalg.norm(sin(d * x1) - solve_linear(Y1, f1)) < tol_err
    assert np.linalg.norm(sin(d * x2) - solve_linear(Y2, f2)) < tol_err

    return "ok"


def launch_all_tests():
    print("Test Schwarz method:", test_schwarz())
    print("Test time integration:", test_integration())
    print("Test time domain solution:", test_time_domain())
