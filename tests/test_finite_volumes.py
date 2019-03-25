"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi
from numpy.random import random
from utils_numeric import solve_linear
from finite_volumes import get_Y_star, integrate_one_step_star, integrate_one_step

"""
    Test function of the module.
    If you don't want to print anything, use test_integrate_one_step.
"""
def launch_all_tests():
    print("Test of compact scheme.")
    print("Test integration with finite volumes:", test_integrate_one_step())

"""
    Silent test function of the module.
    Tests the integration functions of finite_volumes module.
"""
def test_integrate_one_step():
    """
    assert "ok" == test_integrate_one_step_simplest()
    assert "ok" == test_integrate_one_step_with_a_c()
    assert "ok" == test_integrate_multi_step_with_a_c()
    assert "ok" == test_integrate_half_domain()
    assert "ok" == not_constant_test_schwarz()
    """
    #TODO remettre les autres tests
    assert "ok" == complete_test_schwarz2()
    return "ok"
"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are NOT constant, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct.
    It would be better to use a more advanced function than a linear one,
    but at least we have analytical results (no approximations are done in
    the finite volumes framework)
"""
def complete_test_schwarz2():
    # Our domain is [0,1]
    # first function : 
    # we define u as u(x, t) = sin(dx + Tt) in \Omega_2,
    # u(x, t) = D1 / D2 * sin(dx + Tt) + (1-D1[0]/D2[0])*sin(Tt)
    #       in \Omega_1

    T = 5.
    d = 8.
    t = 3.
    dt = 0.05
    M1, M2 = 100, 100
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

    D1 = 2.2 + x1_1_2 **2
    D2 = 2.2 + x2_1_2 **2
    D_prime_mean = np.concatenate((np.diff(np.flipud(D1))/h1, np.diff(D2)/h2))
    D1_prime = 2*x1
    D2_prime = 2*x2

    D1 = 2.2 + np.zeros_like(x1_1_2)
    D2 = 2.2 + np.zeros_like(x2_1_2)
    D1_prime = 0.
    D2_prime = 0.

    D_prime = 2*x_1_2
    ratio_D = D1[0] / D2[0]

    a = 0.#1.3
    c = 0.#0.3

    u_theoric = -pi*(1-D2[0]/D1[0]) + np.concatenate((pi*x1*D2[0]/D1[0], pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2[0]/D1[0],
        np.ones_like(x2)))

    t_n, t = t, t + dt
    neumann = cos(d*1)
    dirichlet = sin(-d) + T*t

    # Note: f is an average and not a local approximation !
    f2 = T * (x2_1_2[1:] - x2_1_2[:-1]) \
            + ratio_D*a*(sin(x2_1_2[1:]) - sin(x2_1_2[:-1])) \
            + c*(-ratio_D/d*(cos(x2_1_2[1:]) - cos(x2_1_2[:-1])) \
                            + T*t*(x2_1_2[1:] - x2_1_2[:-1])) \
            - d*ratio_D*(D2[1:]*cos(x2_1_2[1:]) - D2[:-1]*cos(x2_1_2[:-1]))
    f2 /= h2

    # {inf, sup} bounds of the interval ([x-h/2, x+h/2]):
    x1_sup = -x1_1_2[:-1]
    x1_inf = -x1_1_2[1:]

    f1 = T * (x1_sup - x1_inf) + a*(sin(x1_sup) - sin(x1_inf)) \
            + c*(-cos(x1_sup)/d + cos(x1_inf)/d + T*t*(x1_sup - x1_inf)) \
            - d*(D1[1:]*cos(x1_sup) - D1[:-1]*cos(x1_inf))
    f1 /= h1


    u0 = np.concatenate((np.diff(-cos(-d*x1_1_2[::-1])/d - T*t_n*x1_1_2[::-1]),
        np.diff(-ratio_D*cos(d*x2_1_2)/d + T*t_n*x2_1_2))) / h

    u1 = np.concatenate((np.diff(-cos(-d*x1_1_2[::-1])/d - T*t*x1_1_2[::-1]),
        np.diff(-ratio_D*cos(d*x2_1_2)/d + T*t*x2_1_2))) / h

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1:]
    D = np.concatenate((D1[-2:0:-1], D2[:-1]))
    f = np.concatenate((f1[::-1], f2))


    print(u_np1.shape)
    print(u0.shape)
    print(h.shape)
    print(D.shape)
    print(f.shape)
    # VERIFICATION with FINITE diffERENCEs
    print((u_np1[1:-1] - u0[1:-1])/dt + a*(u_np1[2:] - u_np1[:-2])/(2*h[1:-1]) + c*u_np1[1:-1] \
            - (D[1:]*(u_np1[2:] - u_np1[1:-1]) / h[2:] - D[:-1]*(u_np1[1:-1] - u_np1[:-2])/h[1:-1]) \
            - f[1:-1])


    # Schwarz:
    Lambda_1 = 3.0
    Lambda_2 = 0.3

    # random fixed false initialization:
    u_interface=0.0
    phi_interface= d * D2[0]

    import matplotlib.pyplot as plt
    plt.plot(x, u0, "r")
    plt.plot(x, u1, "b")
    plt.plot(x, u_np1, "k--")
    plt.show()
    exit()

    # Beginning of iterations:
    for i in range(200):
        old_u_interface = u_interface
        u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_2, u_nm1=u2_0,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)

        old_interface = u_interface, phi_interface

        u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=u1_0,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)
        u_inter1, phi_inter1 = old_interface
        #print("error:", (u_interface - real_u_interface))
        #print("convergence_rate:", (u_interface - real_u_interface) / (old_u_interface - real_u_interface))
        #input()
        plt.plot(-x1[::-1], u1_ret, "g")
        plt.plot(x2, u2_ret, "g")
        plt.show()

    assert Lambda_2*u_inter1 + phi_inter1 - Lambda_2*u_interface - phi_interface < 1e-15
    assert abs(u_inter1 - real_u_interface) + abs(phi_inter1 - real_phi_interface) < 1e-13
    return "ok"


"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are NOT constant, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct.
    It would be better to use a more advanced function than a linear one,
    but at least we have analytical results (no approximations are done in
    the finite volumes framework)
"""
def complete_test_schwarz():
    # Our domain is [0,1]
    # first function : 
    # we define u as u(x, t) = sin(dx + Tt) in \Omega_2,
    # u(x, t) = D1 / D2 * sin(dx + Tt) + (1-D1[0]/D2[0])*sin(Tt)
    #       in \Omega_1

    T = 5.
    d = 8.
    t = 3.
    dt = 1.
    M1, M2 = 100, 100
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

    D1 = 2.2 + x1_1_2 **2
    D2 = 2.2 + x2_1_2 **2
    D_prime_mean = np.concatenate((np.diff(np.flipud(D1))/h1, np.diff(D2)/h2))
    D1_prime = 2*x1
    D2_prime = 2*x2

    D1 = 2.2 + np.zeros_like(x1_1_2)
    D2 = 2.2 + np.zeros_like(x2_1_2)
    D1_prime = 0.
    D2_prime = 0.

    D_prime = 2*x_1_2
    ratio_D = D2[0] / D1[0]

    a = 0.#1.3
    c = 0.#0.3

    u_theoric = -pi*(1-D2[0]/D1[0]) + np.concatenate((pi*x1*D2[0]/D1[0], pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2[0]/D1[0],
        np.ones_like(x2)))

    neumann = d*cos(d + T*t)
    dirichlet = ratio_D*sin(-d + T*t) + (1-ratio_D)*sin(T*t)

    t_n, t = t, t + dt
    # Note: f is an average and not a local approximation !
    f2 = (T/d + a) * (sin(d*x2_1_2[1:] + T*t) - sin(d*x2_1_2[:-1] + T*t)) \
            - c/d * (cos(d*x2_1_2[1:] + T*t) - cos(d*x2_1_2[:-1] + T*t)) \
            - (D2[1:] * d * cos(d*x2_1_2[1:] + T*t) \
                - D2[:-1] * d * cos(d*x2_1_2[:-1] + T*t))
    f2 /= h2

    # {inf, sup} bounds of the interval ([x-h/2, x+h/2]):
    x1_sup = -x1_1_2[:-1]
    x1_inf = -x1_1_2[1:]

    f1 = (ratio_D*T/d + a*ratio_D) * (sin(d*x1_sup + T*t) - sin(d*x1_inf + T*t)) \
            + (T*(1-ratio_D)*cos(T*t) + c*(1-ratio_D)*sin(T*t)) * (x1_sup - x1_inf) \
            - ratio_D*c/d * (cos(d*x1_sup + T*t) - cos(d*x1_inf + T*t)) \
            - (D1[:-1] * d * cos(d*x1_sup + T*t) \
                - D1[1:] * d * cos(d*x1_inf + T*t))

    f1 /= h1


    # WITH F APPROXIMATION AND NOT AVERAGE
    """
    f2 = T*cos(d*x2 + T*t) + a*d*cos(d*x2 + T*t) + c*sin(d*x2 + T*t) \
            + (D2[1:] + D2[:-1])*d*d*sin(d*x2 + T*t) -D2_prime*d*cos(d*x2 + T*t)
    f1 = ratio_D*T*cos(-d*x1 + T*t) + T*(1-ratio_D)*cos(T*t) + a*ratio_D*d*cos(-d*x1 + T*t) \
            + c*ratio_D*sin(-d*x1 + T*t) + c*(1-ratio_D)*sin(T*t) \
            + (D1[1:] + D1[:-1])*d*d*sin(-d*x1 + T*t) -D1_prime*d*cos(-d*x1 + T*t)

    f1 = np.flipud(f1)

    #WITH f INTEGRAL BUT ASSUMED T=0
    f2 = a*(sin(d*x2_1_2[1:]) - sin(d*x2_1_2[:-1])) - c / d * (cos(d*x2_1_2[1:]) - cos(d*x2_1_2[:-1])) \
        - (D2[1:] * d*cos(d*x2_1_2[1:]) - D2[:-1] * d * cos(d*x2_1_2[:-1]))
    f1 = a*(sin(d*x1_sup) - sin(d*x1_inf)) - c / d * (cos(d*x1_sup) - cos(d*x1_inf)) \
        - (D1[1:] * d*cos(d*x1_sup) - D1[:-1] * d * cos(d*x1_inf))

    """
    # END OF DANGER ZONE
    u0 = np.concatenate((np.diff(-ratio_D / d * cos(-d*x1_1_2[::-1] + T*t_n) - (1-ratio_D)*sin(T*t_n)*x1_1_2[::-1]),
        np.diff(-cos(d*x2_1_2 + T*t_n) / d))) / h

    #u0 = np.concatenate((ratio_D * sin(-d*x1 + T*t_n) + (1-ratio_D)*sin(T*t_n),
    #    sin(d*x2+T*t_n)))
    u1 = np.concatenate((ratio_D * sin(-d*x1 + T*t) + (1-ratio_D)*sin(T*t),
        sin(d*x2+T*t)))

    u1 = np.concatenate((np.diff(-ratio_D / d * cos(-d*x1_1_2[::-1] + T*t) - (1-ratio_D)*sin(T*t)*x1_1_2[::-1]),
        np.diff(-cos(d*x2_1_2 + T*t) / d))) / h

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1:]

    # Schwarz:
    Lambda_1 = 3.0
    Lambda_2 = 0.3

    # random fixed false initialization:
    u_interface=0.0
    phi_interface= d * D2[0]

    import matplotlib.pyplot as plt
    plt.plot(x, u0, "r")
    plt.plot(x, u1, "b")
    plt.plot(x, u_np1, "k--")
    plt.show()
    exit()

    # Beginning of iterations:
    for i in range(200):
        old_u_interface = u_interface
        u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_2, u_nm1=u2_0,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)

        old_interface = u_interface, phi_interface

        u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=u1_0,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)
        u_inter1, phi_inter1 = old_interface
        #print("error:", (u_interface - real_u_interface))
        #print("convergence_rate:", (u_interface - real_u_interface) / (old_u_interface - real_u_interface))
        #input()
        plt.plot(-x1[::-1], u1_ret, "g")
        plt.plot(x2, u2_ret, "g")
        plt.show()

    assert Lambda_2*u_inter1 + phi_inter1 - Lambda_2*u_interface - phi_interface < 1e-15
    assert abs(u_inter1 - real_u_interface) + abs(phi_inter1 - real_phi_interface) < 1e-13
    return "ok"

"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are NOT constant, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct.
    It would be better to use a more advanced function than a linear one,
    but at least we have analytical results (no approximations are done in
    the finite volumes framework)
"""
def not_constant_test_schwarz():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0

    dt = 0.01
    M1, M2 = 14, 10
    h1, h2 = 1/M1, 1/M2
    h1 = 1/M1 + np.zeros(M1)
    h2 = 1/M2 + np.zeros(M2)
    h1 = np.diff(np.cumsum(np.concatenate(([0],h1)))**3)
    h2 = np.diff(np.cumsum(np.concatenate(([0],h2)))**2)

    # Center of the volumes are x, sizes are h
    x1 = np.cumsum(np.concatenate(([h1[0]/2],(h1[1:] + h1[:-1])/2)))
    x2 = np.cumsum(np.concatenate(([h2[0]/2],(h2[1:] + h2[:-1])/2)))
    x1 = np.flipud(x1)

    # coordinates at half-points:
    x1_1_2 = np.cumsum(np.concatenate(([0],h1)))
    x2_1_2 = np.cumsum(np.concatenate(([0],h2)))
    x_1_2 = np.concatenate((np.flipud(x1_1_2[:-1]), x2_1_2))

    x = np.concatenate((x1, x2))

    D1 = 2.2 + x1_1_2 **2
    D2 = 1.2 + x2_1_2 **2
    D_prime_mean = np.concatenate((np.diff(np.flipud(D1))/h1, np.diff(D2)/h2))
    D1_prime = 2*x1_1_2
    D2_prime = 2*x2_1_2
    D_prime = 2*x_1_2

    a = 1.3
    c = 0.3

    u_theoric = -pi*(1-D2[0]/D1[0]) + np.concatenate((pi*x1*D2[0]/D1[0], pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2[0]/D1[0],
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f is an average and not a local approximation !
    f = c * u_theoric + (a - D_prime_mean)*partial_xu
    f1 = np.flipud(f[:M1])
    f2 = f[M1:]

    u0 = np.zeros_like(u_theoric)
    u_n, u_interface, phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u_n)

    u1_n = np.flipud(u_n[:M1])
    u2_n = u_n[M1:]
    u1_np1 = np.flipud(u_np1[:M1])
    u2_np1 = u_np1[M1:]

    # Schwarz:
    Lambda_1 = 3.0
    Lambda_2 = 0.3

    # random fixed false initialization:
    u_interface=3.0
    phi_interface=16.0

    # Beginning of iterations:
    for i in range(200):
        old_u_interface = u_interface
        u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_2, u_nm1=u2_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)

        old_interface = u_interface, phi_interface

        u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=u1_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)
        u_inter1, phi_inter1 = old_interface
        #print("error:", (u_interface - real_u_interface))
        #print("convergence_rate:", (u_interface - real_u_interface) / (old_u_interface - real_u_interface))
        #input()

    assert Lambda_2*u_inter1 + phi_inter1 - Lambda_2*u_interface - phi_interface < 1e-15
    assert abs(u_inter1 - real_u_interface) + abs(phi_inter1 - real_phi_interface) < 1e-13
    return "ok"

"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are constant, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct,
    except for the variability of D and h.
"""
def test_integrate_half_domain():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0

    M1, M2 = 10, 10
    h1, h2 = 1/M1, 1/M2
    dt = 0.01
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.3
    c = 0.3

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2/D1,
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f should be an average and not a local approximation
    f = c * u_theoric + a*partial_xu
    f1 = np.flipud(f[:M1])
    f2 = f[M1:]

    u0 = np.zeros_like(u_theoric)
    u_n, u_interface, phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u_nm1=u_n)

    u1_n = np.flipud(u_n[:M1])
    u2_n = u_n[M1:]
    u1_np1 = np.flipud(u_np1[:M1])
    u2_np1 = u_np1[M1:]

    # Schwarz:
    Lambda_1 = 3.0
    Lambda_2 = 0.3

    # random fixed false initialization:
    u_interface=3.0
    phi_interface=16.0

    # Beginning of iterations:
    for i in range(200):
        u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_2, u_nm1=u2_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)

        old_interface = u_interface, phi_interface

        u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=u1_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)
        u_inter1, phi_inter1 = old_interface

    assert Lambda_2*u_inter1 + phi_inter1 - Lambda_2*u_interface - phi_interface < 1e-15
    assert abs(u_inter1 - real_u_interface) + abs(phi_inter1 - real_phi_interface) < 1e-15
    return "ok"

"""
    Tests the function "integrate_one_step_star" of finite_volumes.
    h and D are constant, a and c are != 0.
    D is discontinuous over the interface.
    We make multi steps to reach convergence rather
    than one big step
"""
def test_integrate_multi_step_with_a_c():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 10, 10
    h1, h2 = 1/M1, 1/M2
    dt = 1
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = 0.1

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2/D1,
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    f = c * u_theoric + a*partial_xu
    # Note: f should be an average and not a local approximation
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u = np.zeros_like(u_theoric)
    for i in range(13):
        u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                    D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                    neumann=neumann, dirichlet=dirichlet, u_nm1=u)[0]

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"


"""
    Tests the function "integrate_one_step_star" of finite_volumes.
    h and D are constant, a and c are != 0.
    D is discontinuous over the interface.
"""
def test_integrate_one_step_with_a_c():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 10, 10
    h1, h2 = 1/M1, 1/M2
    dt = 10000000
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = 0.4

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2/D1,
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f should be an average and not a local approximation
    f = c * u_theoric + a*partial_xu
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u0 = np.zeros_like(u_theoric)
    u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                neumann=neumann, dirichlet=dirichlet, u_nm1=u0)[0]

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"


"""
    Tests the function "integrate_one_step_star" of finite_volumes.
    h and D are constant, a and c are zeros.
    D is discontinuous over the interface.
    The functions used are extremely simple because it's not trivial
    enough to have an analytical solution
"""
def test_integrate_one_step_simplest():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 20, 20
    h1, h2 = 1/M1, 1/M2
    dt = 100000
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))


    D1 = 2.4
    D2 = 1.4
    a = 0.0
    c = 0.0

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    neumann = pi
    dirichlet = -pi
    f = np.zeros_like(x)
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u0 = np.zeros_like(x)
    u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                neumann=neumann, dirichlet=dirichlet, u_nm1=u0)[0]

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"

