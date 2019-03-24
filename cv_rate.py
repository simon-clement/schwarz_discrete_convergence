import numpy as np
from numpy import pi
import finite_difference
import finite_volumes

"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are constant, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct,
    except for the variability of D and h.
"""
def rate_finite_volumes(Lambda_1, Lambda_2=0):
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0

    M1, M2 = 100, 100
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

    u_n = np.zeros_like(f)

    u1_n = np.flipud(u_n[:M1])
    u2_n = u_n[M1:]

    # random fixed false initialization:
    u_interface=3.0
    phi_interface=16.0

    ecart = []

    for i in range(4):
        km1_interface = u_interface
        phikm1_interface = phi_interface
        u2_ret, u_interface, phi_interface = finite_volumes.integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_2, u_nm1=u2_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)


        u1_ret, u_interface, phi_interface = finite_volumes.integrate_one_step(M=M1,
                h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=u1_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)

        ecart += [abs(u_interface)] # on considère la solution=0?

    return ecart[1] / ecart[0]


"""
    Tests the function "integrate_one_step" of finite_difference.
    h and D are constant and equal on domains, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct,
    except for the variability of D and h.
"""
def rate_finite_differences(Lambda_2, Lambda_1=1.0):
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0

    M1, M2 = 100, 100
    dt = 0.01
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 , 0, M1)
    x2 = np.linspace(0, 1, M2)
    h1 = np.diff(x1)
    h2 = np.diff(x2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 2.2
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

    u_n = np.zeros_like(f)

    u1_n = np.flipud(u_n[:M1])
    u2_n = u_n[M1:]

    # random fixed false initialization:
    u_interface=-3.0
    phi_interface=16.0

    u_inter1 = u_interface
    phi_inter1 = phi_interface
    # Beginning of iterations:
    # TODO faire converger l'algo xD à la fin, trouver le taux
    # de convergence O:)
    ecart = []

    for i in range(4):
        km1_interface = u_interface
        phikm1_interface = phi_interface
        u2_ret, u_interface, phi_interface = finite_difference.integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_2, u_nm1=u2_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)

        u1_ret, u_interface, phi_interface = finite_difference.integrate_one_step(M=M1,
                h=-h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=u1_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)
        ecart += [abs(u_interface)]

    return ecart[1] / ecart[0]




"""
    Tests the function "integrate_one_step" of finite_difference.
    h and D are constant and equal on domains, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct,
    except for the variability of D and h.
    TODO : make it work
"""
def test_convergence_to_star():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0

    M1, M2 = 101, 101
    dt = 0.01
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 , 0, M1)
    x2 = np.linspace(0, 1, M2)
    h1 = np.diff(x1)
    h2 = np.diff(x2)
    x = np.concatenate((x1, x2))

    D1 = 0.2
    D2 = 1.1
    a = 1.0
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

    u_n = np.zeros_like(u_theoric)

    u1_n = np.flipud(u_n[:M1])
    u2_n = u_n[M1:]

    u_n_star = np.concatenate((np.flipud(u1_n[1:]), u2_n))
    u_star, u_real_interface, phi_real_interface = finite_difference.integrate_one_step_star(M1=M1,
                                                                                                           M2=M2, h1=-h1, h2=h2,
                                     D1=D1, D2=D2, a=a, c=c, dt=dt,
                                     f1=f1, f2=f2,
                                     neumann=neumann, dirichlet=dirichlet,
                                     u_nm1=u_n_star)
    # Schwarz:
    Lambda_1 = 1.0
    Lambda_2 = 2.1071897

    # random fixed false initialization:
    u_interface=-3.0
    phi_interface=16.0

    u_inter1 = u_interface
    phi_inter1 = phi_interface
    # Beginning of iterations:
    # TODO faire converger l'algo xD à la fin, trouver le taux
    # de convergence O:)
    ecart = []

    for i in range(40):
        km1_interface = u_interface
        phikm1_interface = phi_interface
        u2_ret, u_interface, phi_interface = finite_difference.integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_2, u_nm1=u2_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)

        u1_ret, u_interface, phi_interface = finite_difference.integrate_one_step(M=M1,
                h=-h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=u1_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)
        ecart += [abs(u_interface - u_real_interface)]

    print(ecart)
    return ecart[1] / ecart[0]


if __name__ == "__main__":
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
            print("This wasn't implemented yet.")
            print("finite differences:", rate_finite_differences(-842., 1.))
            print("finite volumes:", rate_finite_volumes(3.0, 0.3))
        elif sys.argv[1] == "optimize":
            from scipy.optimize import minimize_scalar
            print("rate finite volumes:", minimize_scalar(rate_finite_volumes))
            print("rate finite differences:", minimize_scalar(rate_finite_differences))

        elif sys.argv[1] == "debug":
            test_convergence_to_star()
