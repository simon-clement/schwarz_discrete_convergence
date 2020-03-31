"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi
from numpy.random import random
from tests.utils_numeric import solve_linear
from discretizations.finite_volumes_spline2 import FiniteVolumesSpline2
from figures import DEFAULT
finite_volumes = DEFAULT.new(FiniteVolumesSpline2)
#get_Y_star = finite_volumes.get_Y_star
integrate_one_step = finite_volumes.integrate_one_step
#integrate_one_step_star = finite_volumes.integrate_one_step_star
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
    assert "ok" == simplest_matrix()
    assert "ok" == order_error_term_with_adv_reac()
    #assert "ok" == order_space_error_term_star()
    assert "ok" == order_error_term_star()
    """
    assert "ok" == order_error_term_linear_time_domain2()
    assert "ok" == order_error_term_linear_time()
    #assert "ok" == order_error_term_exp()
    """
    assert "ok" == complete_test_schwarz()
    assert "ok" == modified_equation()
    """
    return "ok"

def order_error_term_with_adv_reac():
    """
    Simple Case : a=c=0
    on se place en (-1, 1)
    Neumann en 1, Dirichlet en -1
    """
    ret = []
    main_error_term = []
    N = 1
    T = .1
    Courant = .1
    values_param = np.arange(40, 400, 10)
    for N in values_param:
        dt = T/N
        D = 2.
        a = .5
        c = .3

        M = 2*int(np.sqrt(Courant/(D*dt)))

        M1, M2 = M,M
        h1 = 1 / M1 + np.zeros(M1)
        h2 = 1 / M2 + np.zeros(M2)
        h = h1[0]
        
        t_n = 0.

        D1, D2 = D, D

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x1 = np.flipud(x1)
        x = np.concatenate((-x1, x2))

        def u_real(x, t):
            return np.sin(x + t)

        def u_bar(x_1_2, h, t):
            return 1/h * (np.cos(x_1_2 + t) - np.cos(x_1_2+h + t))

        def flux(x, t):
            return D * np.cos(x+t)

        def derivee_quatre(x, t):
            return np.cos(x+t)
        def derivee_t1x2(x, t):
            return np.sin(x+t)
        def derivee_t2(x, t):
            return -np.cos(x+t)

        def f_bar(x, h, t):
            return 1/h*((1+a)*(np.sin(x+t+h) - np.sin(x+t)) - (D+c) * (np.cos(x+t+h) - np.cos(x+t)))

        x_flux = np.concatenate((x-h/2,[1]))

        
        u0 = u_bar(x - h/2, h, t_n)
            

        import matplotlib.pyplot as plt
        for t in np.linspace(0, T, N, endpoint=False):
            t_n = t
            t_np1 = t + dt

            dirichlet = u_real(-1, t_np1)
            neumann = flux(1, t_np1)/D # D\partial_x u = flux

            f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
            f2 = f_bar(x_flux[-M2-1:-1], h, t_np1)

            u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1,
                                                                                  M2=M2, h1=h1, h2=h2, D1=D1,
                                                                                  D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                                                                                  neumann=neumann, dirichlet=dirichlet, u_nm1=u0)
            u0 = u_np1
            uexact = u_bar(x - h/2, h, t_np1)

        uexact = u_bar(x - h/2, h, t_np1)
        ret += [np.linalg.norm(uexact - u_np1)/np.linalg.norm(uexact)]
        error = -D*h**2/12 * derivee_quatre(x_flux, t_np1) + h**2/12*derivee_t1x2(x_flux, t_np1)
        error2 = h**4/(240*dt)*derivee_quatre(x_flux, t_np1) - dt/2 * derivee_t2(x_flux, t_np1)# - h**2*dt/24 * derivee_quatre(x_flux, t_np1) + dt**2/6*np.sin(x_flux+t_np1)
        main_error_term += [np.linalg.norm(error+error2)/np.linalg.norm(uexact)]

    values_param = 1/values_param
    ordre = (np.log(ret[-1]) - np.log(ret[5]))/(np.log(values_param[-1]) - np.log(values_param[5]))
    print("ORDER sinus (few points with advection and reaction, should be ~1):  ", ordre)

    assert abs(ordre - 1) < .1
    return "ok"

def modified_equation():
    """
    Simple Case : a=c=0
    on se place en (-1, 1)
    Neumann en 1, Dirichlet en -1
    """
    ret = []
    main_error_term = []
    N = 1
    T = .1
    Courant = .001
    values_param = (8000, )#np.arange(20, 400, 10)
    for N in values_param:
        dt = T/N
        D = 5.

        M = 2*int(np.sqrt(Courant/(D*dt)))

        M1, M2 = M,M
        h1 = 1 / M1 + np.zeros(M1)
        h2 = 1 / M2 + np.zeros(M2)
        h = h1[0]
        
        t_n = 0.

        D1, D2 = D, D

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x1 = np.flipud(x1)
        x = np.concatenate((-x1, x2))

        def u_real(x, t):
            return np.sin(x + t)

        def u_bar(x_1_2, h, t):
            return 1/h * (np.cos(x_1_2 + t) - np.cos(x_1_2+h + t))

        def flux(x, t):
            return D * np.cos(x+t)

        def derivee_quatre(x, t):
            return np.cos(x+t)
        def derivee_x4_bar(x, h, t):
            return 1/h*(np.cos(x+t) - np.cos(x+t+h))
        def derivee_x2(x, t):
            return -np.sin(x+t)
        def derivee_x4(x, t):
            return -np.sin(x+t)
        def derivee_t2_bar(x, h, t):
            return 1/h*(np.cos(x+t+h) - np.cos(x+t))
        def derivee_t3_bar(x, h, t):
            return 1/h*(np.sin(x+t+h) - np.sin(x+t))

        x_flux = np.concatenate((x-h/2,[1]))

        
        u0 = u_bar(x - h/2, h, t_n)
        u_nm1 = u0
            

        import matplotlib.pyplot as plt
        for t in np.linspace(0, T, N, endpoint=False):
            t_n = t
            t_np1 = t + dt

            #modified f_bar :
            def f_bar(x, h, t):
                return 1/h*(np.sin(x+t+h) - np.sin(x+t) - D * (np.cos(x+t+h) - np.cos(x+t)) - dt/2*(np.cos(x+t+h) - np.cos(x+t)) )



            # modified dirichlet condition :
            dirichlet = u_real(-1, t_np1) - h**2/12 * derivee_x2(-1, t_np1)# - derivee_x4(-1, t_np1) * (D**2*dt**2/2 - h**4/180)
            neumann = flux(1, t_np1)/D # D\partial_x u = flux

            f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
            f2 = f_bar(x_flux[-M2-1:-1], h, t_np1)
            f = f_bar(x_flux[:-1], h, t_np1)

            u_np1, real_u_interface, real_phi_interface, phi = integrate_one_step_star(M1=M1,
                                                                                  M2=M2, h1=h1, h2=h2, D1=D1,
                                                                                  D2=D2, a=0., c=0., dt=dt, f1=f1, f2=f2,
                                                                                  neumann=neumann, dirichlet=dirichlet, u_nm1=u_nm1, get_phi=True)
            space_error_term = f - (u_np1 - u_nm1)/dt + (flux(x_flux[1:], t_np1) - flux(x_flux[:-1], t_np1))/h
            #space_error_term -= dt/2 * derivee_t2_bar(x_flux[:-1], h, t_np1) #- h**4/(240*dt) * derivee_x4_bar(x_flux[:-1], h, t_np1)
            u_nm1 = u_np1
            uexact = u_bar(x - h/2, h, t_np1)

        # Now we want to find the equivalent equation solved by our system
        # So we compute f - (\Bar{u}^np1 - \Bar{u}^n)/dt + (phi1/2 - phi-1/2) / h
        # To get the space error (theorically h^4/240dt u^(4) (or d^(4) ?)
        uexact = u_bar(x - h/2, h, t_np1)
        ret += [np.linalg.norm(space_error_term)/np.linalg.norm(uexact)]
        #ret += [np.linalg.norm(uexact - u_np1)/np.linalg.norm(uexact)]

    def to_optimize(params):
        return np.linalg.norm(space_error_term - derivee_t2_bar(x_flux[:-1], h, t_np1)*params[0] - derivee_t3_bar(x_flux[:-1], h, t_np1)*params[1])
    from scipy import optimize
    res_op = optimize.minimize(fun=to_optimize, x0=np.array((1e-4, 4e-4)), options={'gtol': 1e-08, 'eps' : 1e-12})
    print("optimal modif equations :", res_op.x)
    print("for dt=", dt, "and h=", h)

    # on est en gros à 1e-4 pour le terme en sin
    plt.plot(x, space_error_term)
    plt.plot(x, derivee_t2_bar(x_flux[:-1], h, t_np1)*res_op.x[0] + derivee_t3_bar(x_flux[:-1], h, t_np1)*res_op.x[1])
    plt.show()
    values_param = 1/values_param
    plt.loglog(values_param, ret)
    ordre = (np.log(ret[-1]) - np.log(ret[5]))/(np.log(values_param[-1]) - np.log(values_param[5]))
    print("ORDER error term (should be ~?):  ", ordre)
    #print("ORDER sinus (should be ~1):  ", ordre)
    plt.show()

    assert abs(ordre - 1) < .1
    return "ok"

def order_space_error_term():
    """
    Simple Case : a=c=0
    on se place en (-1, 1)
    Neumann en 1, Dirichlet en -1
    """
    ret = []
    main_error_term = []
    N = 1
    T = 100.
    Courant = 10
    values_param = 2**np.arange(4, 15)
    for N in values_param:
        dt = T/N
        D = .1

        M = 2*int(np.sqrt(Courant/(D*dt)))
        #M = 2*int(np.sqrt(Courant/(D*dt**2)))
        M = 2

        M1, M2 = M,M
        h1 = 1 / M1 + np.zeros(M1)
        h2 = 1 / M2 + np.zeros(M2)
        h = h1[0]
        
        t_n = 0.

        D1, D2 = D, D

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x1 = np.flipud(x1)
        x = np.concatenate((-x1, x2))

        def u_real(x, t):
            return np.sin(x + t)

        def u_bar(x_1_2, h, t):
            return 1/h * (np.cos(x_1_2 + t) - np.cos(x_1_2+h + t))

        def flux(x, t):
            return D * np.cos(x+t)

        def derivee_quatre(x, t):
            return np.cos(x+t)
        def derivee_t1x2(x, t):
            return np.sin(x+t)
        def derivee_t2(x, t):
            return -np.cos(x+t)

        def f_bar(x, h, t):
            return 1/h*(np.sin(x+t+h) - np.sin(x+t) - D * (np.cos(x+t+h) - np.cos(x+t)))

        x_flux = np.concatenate((x-h/2,[1]))

        
        u0 = u_bar(x - h/2, h, t_n)
        phi0 = flux(x_flux, t_n)
            

        import matplotlib.pyplot as plt
        for t in np.linspace(0, T, N, endpoint=False):
            t_n = t
            t_np1 = t + dt

            dirichlet = u_real(-1, t_np1)
            neumann = flux(1, t_np1)/D # D\partial_x u = flux

            f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
            f2 = f_bar(x_flux[-M2-1:-1], h, t_np1)

            u_np1, real_u_interface, real_phi_interface, phi_np1 = integrate_one_step_star(M1=M1,
                                                                                  M2=M2, h1=h1, h2=h2, D1=D1,
                                                                                  D2=D2, a=0., c=0., dt=dt, f1=f1, f2=f2,
                                                                                  neumann=neumann, dirichlet=dirichlet, u_nm1=u0, phi_nm1=phi0, get_phi=True)
            u0 = u_np1
            phi0 = phi_np1
            uexact = u_bar(x - h/2, h, t_np1)

        uexact = u_bar(x - h/2, h, t_np1)
        ret += [np.linalg.norm(uexact - u_np1)/np.linalg.norm(uexact)]
        print(ret[-1])

    values_param = 1/values_param
    ordre = (np.log(ret[-1]) - np.log(ret[-2]))/(np.log(values_param[-1]) - np.log(values_param[-2]))
    print("ORDER sinus (should be ~1):  ", ordre)

    #assert abs(ordre - 1) < .1
    return "ok"

def order_error_term():
    """
    Simple Case : a=c=0
    on se place en (-1, 1)
    Neumann en 1, Dirichlet en -1
    """
    ret = []
    main_error_term = []
    N = 1
    T = .1
    Courant = 10
    values_param = 2**np.arange(2, 5)
    for N in values_param:
        dt = T/N
        D = 5.

        M = 2*int(np.sqrt(Courant/(D*dt)))

        M1, M2 = M,M
        h1 = 1 / M1 + np.zeros(M1)
        h2 = 1 / M2 + np.zeros(M2)
        h = h1[0]
        
        t_n = 0.

        D1, D2 = D, D

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x1 = np.flipud(x1)
        x = np.concatenate((-x1, x2))

        def u_real(x, t):
            return np.sin(x + t)

        def u_bar(x_1_2, h, t):
            return 1/h * (np.cos(x_1_2 + t) - np.cos(x_1_2+h + t))

        def flux(x, t):
            return D * np.cos(x+t)

        def derivee_quatre(x, t):
            return np.cos(x+t)
        def derivee_t1x2(x, t):
            return np.sin(x+t)
        def derivee_t2(x, t):
            return -np.cos(x+t)

        def f_bar(x, h, t):
            return 1/h*(np.sin(x+t+h) - np.sin(x+t) - D * (np.cos(x+t+h) - np.cos(x+t)))

        x_flux = np.concatenate((x-h/2,[1]))

        
        u0 = u_bar(x - h/2, h, t_n)
        phi0 = flux(x_flux, t_n)
            

        import matplotlib.pyplot as plt
        for t in np.linspace(0, T, N, endpoint=False):
            t_n = t
            t_np1 = t + dt

            dirichlet = u_real(-1, t_np1)
            neumann = flux(1, t_np1)/D # D\partial_x u = flux

            f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
            f2 = f_bar(x_flux[-M2-1:-1], h, t_np1)

            u_np1, real_u_interface, real_phi_interface, phi_np1 = integrate_one_step_star(M1=M1,
                                                                                  M2=M2, h1=h1, h2=h2, D1=D1,
                                                                                  D2=D2, a=0., c=0., dt=dt, f1=f1, f2=f2,
                                                                                  neumann=neumann, dirichlet=dirichlet, u_nm1=u0, phi_nm1=phi0, get_phi=True)
            u0 = u_np1
            phi0 = phi_np1
            uexact = u_bar(x - h/2, h, t_np1)

        uexact = u_bar(x - h/2, h, t_np1)
        ret += [np.linalg.norm(uexact - u_np1)/np.linalg.norm(uexact)]

    values_param = 1/values_param
    ordre = (np.log(ret[-1]) - np.log(ret[-2]))/(np.log(values_param[-1]) - np.log(values_param[-2]))
    print("ORDER sinus (should be ~1):  ", ordre)

    assert abs(ordre - 1) < .1
    return "ok"

def order_error_term_exp():
    """
    Simple Case : a=c=0
    on se place en (-1, 1)
    Neumann en 1, Dirichlet en -1
    """
    ret = []
    N = 1
    T = 5
    Courant = 1
    values_param = 2**np.arange(4, 15)
    for N in values_param:
        dt = T/N
        D = 1.

        M = 2*int(np.sqrt(Courant/(D*dt)))

        M1, M2 = M,M
        h1 = 1 / M1 + np.zeros(M1)
        h2 = 1 / M2 + np.zeros(M2)
        h = h1[0]
        
        t_n = 0.

        D1, D2 = D, D

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x1 = np.flipud(x1)
        x = np.concatenate((-x1, x2))

        def u_real(x, t):
            return np.exp(x/np.sqrt(D) + t)

        def u_bar(x_1_2, h, t):
            return np.sqrt(D)/h * (u_real(x_1_2+h, t) - u_real(x_1_2, t))

        def flux(x, t):
            return np.sqrt(D) * u_real(x, t)

        x_flux = np.concatenate((x-h/2,[1]))

        
        u0 = u_bar(x - h/2, h, t_n)
            

        f1 = np.zeros(M1)
        f2 = np.zeros(M2)


        import matplotlib.pyplot as plt
        for t in np.linspace(0, T, N, endpoint=False):
            t_n = t
            t_np1 = t + dt

            dirichlet = u_real(-1, t_np1)
            neumann = 1/np.sqrt(D) * u_real(1, t_np1) # \partial_x u = u/sqrt(D)

            u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1,
                                                                                  M2=M2, h1=h1, h2=h2, D1=D1,
                                                                                  D2=D2, a=0., c=0., dt=dt, f1=f1, f2=f2,
                                                                                  neumann=neumann, dirichlet=dirichlet, u_nm1=u0)
            u0 = u_np1
            uexact = u_bar(x - h/2, h, t_np1)

        uexact = u_bar(x - h/2, h, t_np1)
        ret += [np.linalg.norm(uexact - u_np1)/np.linalg.norm(uexact)]

    values_param = T / values_param
    ordre = (np.log(ret[0]) - np.log(ret[-1]))/(np.log(values_param[0]) - np.log(values_param[-1]))
    print("ORDER exp: (should be ~1): ", (np.log(ret[0]) - np.log(ret[-1]))/(np.log(values_param[0]) - np.log(values_param[-1])))
    assert abs(ordre - 1) < .2
    return "ok"


def simplest_matrix():
    """ We won't define a function u: let's just see if the matrix values are correct
    in the simplest case possible (D=1 or 2, a=c=0, h constant, f=0)
    """
    M1, M2 = 5, 5
    h1 = 1 / M1 + np.zeros(M1)
    h2 = 1 / M2 + np.zeros(M2)
    h = h1[0]
    D1 = 2.
    D2 = D1
    dt = 1e-3
    Y_star = get_Y_star(M1, M2, h1, h2, D1, D2, 0., 0., dt)
    # Now let's compute it analytically:
    lower = -dt /h + 1/12 * h / D1 + np.zeros(M1+M2)
    upper = lower+ np.zeros(M1+M2)
    diagonal = dt * 2/h + 10/12 * h / D1 + np.zeros(M1+M2+1)
    # The problem may come from the boundaries.
    # Dirichlet boundary on diag:", 
    diagonal[0] = -dt/h - 5*h/(12*D1)
    # Neumann condition :
    diagonal[-1] = 1
    #Dirichlet boundary on upper:", dt/h1[0] - h1[0]/(12*D1))
    upper[0] = dt/h - h/(12*D1)
    # Neumann condition on lower :
    lower[-1] = 0

    def comp(a, b):
        return np.linalg.norm(np.abs(a - b))

    assert comp(lower, Y_star[0]) < 1e-15
    assert comp(diagonal, Y_star[1])< 1e-15
    assert comp(upper, Y_star[2])< 1e-15
    return "ok"


"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are NOT constant, a and c are != 0.
    Schwarz algorithm is used to converge to the exact solution.
    If this test pass, then the module should be correct.
"""


def complete_test_schwarz():
    # Our domain is [-1,1]
    # we define u as u(x, t) = sin(dx) + Tt in \Omega_1,
    # u(x, t) = D1 / D2 * sin(dx) + Tt      in \Omega_2

    a = 0.#1.2
    c = 0.#.3

    T = 5.
    d = 8.
    t = 3.
    dt = 4.
    M1, M2 = 1000, 1000
    h1, h2 = 1 / M1, 1 / M2
    h1 = 1 / M1 + np.zeros(M1)
    h2 = 1 / M2 + np.zeros(M2)
    h1 = np.diff(np.cumsum(np.concatenate(([0], h1)))**1)
    h2 = np.diff(np.cumsum(np.concatenate(([0], h2)))**1)
    h = np.concatenate((h1[::-1], h2))

    # Center of the volumes are x, sizes are h
    x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
    x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
    x1 = np.flipud(x1)

    # coordinates at half-points:
    x1_1_2 = np.cumsum(np.concatenate(([0], h1)))
    x2_1_2 = np.cumsum(np.concatenate(([0], h2)))
    x_1_2 = np.concatenate((np.flipud(x1_1_2[:-1]), x2_1_2))

    x = np.concatenate((-x1, x2))

    D1 = 1.2 + np.zeros_like(x1_1_2**2)
    D2 = 1.2 + np.zeros_like(x2_1_2**2)

    ratio_D = D1[0] / D2[0]

    t_n, t = t, t + dt
    neumann = ratio_D * d * cos(d * 1)
    dirichlet = sin(-d) + T * t

    # Note: f is an average and not a local approximation !
    f2 = T * (x2_1_2[1:] - x2_1_2[:-1]) \
        + ratio_D * a * (sin(d * x2_1_2[1:]) - sin(d * x2_1_2[:-1])) \
        + c * (-ratio_D / d * (cos(d * x2_1_2[1:]) - cos(d * x2_1_2[:-1]))
               + T * t * (x2_1_2[1:] - x2_1_2[:-1])) \
        - d * ratio_D * (D2[1:] * cos(d * x2_1_2[1:]) -
                         D2[:-1] * cos(d * x2_1_2[:-1]))
    f2 /= h2

    # {inf, sup} bounds of the interval ([x-h/2, x+h/2]):
    x1_sup = -x1_1_2[:-1]
    x1_inf = -x1_1_2[1:]

    f1 = T * (x1_sup - x1_inf) + a * (sin(d * x1_sup) - sin(d * x1_inf)) \
        + c * (-cos(d * x1_sup) / d + cos(d * x1_inf) / d + T * t * (x1_sup - x1_inf)) \
        - d * (D1[:-1] * cos(d * x1_sup) - D1[1:] * cos(d * x1_inf))

    f1 /= h1

    u0 = np.concatenate(
        (np.diff(-cos(-d * x1_1_2[::-1]) / d - T * t_n * x1_1_2[::-1]),
         np.diff(-ratio_D * cos(d * x2_1_2) / d + T * t_n * x2_1_2))) / h

    u1 = np.concatenate(
        (np.diff(-cos(-d * x1_1_2[::-1]) / d - T * t * x1_1_2[::-1]),
         np.diff(-ratio_D * cos(d * x2_1_2) / d + T * t * x2_1_2))) / h

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1,
                                                                          M2=M2, h1=h1, h2=h2, D1=D1,
                                                                          D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                                                                          neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    #print("erreur sur sinusoide (complete_test_schwarz) :", np.linalg.norm(u1 - u_np1))
    #assert np.linalg.norm(u1 - u_np1) < 9 * 1e-3

    # Schwarz parameters:
    Lambda_1 = 0.5
    Lambda_2 = -0.3

    # random fixed false initialization:
    u_interface = 0.0
    phi_interface = d * D2[0]

    u1_0 = np.flipud(u0[:M1])
    u2_0 = u0[M1:]

    # Beginning of iterations:
    for i in range(15):
        old_u_interface = u_interface
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

        old_interface = u_interface, phi_interface

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
        u_inter1, phi_inter1 = old_interface

    u_np1_schwarz = np.concatenate((u1_ret[::-1], u2_ret))
    print("Schwarz error : ", np.linalg.norm(u_np1_schwarz - u_np1))

    assert np.linalg.norm(u_np1_schwarz - u_np1) < 1e-5
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
    M1, M2 = 340, 110
    h1, h2 = 1 / M1, 1 / M2
    h1 = 1 / M1 + np.zeros(M1)
    h2 = 1 / M2 + np.zeros(M2)
    h1 = np.diff(np.cumsum(np.concatenate(([0], h1)))**3)
    h2 = np.diff(np.cumsum(np.concatenate(([0], h2)))**2)

    # Center of the volumes are x, sizes are h
    x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
    x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
    x1 = np.flipud(x1)

    # coordinates at half-points:
    x1_1_2 = np.cumsum(np.concatenate(([0], h1)))
    x2_1_2 = np.cumsum(np.concatenate(([0], h2)))
    x_1_2 = np.concatenate((np.flipud(x1_1_2[:-1]), x2_1_2))

    x = np.concatenate((x1, x2))

    D1 = 2.2 + x1_1_2**2
    D2 = 1.2 + x2_1_2**2
    D_prime_mean = np.concatenate(
        (np.diff(np.flipud(D1)) / h1, np.diff(D2) / h2))
    D1_prime = 2 * x1_1_2
    D2_prime = 2 * x2_1_2
    D_prime = 2 * x_1_2

    a = 1.3
    c = 0.3

    u_theoric = -pi * (1 - D2[0] / D1[0]) + np.concatenate(
        (pi * x1 * D2[0] / D1[0], pi * x2))
    partial_xu = pi * np.concatenate(
        (np.ones_like(x1) * D2[0] / D1[0], np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f is an average and not a local approximation !
    f = c * u_theoric + (a - D_prime_mean) * partial_xu
    f1 = np.flipud(f[:M1])
    f2 = f[M1:]

    u0 = np.zeros_like(u_theoric)
    u_n, u_interface, phi_interface = integrate_one_step_star(M1=M1,
                                                              M2=M2, h1=h1, h2=h2, D1=D1,
                                                              D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                                                              neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1,
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
    u_interface = 3.0
    phi_interface = 16.0

    # Beginning of iterations:
    for i in range(200):
        old_u_interface = u_interface
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
            u_nm1=u2_n,
            u_interface=u_interface,
            phi_interface=phi_interface,
            upper_domain=True)

        old_interface = u_interface, phi_interface

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
            u_nm1=u1_n,
            u_interface=u_interface,
            phi_interface=phi_interface,
            upper_domain=False)
        u_inter1, phi_inter1 = old_interface
        #print("error:", (u_interface - real_u_interface))
        #print("convergence_rate:", (u_interface - real_u_interface) / (old_u_interface - real_u_interface))
        # input()

    assert Lambda_2 * u_inter1 + phi_inter1 - \
        Lambda_2 * u_interface - phi_interface < 1e-10
    # Note : on le sait, le schéma est mauvais si $h$ est non constant (et encore plus si h discontinu)
    assert abs(u_inter1 - real_u_interface) + abs(phi_inter1 -
                                                  real_phi_interface) < 1
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
    h1, h2 = 1 / M1, 1 / M2
    dt = 0.01
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1 / 2, -h1 / 2, M1)
    x2 = np.linspace(h2 / 2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    #TODO reùettre 2.2, 1.2
    D1 = 2.2
    D2 = 1.2
    a = 1.3
    c = 0.3

    u_theoric = -pi * (1 - D2 / D1) + np.concatenate(
        (pi * x1 * D2 / D1, pi * x2))
    partial_xu = pi * np.concatenate(
        (np.ones_like(x1) * D2 / D1, np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f should be an average and not a local approximation
    f = c * u_theoric + a * partial_xu
    f1 = np.flipud(f[:M1])
    f2 = f[M1:]

    u0 = np.zeros_like(u_theoric)
    u_n, u_interface, phi_interface = integrate_one_step_star(M1=M1,
                                                              M2=M2, h1=h1, h2=h2, D1=D1,
                                                              D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                                                              neumann=neumann, dirichlet=dirichlet, u_nm1=u0)

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1,
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
    u_interface = 3.0
    phi_interface = 16.0

    # Beginning of iterations:
    for i in range(100):
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
            u_nm1=u2_n,
            u_interface=u_interface,
            phi_interface=phi_interface,
            upper_domain=True)

        old_interface = u_interface, phi_interface

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
            u_nm1=u1_n,
            u_interface=u_interface,
            phi_interface=phi_interface,
            upper_domain=False)
        u_inter1, phi_inter1 = old_interface

        assert Lambda_1 * u_inter1 + phi_inter1 - \
            Lambda_1 * u_interface - phi_interface < 1e-15
    assert abs(u_inter1 - real_u_interface) + abs(phi_inter1 -
                                                  real_phi_interface) < 1e-2
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
    M1, M2 = 40, 40
    h1, h2 = 1 / M1, 1 / M2
    dt = 1
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1 / 2, -h1 / 2, M1)
    x2 = np.linspace(h2 / 2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = .1

    u_theoric = -pi * (1 - D2 / D1) + np.concatenate(
        (pi * x1 * D2 / D1, pi * x2))
    partial_xu = pi * np.concatenate(
        (np.ones_like(x1) * D2 / D1, np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    f = c * u_theoric + a * partial_xu
    # Note: f should be an average and not a local approximation
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u = np.zeros_like(u_theoric)
    for i in range(30):
        u = integrate_one_step_star(M1=M1,
                                    M2=M2,
                                    h1=h1,
                                    h2=h2,
                                    D1=D1,
                                    D2=D2,
                                    a=a,
                                    c=c,
                                    dt=dt,
                                    f1=f1,
                                    f2=f2,
                                    neumann=neumann,
                                    dirichlet=dirichlet,
                                    u_nm1=u)[0]

    u1 = u[:M1]
    u2 = u[M1 + 1:]

    print(np.linalg.norm(u_theoric - u))
    assert abs(-pi * (1 - D2 / D1) + pi * (1 - h2 / 2) - u[-1]) < 1e-11
    assert np.linalg.norm(u_theoric - u) < 1e-11
    assert np.linalg.norm(np.diff(u1) - pi * h1 * D2 / D1) < 1e-12
    assert np.linalg.norm(np.diff(u2) - pi * h2) < 1e-12
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
    h1, h2 = 1 / M1, 1 / M2
    dt = 10000000
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1 / 2, -h1 / 2, M1)
    x2 = np.linspace(h2 / 2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = 0.4

    u_theoric = -pi * (1 - D2 / D1) + np.concatenate(
        (pi * x1 * D2 / D1, pi * x2))
    partial_xu = pi * np.concatenate(
        (np.ones_like(x1) * D2 / D1, np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f should be an average and not a local approximation
    f = c * u_theoric + a * partial_xu
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u0 = np.zeros_like(u_theoric)
    u = integrate_one_step_star(M1=M1,
                                M2=M2,
                                h1=h1,
                                h2=h2,
                                D1=D1,
                                D2=D2,
                                a=a,
                                c=c,
                                dt=dt,
                                f1=f1,
                                f2=f2,
                                neumann=neumann,
                                dirichlet=dirichlet,
                                u_nm1=u0)[0]

    u1 = u[:M1]
    u2 = u[M1 + 1:]

    assert abs(-pi * (1 - D2 / D1) + pi * (1 - h2 / 2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi * h1 * D2 / D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi * h2) < 1e-6
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
    M1, M2 = 10, 10
    h1, h2 = 1 / M1, 1 / M2
    dt = 1
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1 / 2, -h1 / 2, M1)
    x2 = np.linspace(h2 / 2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.4
    D2 = 1.4
    a = 0.0
    c = 0.0

    u_theoric = -pi * (1 - D2 / D1) + np.concatenate(
        (pi * x1 * D2 / D1, pi * x2))
    neumann = pi
    dirichlet = -pi
    f = np.zeros_like(x)
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u0 = u_theoric#np.zeros_like(x)
    u = integrate_one_step_star(M1=M1,
                                M2=M2,
                                h1=h1,
                                h2=h2,
                                D1=D1,
                                D2=D2,
                                a=a,
                                c=c,
                                dt=dt,
                                f1=f1,
                                f2=f2,
                                neumann=neumann,
                                dirichlet=dirichlet,
                                u_nm1=u0)[0]

    u1 = u[:M1]
    u2 = u[M1 + 1:]

    assert abs(-pi * (1 - D2 / D1) + pi * (1 - h2 / 2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi * h1 * D2 / D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi * h2) < 1e-6
    return "ok"

def order_error_term_linear_time_domain2():
    """
    Simple Case : a=c=0
    on se place en (-1, 0)
    Dirichlet en -1
    Notre fonction est sin(x) + t
    Donc le schéma en temps résout exactement sa partie,
    l'erreur en h**2 vient seulement de l'espace :)
    On compense l'erreur au bord en h**2 par une condition
    de "Dirichlet" : u + h**2/12*u''
    It is an order 2 method but the solution in time is linear:
    the only error can come frombd conditions. for Neumann bd there is no error
    for Dirichlet bd, it is order 3
    """
    T = 40.
    Lambda = -1.
    Courant = 10.
    D = 3.
    finite_volumes.LAMBDA_2 = Lambda
    finite_volumes.COURANT_NUMBER = Courant
    finite_volumes.D1 = D
    finite_volumes.D2 = D
    finite_volumes.A = 0.
    finite_volumes.C = 0.
    finite_volumes.SIZE_DOMAIN_1 = 1.
    finite_volumes.SIZE_DOMAIN_2 = 1.

    # on peut affirmer la chose suivante : l'erreur est en h**2, et pas en dt.
    ret = []
    for M in (16, 32):
        dt = 1/M**2*Courant/D # en essayant avec dt = Courant/M/D, only 10 steps to go faast
        finite_volumes.DT = dt

        print("M:", M, "dt:", dt)
        N = int(T/dt)

        M1, M2 = M,M
        finite_volumes.M1 = M
        finite_volumes.M2 = M
        h1 = 1 / M1 + np.zeros(M1)
        h2 = 1 / M2 + np.zeros(M2)
        h = h2[0]
        
        t_initial = 2.
        t_final = t_initial + T

        D1, D2 = D, D

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x1 = np.flipud(x1)
        x = np.concatenate((-x1, x2))

        def u_real(x, t):
            return np.sin(x) + t

        def u_prime(x, t):
            return np.cos(x)

        def u_bar(x_1_2, h, t):
            return 1/h * (np.cos(x_1_2) - np.cos(x_1_2+h)) + t

        def flux(x, t):
            return D * np.cos(x)

        def f_bar(x, h, t): # time derivative is... 1
            return 1 - 1/h * D * (np.cos(x+h) - np.cos(x))

        def u_seconde_space(x, t):
            return -np.sin(x)

        x_flux = np.concatenate((x-h/2,[1]))

        u2_0 = u_bar(x2 - h2/2, h2, t_initial)
        phi2_0 = flux(x_flux[M1:], t_initial)
        additional = [u2_0]

        from progressbar import ProgressBar
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):
            t_np1 = t_n + dt

            neumann = flux(1, t_np1)

            phi_int = flux(0, t_n + dt)
            phi_int_nm1_2 = flux(0, t_n + dt/2)
            phi_int_nm1 = flux(0, t_n)

            u_int = u_real(0, t_n + dt) #+ h**2/12 * u_seconde_space(0, t_np1)
            u_int_nm1_2 = u_real(0, t_n + dt/2) #+ h**2/12 * u_seconde_space(0, t_n+dt/2)
            u_int_nm1 = u_real(0, t_n) #+ h**2/12 * u_seconde_space(0, t_n)

            f2 = (f_bar(x_flux[M1:-1], h, t_np1))
            f2_nm1_2 = (f_bar(x_flux[M1:-1], h, t_n + dt/2))
            f2_nm1 = (f_bar(x_flux[M1:-1], h, t_n))
            f2 = np.concatenate(([f2[0]], np.diff(f2), [f2[-1]]))

            phi_np1, real_u_interface, real_phi_interface, *additional = integrate_one_step(f=f2,
                                                                             f_nm1_2=f2_nm1_2,
                                                                             f_nm1=f2_nm1,
                                                                             bd_cond=neumann,
                                                                             bd_cond_nm1_2=neumann,
                                                                             bd_cond_nm1=neumann,
                                                                             Lambda=Lambda,
                                                                             u_nm1=phi2_0,
                                                                             u_interface=u_int,
                                                                             u_nm1_2_interface=u_int_nm1_2,
                                                                             u_nm1_interface=u_int_nm1,
                                                                             phi_interface=phi_int,
                                                                             phi_nm1_2_interface=phi_int_nm1_2,
                                                                             phi_nm1_interface=phi_int_nm1,
                                                                             additional=additional,
                                                                             upper_domain=True)
            phi2_0 = phi_np1
            u2_0 = additional[0]

            nb_plots = 4
            if int(N * (t_n - t_initial) / T) % int(N/nb_plots) == 0:
                import matplotlib.pyplot as plt
                plt.plot(x2, u2_0, "b")
                plt.plot(x2, u_bar(x2-h2/2, h2, t_n+dt), "r")
                # plt.plot(x_flux[M1:], phi2_0, "b", label="approximation")
                # plt.plot(x_flux[M1:], flux(x_flux[M1:], t_initial), "r", label="solution")
                plt.show()
                print("enter to continue, or ctrl-C to stop")
                input()

        ret += [np.linalg.norm(u_bar(x2-h2/2, h2, t_n+dt) - u2_0)/np.linalg.norm(u_bar(x2-h2/2, h2, t_n+dt))]
    print(ret)
    print(np.log(ret[0]/ret[1])/np.log(2))
    assert abs(3 - np.log(ret[0]/ret[1])/np.log(2)) < .1
    return "ok"

def order_error_term_linear_time():
    """
    Simple Case : a=c=0
    on se place en (-1, 0)
    Dirichlet en -1
    Notre fonction est sin(x) + t
    Donc le schéma en temps résout exactement sa partie,
    l'erreur en h**2 vient seulement de l'espace :)
    On compense l'erreur au bord en h**2 par une condition
    de "Dirichlet" : u + h**2/12*u''
    It is an order 2 method but the solution in time is linear:
    the only error can come frombd conditions. for Neumann bd there is no error
    for Dirichlet bd, it is order 3
    """
    T = 40.
    Lambda = 1.
    Courant = 10.
    D = 10.
    finite_volumes.LAMBDA_1 = Lambda
    finite_volumes.D1 = D
    finite_volumes.D2 = D
    finite_volumes.A = 0.
    finite_volumes.C = 0.
    finite_volumes.SIZE_DOMAIN_1 = 1.
    finite_volumes.SIZE_DOMAIN_2 = 1.

    # on peut affirmer la chose suivante : l'erreur est en h**2, et pas en dt.
    ret = []
    for M in (8, 16):
        dt = 1/M**2*Courant/D # en essayant avec dt = Courant/M/D, only 10 steps to go faast
        finite_volumes.DT = dt

        print("M:", M, "dt:", dt)
        N = int(T/dt)

        M1, M2 = M,M
        finite_volumes.M1 = M
        finite_volumes.M2 = M
        h1 = 1 / M1 + np.zeros(M1)
        h2 = 1 / M2 + np.zeros(M2)
        h = h1[0]
        
        t_initial = 2.
        t_final = t_initial + T

        D1, D2 = D, D

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x1 = np.flipud(x1)
        x = np.concatenate((-x1, x2))

        def u_real(x, t):
            return np.sin(x) + t

        def u_bar(x_1_2, h, t):
            return 1/h * (np.cos(x_1_2) - np.cos(x_1_2+h)) + t

        def flux(x, t):
            return D * np.cos(x)

        def f_bar(x, h, t): # time derivative is... 1
            return 1 - 1/h * D * (np.cos(x+h) - np.cos(x))

        def u_seconde_space(x, t):
            return -np.sin(x)

        x_flux = np.concatenate((x-h/2,[1]))

        u1_0 = np.flipud(u_bar(-x1 - h1/2, h1, t_initial))
        phi1_0 = np.flipud(flux(x_flux[:M1+1], t_initial))
        additional = [u1_0]

        from progressbar import ProgressBar
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):
            t_np1 = t_n + dt

            dirichlet = u_real(-1, t_np1) #+ h**2/12 * u_seconde_space(-1, t_np1)
            dirichlet_nm1_2 = u_real(-1, t_n + dt/2)# + h**2/12 * u_seconde_space(-1, t_n+dt/2)
            dirichlet_nm1 = u_real(-1, t_n) #+ h**2/12 * u_seconde_space(-1, t_n)

            phi_int = flux(0, t_n + dt)
            phi_int_nm1_2 = flux(0, t_n + dt/2)
            phi_int_nm1 = flux(0, t_n)

            u_int = u_real(0, t_n + dt) #+ h**2/12 * u_seconde_space(0, t_np1)
            u_int_nm1_2 = u_real(0, t_n + dt/2) #+ h**2/12 * u_seconde_space(0, t_n+dt/2)
            u_int_nm1 = u_real(0, t_n) #+ h**2/12 * u_seconde_space(0, t_n)

            f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
            f1_nm1_2 = np.flipud(f_bar(x_flux[:M1], h, t_n + dt/2))
            f1_nm1 = np.flipud(f_bar(x_flux[:M1], h, t_n))
            f1 = np.concatenate(([f1[0]], np.diff(f1), [f1[-1]]))

            phi_np1, real_u_interface, real_phi_interface, *additional = integrate_one_step(f=f1,
                                                                             f_nm1_2=f1_nm1_2,
                                                                             f_nm1=f1_nm1,
                                                                             bd_cond=dirichlet,
                                                                             bd_cond_nm1_2=dirichlet_nm1_2,
                                                                             bd_cond_nm1=dirichlet_nm1,
                                                                             Lambda=Lambda,
                                                                             u_nm1=phi1_0,
                                                                             u_interface=u_int,
                                                                             u_nm1_2_interface=u_int_nm1_2,
                                                                             u_nm1_interface=u_int_nm1,
                                                                             phi_interface=phi_int,
                                                                             phi_nm1_2_interface=phi_int_nm1_2,
                                                                             phi_nm1_interface=phi_int_nm1,
                                                                             additional=additional,
                                                                             upper_domain=False)
            phi1_0 = phi_np1
            u1_0 = additional[0]

            # nb_plots = 4
            # if int(N * (t_n - t_initial) / T) % int(N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #     #plt.plot(-x1, np.flipud(u1_0), "b")
            #     #plt.plot(-x1, u_bar(-x1-h1/2, h1, t_n+dt), "r")
            #     plt.plot(x_flux[:M1+1], np.flipud(phi1_0), "b", label="approximation")
            #     plt.plot(x_flux[:M1+1], flux(x_flux[:M1+1], t_initial), "r", label="solution")
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

        ret += [np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt))]
    print(np.log(ret[0]/ret[1])/np.log(2))
    assert abs(3 - np.log(ret[0]/ret[1])/np.log(2)) < .1
    return "ok"

