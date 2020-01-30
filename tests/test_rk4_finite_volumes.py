"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi
from numpy.random import random
from tests.utils_numeric import solve_linear
from discretizations.rk4_finite_volumes import Rk4FiniteVolumes
from progressbar import ProgressBar
finite_volumes = Rk4FiniteVolumes()
get_Y_star = finite_volumes.get_Y_star
integrate_one_step = finite_volumes.integrate_one_step
integrate_one_step_star = finite_volumes.integrate_one_step_star
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

    #assert "ok" == first_correctness_test1()
    #assert "ok" == first_correctness_test2()
    #assert "ok" == order_error_term()
    assert "ok" == order_error_term_linear_time()
    return "ok"


def first_correctness_test1():
    """
    Simple Case : a=c=0
    on se place en (-1, 0)
    Dirichlet en -1
    """
    T = 3.
    Lambda = -1.
    Courant = .1
    dt = .001
    N = int(T/dt)
    D = 5.

    M = int(np.sqrt(Courant/(D*dt)))
    print("M:", M)

    M1, M2 = M,M
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

    
    u0 = u_bar(x - h/2, h, t_initial)
    u1_0 = np.flipud(u_bar(-x1 - h1/2, h1, t_initial))
        

    for t_n in np.linspace(t_initial, t_final, N, endpoint=False):
        t_np1 = t_n + dt

        dirichlet = u_real(-1, t_np1)
        dirichlet_nm1_2 = u_real(-1, t_n + dt/2)
        dirichlet_nm1 = u_real(-1, t_n)

        phi_int = flux(0, t_n + dt)
        phi_int_nm1_2 = flux(0, t_n + dt/2)
        phi_int_nm1 = flux(0, t_n)

        u_int = u_real(0, t_n + dt)
        u_int_nm1_2 = u_real(0, t_n + dt/2)
        u_int_nm1 = u_real(0, t_n)

        f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
        f1_nm1_2 = np.flipud(f_bar(x_flux[:M1], h, t_n + dt/2))
        f1_nm1 = np.flipud(f_bar(x_flux[:M1], h, t_n))

        u_np1, real_u_interface, real_phi_interface = integrate_one_step(M=M1,
                                                                         h=h1,
                                                                         D=D1,
                                                                         a=0., c=0., dt=dt,
                                                                         f=f1,
                                                                         f_nm1_2=f1_nm1_2,
                                                                         f_nm1=f1_nm1,
                                                                         bd_cond=dirichlet,
                                                                         bd_cond_nm1_2=dirichlet_nm1_2,
                                                                         bd_cond_nm1=dirichlet_nm1,
                                                                         Lambda=Lambda,
                                                                         u_nm1=u1_0,
                                                                         u_interface=u_int,
                                                                         u_nm1_2_interface=u_int_nm1_2,
                                                                         u_nm1_interface=u_int_nm1,
                                                                         phi_interface=phi_int,
                                                                         phi_nm1_2_interface=phi_int_nm1_2,
                                                                         phi_nm1_interface=phi_int_nm1,
                                                                         upper_domain=False)
        u1_0 = u_np1
        """
        nb_plots = 4
        if int(N * (t_n - t_initial) / T) % int(N/nb_plots) == 0:
            import matplotlib.pyplot as plt
            plt.plot(-x1, np.flipud(u1_0), "b")
            plt.plot(-x1, u_bar(-x1-h1/2, h1, t_n+dt), "r")
            plt.show()
        """

    print("erreur domaine 1:", np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt)))
    return "ok"

def first_correctness_test2():
    """
    Simple Case : a=c=0
    on se place en (0, 1)
    Neumann en 1, Dirichlet en -1
    """
    T = .5
    Lambda = -1
    Courant = .1
    dt = .001
    N = int(T/dt)
    D = 5.

    M = int(np.sqrt(Courant/(D*dt)))
    print("M:", M)

    M1, M2 = M,M
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

    
    u0 = u_bar(x - h/2, h, t_initial)
    u2_0 = u_bar(x2 - h2/2, h2, t_initial)
        

    import matplotlib.pyplot as plt
    for t_n in np.linspace(t_initial, t_final, N, endpoint=False):
        t_np1 = t_n + dt

        dirichlet = u_real(-1, t_np1)

        neumann = flux(1, t_n + dt)/D # D\partial_x u = flux
        neumann_nm1_2 = flux(1, t_n + dt/2)/D # D\partial_x u = flux
        neumann_nm1 = flux(1, t_n)/D # D\partial_x u = flux

        phi_int = flux(0, t_n + dt)
        phi_int_nm1_2 = flux(0, t_n + dt/2)
        phi_int_nm1 = flux(0, t_n)

        u_int = u_real(0, t_n + dt)
        u_int_nm1_2 = u_real(0, t_n + dt/2)
        u_int_nm1 = u_real(0, t_n)

        f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
        f2 = f_bar(x_flux[-M2-1:-1], h, t_n + dt)
        f2_nm1_2 = f_bar(x_flux[-M2-1:-1], h, t_n + dt/2)
        f2_nm1 = f_bar(x_flux[-M2-1:-1], h, t_n)

        u_np1, real_u_interface, real_phi_interface = integrate_one_step(M=M2,
                                                                         h=h2,
                                                                         D=D2,
                                                                         a=0., c=0., dt=dt,
                                                                         f=f2,
                                                                         f_nm1_2=f2_nm1_2,
                                                                         f_nm1=f2_nm1,
                                                                         bd_cond=neumann,
                                                                         bd_cond_nm1_2=neumann_nm1_2,
                                                                         bd_cond_nm1=neumann_nm1,
                                                                         Lambda=Lambda,
                                                                         u_nm1=u2_0,
                                                                         u_interface=u_int,
                                                                         u_nm1_2_interface=u_int_nm1_2,
                                                                         u_nm1_interface=u_int_nm1,
                                                                         phi_interface=phi_int,
                                                                         phi_nm1_2_interface=phi_int_nm1_2,
                                                                         phi_nm1_interface=phi_int_nm1,
                                                                         upper_domain=True)
        u2_0 = u_np1
        """
        nb_plots = 4
        if int(N * (t_n - t_initial) / T) % int(N/nb_plots) == 0:
            import matplotlib.pyplot as plt
            plt.plot(x2, u2_0, "b")
            plt.plot(x2, u_bar(x2-h2/2, h2, t_n+dt), "r")
            #plt.plot(x, u1, "k--")
            plt.show()
        """
    print("erreur domaine 2 :", np.linalg.norm(u2_0 - u_bar(x2-h2/2, h2, t_n+dt))/np.linalg.norm(u2_0))

    return "ok"


def order_error_term():
    """
    Simple Case : a=c=0
    on se place en (-1, 0)
    Dirichlet en -1
    """
    T = 8.
    Lambda = -1 # does not work for lambda<0 ? .-.
    Courant = .02
    D = 5

    M = 4
    dt = Courant/M**2/D # en essayant avec dt = Courant/M/D,
    # on peut affirmer la chose suivante : l'erreur est en h**2, et pas en dt.
    N = int(T/dt)
    print("M:", M, "dt:", dt)

    M1, M2 = M,M
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
    def derivee_x2(x, t):
        return -np.cos(x+t)

    def f_bar(x, h, t):
        return 1/h*(np.sin(x+t+h) - np.sin(x+t) - D * (np.cos(x+t+h) - np.cos(x+t)))

    x_flux = np.concatenate((x-h/2,[1]))

    
    u0 = u_bar(x - h/2, h, t_initial)
    u1_0 = np.flipud(u_bar(-x1 - h1/2, h1, t_initial))
        
    progress = ProgressBar()
    for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):
        t_np1 = t_n + dt

        dirichlet = u_real(-1, t_np1)
        dirichlet_nm1_2 = u_real(-1, t_n + dt/2)
        dirichlet_nm1 = u_real(-1, t_n)

        phi_int = flux(0, t_n + dt)
        phi_int_nm1_2 = flux(0, t_n + dt/2)
        phi_int_nm1 = flux(0, t_n)

        u_int = u_real(0, t_n + dt)
        u_int_nm1_2 = u_real(0, t_n + dt/2)
        u_int_nm1 = u_real(0, t_n)

        f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
        f1_nm1_2 = np.flipud(f_bar(x_flux[:M1], h, t_n + dt/2))
        f1_nm1 = np.flipud(f_bar(x_flux[:M1], h, t_n))

        u_np1, real_u_interface, real_phi_interface = integrate_one_step(M=M1,
                                                                         h=h1,
                                                                         D=D1,
                                                                         a=0., c=0., dt=dt,
                                                                         f=f1,
                                                                         f_nm1_2=f1_nm1_2,
                                                                         f_nm1=f1_nm1,
                                                                         bd_cond=dirichlet,
                                                                         bd_cond_nm1_2=dirichlet_nm1_2,
                                                                         bd_cond_nm1=dirichlet_nm1,
                                                                         Lambda=Lambda,
                                                                         u_nm1=u1_0,
                                                                         u_interface=u_int,
                                                                         u_nm1_2_interface=u_int_nm1_2,
                                                                         u_nm1_interface=u_int_nm1,
                                                                         phi_interface=phi_int,
                                                                         phi_nm1_2_interface=phi_int_nm1_2,
                                                                         phi_nm1_interface=phi_int_nm1,
                                                                         upper_domain=False)
        u1_0 = u_np1
        """
        nb_plots = 4
        if int(N * (t_n - t_initial) / T) % int(N/nb_plots) == 0:
            import matplotlib.pyplot as plt
            plt.plot(-x1, np.flipud(u1_0), "b")
            plt.plot(-x1, u_bar(-x1-h1/2, h1, t_n+dt), "r")
            plt.show()
        """

    print("erreur domaine 1:", np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt)))
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
    """
    T = 8.
    Lambda = -1
    Courant = .2
    D = 5

    # on peut affirmer la chose suivante : l'erreur est en h**2, et pas en dt.
    ret = []
    for M in (8, 16):
        dt = 1/M**2*Courant/D # en essayant avec dt = Courant/M/D, only 10 steps to go faast
        print("M:", M, "dt:", dt)
        N = int(T/dt)

        M1, M2 = M,M
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
        phi1_0 = [np.flipud(flux(x_flux[:M1+1], t_initial))]
            
        from progressbar import ProgressBar
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):
            t_np1 = t_n + dt

            dirichlet = u_real(-1, t_np1) + h**2/12 * u_seconde_space(-1, t_np1)
            dirichlet_nm1_2 = u_real(-1, t_n + dt/2) + h**2/12 * u_seconde_space(-1, t_n+dt/2)
            dirichlet_nm1 = u_real(-1, t_n) + h**2/12 * u_seconde_space(-1, t_n)

            phi_int = flux(0, t_n + dt)
            phi_int_nm1_2 = flux(0, t_n + dt/2)
            phi_int_nm1 = flux(0, t_n)

            u_int = u_real(0, t_n + dt) + h**2/12 * u_seconde_space(0, t_np1)
            u_int_nm1_2 = u_real(0, t_n + dt/2) + h**2/12 * u_seconde_space(0, t_n+dt/2)
            u_int_nm1 = u_real(0, t_n) + h**2/12 * u_seconde_space(0, t_n)

            f1 = np.flipud(f_bar(x_flux[:M1], h, t_np1))
            f1_nm1_2 = np.flipud(f_bar(x_flux[:M1], h, t_n + dt/2))
            f1_nm1 = np.flipud(f_bar(x_flux[:M1], h, t_n))

            u_np1, real_u_interface, real_phi_interface, *phi_np1 = integrate_one_step(M=M1,
                                                                             h=h1,
                                                                             D=D1,
                                                                             a=0., c=0., dt=dt,
                                                                             f=f1,
                                                                             f_nm1_2=f1_nm1_2,
                                                                             f_nm1=f1_nm1,
                                                                             bd_cond=dirichlet,
                                                                             bd_cond_nm1_2=dirichlet_nm1_2,
                                                                             bd_cond_nm1=dirichlet_nm1,
                                                                             Lambda=Lambda,
                                                                             u_nm1=u1_0,
                                                                             u_interface=u_int,
                                                                             u_nm1_2_interface=u_int_nm1_2,
                                                                             u_nm1_interface=u_int_nm1,
                                                                             phi_interface=phi_int,
                                                                             phi_nm1_2_interface=phi_int_nm1_2,
                                                                             phi_nm1_interface=phi_int_nm1,
                                                                             phi_for_FV=phi1_0,
                                                                             upper_domain=False)
            phi1_0 = phi_np1
            u1_0 = u_np1
            """
            nb_plots = 4
            if int(N * (t_n - t_initial) / T) % int(N/nb_plots) == 0:
                import matplotlib.pyplot as plt
                plt.plot(-x1, np.flipud(u1_0), "b")
                plt.plot(-x1, u_bar(-x1-h1/2, h1, t_n+dt), "r")
                plt.show()
            """

        ret += [np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(-x1-h1/2, h1, t_n+dt))]
    print(ret)
    assert abs(4 - np.log(ret[0]/ret[1])/np.log(2)) < .1
    return "ok"

