"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi
from numpy.random import random
from tests.utils_numeric import solve_linear
"""
    Test function of the module.
    Tests the finite volumes space scheme, with all the time schemes.
"""


def launch_all_tests():
    print("Test of Finite volumes schemes.")

    print("Test integration with finite volumes, spline 2:", test_integrate_one_step())


"""
    Silent test function of the module.
    Tests the integration functions of finite_volumes module.
"""


def test_integrate_one_step():
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.Manfredi import Manfredi
    u_ubar_flux_fbar = intricated_spacetime
    test_all_time_schemes_domain1(BackwardEuler, u_ubar_flux_fbar)
    test_all_time_schemes_domain1(Manfredi, u_ubar_flux_fbar)
    test_all_time_schemes_domain1(ThetaMethod, u_ubar_flux_fbar)

    # test_all_time_schemes_domain1(RK2, u_ubar_flux_fbar)
    # test_all_time_schemes_domain1(RK4, u_ubar_flux_fbar)

    return "ok"

def linear_time(a, c, D):
    def u_real(x, t): return np.sin(x) + t
    def u_bar(x_1_2, h, t): return 1/h * (np.cos(x_1_2) - np.cos(x_1_2+h)) + t
    def flux(x, t): return D * np.cos(x)
    def f_bar(x, h, t): return 1 - 1/h * D * (np.cos(x+h) - np.cos(x))
    return u_real, u_bar, flux, f_bar

def cosinus_time(a, c, D):
    def u_real(x, t): return np.sin(x) + np.cos(t)
    def u_bar(x_1_2, h, t): return 1/h * (np.cos(x_1_2) - np.cos(x_1_2+h)) + np.cos(t)
    def flux(x, t): return D * np.cos(x)
    def f_bar(x, h, t): return -np.sin(t) - 1/h * D * (np.cos(x+h) - np.cos(x))
    return u_real, u_bar, flux, f_bar

def intricated_spacetime(a, c, D):
    def u_real(x, t): return np.sin(x + t)
    def u_bar(x_1_2, h, t): return 1/h * (np.cos(x_1_2 + t) - np.cos(x_1_2+h + t))
    def flux(x, t): return D * np.cos(x + t)
    def f_bar(x, h, t): return 1/h*(np.sin(x+h+t) - np.sin(x+t)) - 1/h * D * (np.cos(x+h+t) - np.cos(x+t))
    return u_real, u_bar, flux, f_bar

def exp_spacetime(a, c, D):
    def u_real(x, t): return np.exp(x + t)
    def u_bar(x_1_2, h, t): return 1/h * (np.exp(x_1_2+h + t) - np.exp(x_1_2 + t))
    def flux(x, t): return D * np.exp(x + t)
    def f_bar(x, h, t): return 1/h*(np.exp(x+h+t) - np.exp(x+t)) - 1/h * D * (np.exp(x+h+t) - np.exp(x+t))
    return u_real, u_bar, flux, f_bar

def exp_space_quad_time(a, c, D):
    def u_real(x, t): return np.exp(x) * t**2
    def u_bar(x_1_2, h, t): return t**2/h * (np.exp(x_1_2+h) - np.exp(x_1_2))
    def flux(x, t): return D * np.exp(x) * t**2
    def f_bar(x, h, t): return 2*t/h*(np.exp(x+h) - np.exp(x)) - 1/h * D * (np.exp(x+h) - np.exp(x))*t**2
    return u_real, u_bar, flux, f_bar

def exp_space_cubic_time(a, c, D):
    def u_real(x, t): return np.exp(x) * t**3
    def u_bar(x_1_2, h, t): return t**3/h * (np.exp(x_1_2+h) - np.exp(x_1_2))
    def flux(x, t): return D * np.exp(x) * t**3
    def f_bar(x, h, t): return 3*t**2/h*(np.exp(x+h) - np.exp(x)) - 1/h * D * (np.exp(x+h) - np.exp(x))*t**3
    return u_real, u_bar, flux, f_bar


# This is not the same function as test_finite_differences.test_all_time_schemes_domain1
# because the right hand side is not the same, the comparison is not the same,
# we integrate in time the fluxes and not u
def test_all_time_schemes_domain1(time_scheme, u_ubar_flux_fbar=linear_time):
    """
    Simplest Case : a=c=0
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
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()
    from discretizations.space.quad_splines_fv import QuadSplinesFV as space_scheme

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 4.
    Lambda = 1.
    Courant = 10. # RK2 experimental CFL condition
    D = 1.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.C = 0.
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_ubar_flux_fbar = u_ubar_flux_fbar(builder.A, builder.C, D)

    scheme = builder.build(time_scheme, space_scheme)

    ret = []
    # Loop to compare different settings:
    for M in (8, 16):
        dt = 1/M**2*Courant/D
        M=4096

        scheme.DT = dt

        N = int(T/dt)
        T = N*dt

        scheme.M1 = M
        scheme.M2 = M
        h1 = 1 / M + np.zeros(M)
        h = h1[0]
        
        t_initial = 2.
        t_final = t_initial + T

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x1 = np.flipud(-x1)
        x_flux = np.concatenate((x1-h/2,[0]))

        ######################
        # END OF THE SETTING DESCRIPTION.
        ######################

        ######################
        # EQUATION DEFINITION: WHICH U ? WHICH F ?
        ######################
        u_real, u_bar, flux, f_bar = u_ubar_flux_fbar

        ###########################
        # TIME INTEGRATION:
        ############################

        # The following loop is valid independently of the u chosen.
        u1_0 = np.flipud(u_bar(x1 - h1/2, h1, t_initial))
        phi1_0 = np.flipud(flux(x_flux, t_initial))
        additional = [u1_0]
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(-1, t_n + dt*time)

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f1(time):
                ret = np.flipud(f_bar(x_flux[:-1], h, t_n + time*dt))
                return np.concatenate(([ret[0]], np.diff(ret), [ret[-1]]))

            phi_np1, real_u_interface, real_phi_interface, *additional = scheme.integrate_one_step(f=f1,
                                                                             bd_cond=dirichlet,
                                                                             u_nm1=phi1_0,
                                                                             u_interface=u_int,
                                                                             phi_interface=phi_int,
                                                                             additional=additional,
                                                                             upper_domain=False)
            phi1_0 = phi_np1
            u1_0 = additional[0]

            # nb_plots = 4
            # if int(4*N * (t_n - t_initial) / T) % int(4*N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #     plt.plot(x1, np.flipud(u1_0), "b")
            #     plt.plot(x1, u_bar(x1-h1/2, h1, t_n+dt), "r")
            #     # plt.plot(x_flux, np.flipud(phi1_0), "b", label="approximation")
            #     # plt.plot(x_flux, flux(x_flux, t_initial), "r--", label="solution")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

        ret += [np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt))]
        print("errors: ", ret)
    print("Order in time (careful: Courant constant) for", time_scheme.__name__, np.log(ret[0]/ret[1])/np.log(2)/2)
    if abs(3 - np.log(ret[0]/ret[1])/np.log(2)) < .1:
        return "ok"
    else:
        return "Order : " + str(np.log(ret[0]/ret[1])/np.log(2))


