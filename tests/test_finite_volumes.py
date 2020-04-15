"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi
from numpy.random import random
from tests.test_cases import intricated_spacetime, linear_time, cosinus_time, exp_space_quad_time
"""
    Test function of the module.
    Tests the finite volumes space scheme, with all the time schemes.
"""


def launch_all_tests():
    print("Test integration with finite volumes, 4rth order", test_integrate_one_step())


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
    def verify_order(time_scheme, function, order, **kwargs): # /!\ give order in space !
        try:
            experimental_order = test_any_time_scheme_domain1(time_scheme, function,  **kwargs)
            if abs(experimental_order - order) < .4:
                return
            else:
                print("Error for ", time_scheme.__name__, function.__name__,
                        ": should be space order", order, "but is:", experimental_order)
        except(KeyboardInterrupt):
            print("Interrupted test", time_scheme.__name__, function.__name__, "by user interruption.")

    print("MANFREDI SCHEME:")
    # Manfredi is a particular scheme. It makes an order 1 error when using a rhs
    verify_order(Manfredi, intricated_spacetime, 2)
    verify_order(Manfredi, linear_time, 4) # It should be 4 but it is 2 because of bad reaction handling
    verify_order(Manfredi, cosinus_time, 2)
    verify_order(Manfredi, exp_space_quad_time, 2)

    print("BACKWARD EULER SCHEME:")
    # Backward Euler is order 1 in time, so 2 in space (constant Courant parabolic number)
    verify_order(BackwardEuler, intricated_spacetime, 2)
    verify_order(BackwardEuler, linear_time, 4) # exact in time so the only error is space
    verify_order(BackwardEuler, cosinus_time, 2)
    verify_order(BackwardEuler, exp_space_quad_time, 2)

    # the 2nd order time schemes gives 4rth order in space error.

    print("THETA SCHEME:")
    verify_order(ThetaMethod, intricated_spacetime, 4)
    verify_order(ThetaMethod, linear_time, 4)
    verify_order(ThetaMethod, cosinus_time, 4)
    verify_order(ThetaMethod, exp_space_quad_time, 4)

    print("RK2 SCHEME:")
    verify_order(RK2, intricated_spacetime, 4)
    verify_order(RK2, linear_time, 4)
    verify_order(RK2, cosinus_time, 4)
    verify_order(RK2, exp_space_quad_time, 4)

    # 4th orders time scheme are limited by space error:
    print("RK4 SCHEME:")
    verify_order(RK4, intricated_spacetime, 4)
    verify_order(RK4, linear_time, 4)
    verify_order(RK4, cosinus_time, 4)
    verify_order(RK4, exp_space_quad_time, 4)

    return "ok"


# This is not the same function as test_finite_differences.test_all_time_schemes_domain1
# because the right hand side is not the same, the comparison is not the same,
# we integrate in time the fluxes and not u
def test_any_time_scheme_domain1(time_scheme, u_ubar_flux_fbar=linear_time):
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

    returns the order in space (i.e. order in time / 2 because courant parabolic is fixed)
    """
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()
    from discretizations.space.fourth_order_fv import FourthOrderFV as space_scheme

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 4.
    Lambda = 1.
    Courant=.1
    D = 2.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.C = .0
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_ubar_flux_fbar = u_ubar_flux_fbar(builder.A, builder.C, D)

    scheme = builder.build(time_scheme, space_scheme)

    ret = []
    # Loop to compare different settings:
    for M in (8, 16):
        dt = 1/M**2*Courant/D

        scheme.DT = dt

        N = int(T/dt)

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
        keys = ('u', 'u^(2)', 'Bar{u}', 'Ddu/dx', 'Bar{f}', 'Bar{f}^(2)')
        u_real, u_sec, u_bar, flux, f_bar, f_bar_sec = (u_ubar_flux_fbar[i] for i in keys)

        ###########################
        # TIME INTEGRATION:
        ############################

        n=0
        # The following loop is valid independently of the u chosen.
        u1_0 = np.flipud(u_bar(x1 - h1/2, h1, t_initial))
        phi1_0 = np.flipud(flux(x_flux, t_initial))
        additional = [u1_0]
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(-1, t_n + dt*time) + h**2/12 * u_sec(-1, t_n + dt*time)

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time) + h**2/12 * u_sec(0, t_n + dt*time)

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
            # if n % int(N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #     plt.plot(x1, np.flipud(u1_0), "b", label="approximation")
            #     plt.plot(x1, u_bar(x1-h1/2, h1, t_n+dt), "r--", label="solution")
            #     print("u at extremity:",u_real(-1, t_n + dt))
            #     # plt.plot(x_flux, np.flipud(phi1_0), "b", label="approximation")
            #     # plt.plot(x_flux, flux(x_flux, t_n+dt), "r--", label="solution")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()
            # n += 1

        ret += [np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt))]
        #print("errors: ", ret)

    return np.log(ret[0]/ret[1])/np.log(2)


