"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi
from numpy.random import random
import matplotlib.pyplot as plt
from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma
"""
    Test function of the module.
    Tests the finite volumes space scheme, with all the time schemes.
"""


def launch_all_tests():
    test_like_Manfredi_paper_FV()
    test_with_all_space_schemes()

def test_with_all_space_schemes():
    first_M=10
    from discretizations.time.theta_method import ThetaMethod
    import tests.test_finite_differences
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    import tests.test_finite_volumes
    import tests.test_finite_volumes_spline2
    from tests.test_cases import intricated_spacetime, linear_time, cosinus_time, exp_space_quad_time, const_space_cubic_time, zero_f, const_space_quad_time
    print("the Padé scheme is supposed to be of order 2 in time (thus 4 in space)")
    for test_case in (zero_f, const_space_cubic_time, const_space_quad_time, intricated_spacetime, linear_time, cosinus_time, exp_space_quad_time):#(const_space_cubic_time, intricated_spacetime, linear_time, cosinus_time, exp_space_quad_time):
        print("testing", test_case.__name__)
        print("Time order with ThetaSchema FV4:", \
                tests.test_finite_volumes.test_any_time_scheme_domain1(ThetaMethod,
                test_case, first_M, )/2)
        print("Time order PadeLowTildeGamma FV4:", \
                tests.test_finite_volumes.test_any_time_scheme_domain1(PadeLowTildeGamma,
                test_case, first_M )/2)

        """
        print("Order with FV4:", \
            tests.test_finite_volumes.test_any_time_scheme_domain1(time_scheme=PadeLowTildeGamma,
                first_M=first_M, u_ubar_flux_fbar=test_case)/2)
        print("Order with FV2:", \
            tests.test_finite_volumes_spline2.test_any_time_scheme_domain1(time_scheme=PadeLowTildeGamma,
                first_M=first_M, u_ubar_flux_fbar=test_case)/2)
        """


def test_like_Manfredi_paper_FV():
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
    from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.RK4 import RK4

    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()
    from discretizations.space.quad_splines_fv import QuadSplinesFV as space_scheme
    #from discretizations.space.fourth_order_fv import FourthOrderFV as space_scheme

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = .1
    Lambda = 1e9
    Courant=80.
    D = .3
    plot_initial = True

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_2 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.R = .0
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    high_freq = 500
    ######################
    # EQUATION DEFINITION: WHICH U ? WHICH F ?
    ######################
    def u_real(x, t): return 3*np.sin(7*x) + np.sin(high_freq*x)
    def flux_real(x, t): return D*(7*3*np.cos(7*x) + high_freq*np.cos(high_freq*x))
    def u_bar(x, h, t): return (3*(np.cos(7*x) - np.cos(7*(x+h)))/7 + (np.cos(high_freq*x) - np.cos(high_freq*(x+h)))/high_freq)/h
    def flux_initial(x, t): return D*(3*7*np.cos(7*x) + high_freq*np.cos(high_freq*x))
    def f_bar(x, h, t): return np.zeros_like(x)

    M = 220

    ret = []
    # Loop to compare different settings:
    for scheme, name in zip((builder.build(ThetaMethod, space_scheme),
        builder.build(PadeLowTildeGamma, space_scheme)), ("ThetaMethod", "PadeLowTildeGamma")):
        dt = 1/M**2*Courant/D

        scheme.DT = dt

        N = int(T/dt)

        scheme.M1 = M
        scheme.M2 = M
        h2 = 1 / M + np.zeros(M)
        h = h2[0]
        
        t_initial = 0.
        t_final = t_initial + T

        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x_flux = np.concatenate((x2-h/2,[0]))

        x_accurate = np.linspace(0, scheme.SIZE_DOMAIN_2, 30000, endpoint=False)
        ######################
        # END OF THE SETTING DESCRIPTION.
        ######################

        ###########################
        # TIME INTEGRATION:
        ############################

        n=0
        # The following loop is valid independently of the u chosen.
        u2_0 = u_bar(x2 - h2/2, h2, t_initial)
        phi2_0 = flux_initial(x_flux, t_initial)
        additional = [u2_0]
        progress = ProgressBar()
        if plot_initial:
            plt.plot(x_accurate, u_real(x_accurate, t_initial), label="initial, real")
            plt.plot(x_accurate, [scheme.reconstruction_spline(phi2_0, u2_0, upper_domain=True, x=x) for x in x_accurate], "k--", label="initial, reconstruction")

            plot_initial = False
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(1, t_n + dt*time)

            def phi_int(time): return 0.#flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f2(time):
                return np.zeros(scheme.size_f(upper_domain=True))

            phi_np1, real_u_interface, real_phi_interface, *additional = scheme.integrate_one_step(f=f2,
                                                                             bd_cond=dirichlet,
                                                                             u_nm1=phi2_0,
                                                                             u_interface=u_int,
                                                                             phi_interface=phi_int,
                                                                             additional=additional,
                                                                             upper_domain=True)
            phi2_0 = phi_np1
            u2_0 = additional[0]

        #print("errors: ", ret)
        #plt.plot(phi2_0, label=name)
        plt.plot(x_accurate, [scheme.reconstruction_spline(phi2_0, u2_0, upper_domain=True, x=x) for x in x_accurate], label=name)

    plt.legend()
    plt.show()

    return


def test_like_Manfredi_paper_FD():
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
    from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma
    from discretizations.time.theta_method import ThetaMethod

    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()
    from discretizations.space.FD_naive import FiniteDifferencesNaive as space_scheme

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = .4
    Lambda = 1e9
    Courant=100.
    D = .05
    plot_initial = True

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.R = .0
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    high_freq = 130
    ######################
    # EQUATION DEFINITION: WHICH U ? WHICH F ?
    ######################
    def u_real(x, t): return 3*np.sin(6*x) + np.sin(high_freq*x)
    def flux_initial(x, t): return (3*6*np.cos(6*x) + high_freq*np.cos(high_freq*x))
    def f_bar(x, h, t): return np.zeros_like(x)

    M = 228

    ret = []
    # Loop to compare different settings:
    for scheme, name in zip((builder.build(PadeLowTildeGamma, space_scheme),
        builder.build(ThetaMethod, space_scheme)), ("PadeLowTildeGamma", "C-N")):
        dt = 1/M**2*Courant/D

        scheme.DT = dt

        N = int(T/dt)

        scheme.M1 = M
        scheme.M2 = M
        h1 = 1 / M + np.zeros(M)
        h = h1[0]
        
        t_initial = 0.
        t_final = t_initial + T

        x1 = np.cumsum(np.concatenate(([0], h1)))
        x1 = np.flipud(-x1[:-1])
        x_flux = np.concatenate((x1-h/2,[0]))

        ######################
        # END OF THE SETTING DESCRIPTION.
        ######################

        ###########################
        # TIME INTEGRATION:
        ############################

        n=0
        # The following loop is valid independently of the u chosen.
        u1_0 = np.flipud(u_real(x1, t_initial))
        progress = ProgressBar()
        if plot_initial:
            plt.plot(x1, np.flipud(u1_0), label="initial")
            plot_initial = False
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(-1, t_n + dt*time)

            def phi_int(time): return 0.#flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f1(time):
                return np.zeros(scheme.size_f(upper_domain=False))

            u_np1, real_u_interface, real_phi_interface, *additional = scheme.integrate_one_step(f=f1,
                                                                             bd_cond=dirichlet,
                                                                             u_nm1=u1_0,
                                                                             u_interface=u_int,
                                                                             phi_interface=phi_int,
                                                                             additional=[None],
                                                                             upper_domain=False)
            u1_0 = u_np1

        #print("errors: ", ret)
        plt.plot(x1, np.flipud(u_np1), label=name)

    plt.legend()
    plt.show()

    return np.log(ret[0]/ret[1])/np.log(2)


