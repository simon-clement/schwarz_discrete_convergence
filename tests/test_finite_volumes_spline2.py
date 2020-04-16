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
    print("Test of Finite volumes scheme based on quadratic splines.")
    print("Test integration with finite volumes, spline 2:", test_integrate_one_step())
    # print("Test of reconstruction:", test_reconstruction())

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
    verify_order(Manfredi, linear_time, 2)
    verify_order(Manfredi, cosinus_time, 2)
    verify_order(Manfredi, exp_space_quad_time, 2)

    print("BACKWARD EULER SCHEME:")
    # Backward Euler is order 1 in time, so 2 in space (constant Courant parabolic number)
    verify_order(BackwardEuler, intricated_spacetime, 2)
    verify_order(BackwardEuler, linear_time, 3) # 3 bcause perfect in timeare bad ?
    verify_order(BackwardEuler, cosinus_time, 2)
    verify_order(BackwardEuler, exp_space_quad_time, 2)

    # the 2nd order time schemes gives 4rth order in space error.

    print("THETA SCHEME:")
    verify_order(ThetaMethod, intricated_spacetime, 2) # isn't it strange to have 2?
    verify_order(ThetaMethod, linear_time, 3)
    verify_order(ThetaMethod, cosinus_time, 3)
    verify_order(ThetaMethod, exp_space_quad_time, 2)

    print("RK2 SCHEME:")
    verify_order(RK2, intricated_spacetime, 2)
    verify_order(RK2, linear_time, 3)
    verify_order(RK2, cosinus_time, 3)
    verify_order(RK2, exp_space_quad_time, 2)

    # 4th orders time scheme are limited by space error:
    print("RK4 SCHEME:")
    verify_order(RK4, intricated_spacetime, 2)
    verify_order(RK4, linear_time, 3)
    verify_order(RK4, cosinus_time, 3)
    verify_order(RK4, exp_space_quad_time, 2)

    return "ok"

def exp_space_cubic_time(a, c, D):
    def u_real(x, t): return np.exp(x) * t**3
    def u_bar(x_1_2, h, t): return t**3/h * (np.exp(x_1_2+h) - np.exp(x_1_2))
    def flux(x, t): return D * np.exp(x) * t**3
    def f_bar(x, h, t): return 3*t**2/h*(np.exp(x+h) - np.exp(x)) - 1/h * D * (np.exp(x+h) - np.exp(x))*t**3
    return {'u':u_real, 'Bar{u}':u_bar, 'Ddu/dx':flux, 'Bar{f}':f_bar}


# This is not the same function as test_finite_differences.test_any_time_scheme_domain1
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
    """
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()
    from discretizations.space.quad_splines_fv import QuadSplinesFV as space_scheme

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 4.
    Lambda = 0.
    Courant = .1 # RK2 experimental CFL condition
    D = .2

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
        keys = ('u', 'Bar{u}', 'Ddu/dx', 'Bar{f}')
        u_real, u_bar, flux, f_bar = (u_ubar_flux_fbar[i] for i in keys)

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
            #     x1_accurate = np.linspace(0, -1, 1000, endpoint=False)
            #     plt.plot(x1_accurate, u_real(x1_accurate, t_n+dt), label="exact u_real")
            #     plt.plot(x1_accurate,
            #             [scheme.reconstruction_spline(phi1_0, u1_0, upper_domain=False, x=x) for x in x1_accurate], label="accurate reconstruction")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

        ret += [np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt))]
        # print("errors: ", ret)
    return np.log(ret[0]/ret[1])/np.log(2)

# This is not the same function as test_finite_differences.test_any_time_scheme_domain1
# because the right hand side is not the same, the comparison is not the same,
# we integrate in time the fluxes and not u
def test_any_time_scheme_domain2(time_scheme, u_ubar_flux_fbar=linear_time):
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
    T = 2.
    Lambda = 1e9
    Courant = .1 # RK2 experimental CFL condition
    D = .2

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_2 = Lambda
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

        scheme.DT = dt

        N = int(T/dt)
        T = N*dt

        scheme.M1 = M
        scheme.M2 = M
        h2 = 1 / M + np.zeros(M)
        h = h2[0]
        
        t_initial = 2.
        t_final = t_initial + T

        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x_flux = np.concatenate((x2-h/2, [1]))

        ######################
        # END OF THE SETTING DESCRIPTION.
        ######################

        ######################
        # EQUATION DEFINITION: WHICH U ? WHICH F ?
        ######################
        keys = ('u', 'Bar{u}', 'Ddu/dx', 'Bar{f}')
        u_real, u_bar, flux, f_bar = (u_ubar_flux_fbar[i] for i in keys)

        ###########################
        # TIME INTEGRATION:
        ############################

        # The following loop is valid independently of the u chosen.
        u2_0 = u_bar(x2 - h2/2, h2, t_initial)
        phi2_0 = flux(x_flux, t_initial)
        additional = [u2_0]
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def neumann(time): return flux(1, t_n + dt*time)

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f2(time):
                ret = f_bar(x_flux[:-1], h, t_n + time*dt)
                return np.concatenate(([ret[0]], np.diff(ret), [ret[-1]]))

            phi_np1, real_u_interface, real_phi_interface, *additional = scheme.integrate_one_step(f=f2,
                                                                             bd_cond=neumann,
                                                                             u_nm1=phi2_0,
                                                                             u_interface=u_int,
                                                                             phi_interface=phi_int,
                                                                             additional=additional,
                                                                             upper_domain=True)

            phi2_0 = phi_np1
            u2_0 = additional[0]

            # nb_plots = 4
            # if int(4*N * (t_n - t_initial) / T) % int(4*N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #     plt.plot(x2, u2_0, label="u_bar")
            #     plt.plot(x2, u_bar(x2-h2/2, h2, t_n+dt), label="exact ubar")
            #     plt.plot(x_flux, phi2_0, label="phi")
            #     plt.plot(x_flux, flux(x_flux, t_initial), label="exact phi")
            #     x2_accurate = np.linspace(0, 1, 1000, endpoint=False)
            #     plt.plot(x2_accurate, u_real(x2_accurate, t_n+dt), label="exact u_real")
            #     plt.plot(x2_accurate,
            #             [scheme.reconstruction_spline(phi2_0, u2_0, upper_domain=True, x=x) for x in x2_accurate], label="accurate reconstruction")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()


        ret += [np.linalg.norm(u_bar(x2-h2/2, h2, t_n+dt) - u2_0)/np.linalg.norm(u_bar(x2-h2/2, h2, t_n+dt))]
        # print("errors: ", ret)
    return np.log(ret[0]/ret[1])/np.log(2)


def test_reconstruction():
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
    from discretizations.time.Manfredi import Manfredi
    from discretizations.time.theta_method import ThetaMethod

    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()
    from discretizations.space.quad_splines_fv import QuadSplinesFV as space_scheme
    import matplotlib.pyplot as plt

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 1.
    Lambda = 1e9
    Courant=10.
    D = 1.
    plot_initial = True

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_1 = Lambda
    builder.LAMBDA_2 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.C = .0
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    high_freq = 5
    ######################
    # EQUATION DEFINITION: WHICH U ? WHICH F ?
    ######################
    def u_real(x, t): return np.sin(6*x) + np.sin(high_freq*x)
    def u_bar(x, h, t): return (np.cos(6*x)/6 - np.cos(6*x+6*h)/6 + (np.cos(high_freq*x) - np.cos(high_freq*(x+h)))/high_freq)/h
    def flux_initial(x, t): return D*(6*np.cos(6*x) + high_freq*np.cos(high_freq*x))
    def f_bar(x, h, t): return np.zeros_like(x)

    M = 7

    scheme = builder.build(Manfredi, space_scheme)
    ret = []
    # Loop to compare different settings:
    dt = 1/M**2*Courant/D

    scheme.DT = dt

    N = int(T/dt)

    scheme.M1 = M
    scheme.M2 = M
    h1 = 1 / M + np.zeros(M)
    h2 = 1 / M + np.zeros(M)
    
    t_initial = 0.
    t_final = t_initial + T

    x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
    x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))

    x1_flipped = -x1
    x1 = np.flipud(-x1)
    x1_flux = np.concatenate((x1-h1[0]/2,[0]))

    x2_flux = np.concatenate((x2-h2[0]/2, [scheme.SIZE_DOMAIN_2]))

    x1_accurate = np.linspace(0, -scheme.SIZE_DOMAIN_1, 1000, endpoint=False)
    x2_accurate = np.linspace(0, scheme.SIZE_DOMAIN_2, 1000, endpoint=False)
    ######################
    # END OF THE SETTING DESCRIPTION.
    ######################

    # The following loop is valid independently of the u chosen.
    u1_0 = np.flipud(u_bar(x1 - h1/2, h1, t_initial))
    phi1_0 = np.flipud(flux_initial(x1_flux, t_initial))
    additional = [u1_0]
    progress = ProgressBar()
    if plot_initial:
        print(x1)
        plt.plot(x1_accurate, u_real(x1_accurate, t_initial), label="initial accurate: domain 1")
        plt.plot(x1_flipped, u1_0, label="u_bar domain 1")
        plt.plot(x1_accurate, [scheme.reconstruction_spline(phi1_0, u1_0, upper_domain=False, x=x) for x in x1_accurate], "k--", label="reconstruction_spline domain 1")

    # The following loop is valid independently of the u chosen.
    u2_0 = u_bar(x2 - h2/2, h2, t_initial)
    phi2_0 = flux_initial(x2_flux, t_initial)
    progress = ProgressBar()
    if plot_initial:
        plt.plot(x2_accurate, u_real(x2_accurate, t_initial), label="initial accurate domain 2")
        plt.plot(x2, u2_0, label="u_bar domain 2")
        plt.plot(x2_accurate, [scheme.reconstruction_spline(phi2_0, u2_0, upper_domain=True, x=x) for x in x2_accurate], "k--", label="reconstruction_spline domain 2")
        plt.legend()

        plt.show()

    return


