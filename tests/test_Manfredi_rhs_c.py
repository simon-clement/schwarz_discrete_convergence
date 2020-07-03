"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi
from numpy.random import random
import matplotlib.pyplot as plt
from discretizations.time.Manfredi import Manfredi
"""
    Test function of the module.
    Tests the finite volumes space scheme, with all the time schemes.
"""

def launch_all_tests():
    test_FD()
    test_FV2()
    test_FV4()

def test_FV4():
    first_M=800
    first_dt = 1e-2
    c = 20.
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from tests.test_cases import zero_f, const_space_cubic_time, intricated_spacetime, const_space_cos_time, cosinus_time
    for test_case in (const_space_cos_time, cosinus_time, intricated_spacetime, zero_f):
        print("the Padé scheme is supposed to be of order 2 in time (thus 4 in space)")

        print("testing", test_case.__name__)
        print("Time order with Manfredi:", \
        test_any_time_scheme_domain1_FV4(time_scheme=Manfredi,
            first_M=first_M, u_ubar_flux_fbar=test_case, first_dt=first_dt, c=c))
        # print("Time order with ThetaMethod:", \
        # test_any_time_scheme_domain1_FV4(time_scheme=ThetaMethod,
        #             first_M=first_M, u_ubar_flux_fbar=test_case, first_dt=first_dt, c=c))

def test_FV2():
    first_M=80000
    first_dt = 1e-2
    c = 2.
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from tests.test_cases import zero_f, const_space_cubic_time, intricated_spacetime, const_space_cos_time, cosinus_time
    for test_case in (intricated_spacetime, const_space_cos_time, cosinus_time, zero_f):
        print("the Padé scheme is supposed to be of order 2 in time (thus 4 in space)")

        print("testing", test_case.__name__)
        print("Time order with Manfredi:", \
        test_any_time_scheme_domain2_FV2(time_scheme=Manfredi,
            first_M=first_M, u_ubar_flux_fbar=test_case, first_dt=first_dt, c=c, adjust_M=False))
        # print("Time order with ThetaMethod:", \
        # test_any_time_scheme_domain2_FV2(time_scheme=ThetaMethod,
        #             first_M=first_M, u_ubar_flux_fbar=test_case, first_dt=first_dt, c=c))

def test_FD():
    first_M=800
    first_dt = 1e-1
    c = 0.1
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from tests.test_cases import zero_f, const_space_cubic_time, intricated_spacetime, const_space_quad_time, cosinus_time, const_space_cos_time
    for test_case in (const_space_cos_time, ):#intricated_spacetime, zero_f, const_space_cos_time, ):
        print("the Padé scheme is supposed to be of order 2 in time (thus 4 in space)")

        print("testing", test_case.__name__)
        print("Time order with Manfredi:", \
        test_any_time_scheme_domain1_FD(time_scheme=Manfredi,
            space_scheme=FiniteDifferencesNaive, first_M=first_M, u_flux_f=test_case, first_dt=first_dt, c=c, adjust_M=False))
        print("Time order with BackwardEuler:", \
        test_any_time_scheme_domain1_FD(time_scheme=BackwardEuler,
                    space_scheme=FiniteDifferencesNaive, first_M=first_M, u_flux_f=test_case, first_dt=first_dt, c=c, adjust_M=False))



# This is not the same function as test_finite_volumes*.test_any_time_scheme_domain1
# because the right hand side is not the same, the comparison is not the same,
# we integrate in time u and not the fluxes
def test_any_time_scheme_domain1_FD(time_scheme, space_scheme, u_flux_f, first_M=8, first_dt=1e-2, c=0., adjust_M=True):
    """
    Simplest Case : a=c=0
    on se place en (-1, 0)
    Dirichlet en -1
    Notre fonction est sin(x) + t
    Donc le schéma en temps résout exactement sa partie,
    l'erreur en h**2 vient seulement de l'espace :)
    On compense l'erreur au bord en h**2 par une condition
    de "Dirichlet" : u + h**2/12*u''
    """
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 1.
    Lambda = 1e9
    Courant = 1. # 0.16 is RK2 experimental CFL condition
    D = 1.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.R = c
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_flux_f = u_flux_f(builder.A, builder.R, D)

    ret = []
    M = first_M
    # Loop to compare different settings:
    for dt in (first_dt, first_dt/2):
        if int((Courant/(D*dt))) > first_M and adjust_M:
            M = int((Courant/(D*dt)))
            print("warning: M is limitant")

        builder.DT = dt

        N = int(T/dt)
        T = N * dt

        builder.M1 = M
        builder.M2 = M

        scheme = builder.build(time_scheme, space_scheme)

        h1, h2 = scheme.get_h()
        h = h1[0]
        assert h1[0]<0
        
        t_initial = 0.
        t_final = t_initial + T

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x_u = np.concatenate((x1-h/2, [-builder.SIZE_DOMAIN_1]))

        ######################
        # END OF THE SETTING DESCRIPTION.
        ######################

        ######################
        # EQUATION DEFINITION: WHICH U ? WHICH F ?
        ######################

        keys = ('u', 'Ddu/dx', 'f')
        u_real, flux, f = (u_flux_f[i] for i in keys)

        ###########################
        # TIME INTEGRATION:
        ############################

        # The following loop is valid independently of the u chosen.
        u1_0 = u_real(x_u, t_initial)
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def neumann(time): return flux(-1, t_n + dt*time)/D

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f1(time): return f(x_u, t_n + dt*time)

            u_np1, _, _ = scheme.integrate_one_step(f=f1,
                                                 bd_cond=neumann,
                                                 u_nm1=u1_0,
                                                 u_interface=u_int,
                                                 phi_interface=phi_int,
                                                 upper_domain=False)

            # nb_plots = 4
            # if int(4 * N * (t_n - t_initial) / T) % int(4*N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #     plt.plot(x_u, u_np1, "g", label="approximation np1")
            #     plt.plot(x_u, u_real(x_u, t_n+dt), "y--", label="solution np1")
            #     #plt.plot(x_u, u1_0 - u_real(x_u, t_n+dt), label="error")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

            u1_0 = u_np1

        ret += [np.linalg.norm(u_real(x_u, t_n+dt) - u1_0)/np.linalg.norm(u_real(x_u, t_n+dt))]

    print(ret)
    # print(np.log(ret[0]/ret[1])/np.log(2)) # <- time order
    return np.log(ret[0]/ret[1])/np.log(2)


def test_any_time_scheme_domain2(time_scheme, space_scheme, u_flux_f, first_M=8, first_dt=1e-2, c=0.):
    """
    Simplest Case : a=c=0
    on se place en (-1, 0)
    Dirichlet en -1
    Notre fonction est sin(x) + t
    Donc le schéma en temps résout exactement sa partie,
    l'erreur en h**2 vient seulement de l'espace :)
    On compense l'erreur au bord en h**2 par une condition
    de "Dirichlet" : u + h**2/12*u''
    """
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 1.
    Lambda = 1e9
    Courant = 1. # 0.16 is RK2 experimental CFL condition
    D = 1.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_2 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.R = c
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_flux_f = u_flux_f(builder.A, builder.R, D)

    ret = []
    M = first_M
    # Loop to compare different settings:
    for dt in (first_dt, first_dt/2):
        if int((Courant/(D*dt))) > first_M and adjust_M:
            M = int((Courant/(D*dt)))
            print("warning: M is limitant")

        builder.DT = dt

        N = int(T/dt)
        T = N * dt

        builder.M1 = M
        builder.M2 = M

        scheme = builder.build(time_scheme, space_scheme)

        h1, h2 = scheme.get_h()
        h = h2[0]
        assert h2[0]>0
        
        t_initial = 0.
        t_final = t_initial + T

        x2 = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x_u = np.concatenate((x2-h/2, [builder.SIZE_DOMAIN_2]))

        ######################
        # END OF THE SETTING DESCRIPTION.
        ######################

        ######################
        # EQUATION DEFINITION: WHICH U ? WHICH F ?
        ######################

        keys = ('u', 'Ddu/dx', 'f')
        u_real, flux, f = (u_flux_f[i] for i in keys)

        ###########################
        # TIME INTEGRATION:
        ############################

        # The following loop is valid independently of the u chosen.
        u2_0 = u_real(x_u, t_initial)
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(1, t_n + dt*time)

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f2(time): return f(x_u, t_n + dt*time)

            u_np1, _, _ = scheme.integrate_one_step(f=f2,
                                                 bd_cond=dirichlet,
                                                 u_nm1=u2_0,
                                                 u_interface=u_int,
                                                 phi_interface=phi_int,
                                                 upper_domain=True)

            # nb_plots = 4
            # if int(4 * N * (t_n - t_initial) / T) % int(4*N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #     plt.plot(x_u, u_np1, "g", label="approximation np1")
            #     plt.plot(x_u, u_real(x_u, t_n+dt), "y--", label="solution np1")
            #     #plt.plot(x_u, u1_0 - u_real(x_u, t_n+dt), label="error")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

            u2_0 = u_np1

        ret += [np.linalg.norm(u_real(x_u, t_n+dt) - u2_0)/np.linalg.norm(u_real(x_u, t_n+dt))]

    print(ret)
    # print(np.log(ret[0]/ret[1])/np.log(2)) # <- time order
    return np.log(ret[0]/ret[1])/np.log(2)

def test_any_time_scheme_domain1_FV4(time_scheme, u_ubar_flux_fbar, first_M=8, first_dt=1e-2, c=0., adjust_M=True):
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
    Lambda = 1e9
    Courant=.1
    D = 2.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.R = c
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_ubar_flux_fbar = u_ubar_flux_fbar(builder.A, builder.R, D)

    scheme = builder.build(time_scheme, space_scheme)

    ret = []
    M = first_M
    # Loop to compare different settings:
    for dt in (first_dt, first_dt/2):
        if int((Courant/(D*dt))) > first_M:
            M = int((Courant/(D*dt)))
            print("warning: M is limitant")

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

            #def dirichlet(time): return u_real(-1, t_n + dt*time) + h**2/12 * u_sec(-1, t_n + dt*time)

            def neumann(time): return flux(-1, t_n + dt*time)/D

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time) + h**2/12 * u_sec(0, t_n + dt*time)

            def f1(time):
                ret = np.flipud(f_bar(x_flux[:-1], h, t_n + time*dt))
                return np.concatenate(([ret[0]], np.diff(ret), [ret[-1]]))

            phi_np1, real_u_interface, real_phi_interface, *additional = scheme.integrate_one_step(f=f1,
                                                                             bd_cond=neumann,
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
            #     plt.plot(x1, np.flipud(u1_0), "b", label="approximation u1")
            #     plt.plot(x1, u_bar(x1-h1/2, h1, t_n+dt), "r--", label="solution u_bar")
            #     plt.plot(x_flux, np.flipud(phi1_0), label="approximation phi")
            #     plt.plot(x_flux, flux(x_flux, t_n+dt), label="solution phi")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()
            # n += 1

        ret += [np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt) - np.flipud(u1_0))/np.linalg.norm(u_bar(x1-h1/2, h1, t_n+dt))]

    print(ret)
    return np.log(ret[0]/ret[1])/np.log(2)


def test_any_time_scheme_domain2_FV2(time_scheme, u_ubar_flux_fbar, first_M=10, first_dt=1e-2, c=0., adjust_M=True):
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()
    from discretizations.space.quad_splines_fv import QuadSplinesFV as space_scheme

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 2.
    Lambda = 0.
    Courant = .1 # RK2 experimental CFL condition
    D = 1.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_2 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.R = c
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_ubar_flux_fbar = u_ubar_flux_fbar(builder.A, builder.R, D)

    scheme = builder.build(time_scheme, space_scheme)

    ret = []
    # Loop to compare different settings:
    M = first_M
    dts = (first_dt, first_dt/2, first_dt/3, first_dt/4, first_dt/8)
    #dts = (first_dt, first_dt/2)
    for dt in dts:
        if int((Courant/(D*dt))) > first_M and adjust_M:
            M = int((Courant/(D*dt)))
            print("warning: M is limitant")

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

            def dirichlet(time): return u_real(1, t_n + dt*time)

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f2(time):
                ret = f_bar(x_flux[:-1], h, t_n + time*dt)
                return np.concatenate(([ret[0]], np.diff(ret), [ret[-1]]))

            phi_np1, real_u_interface, real_phi_interface, *additional = scheme.integrate_one_step(f=f2,
                                                                             bd_cond=dirichlet,
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
            #     plt.plot(x2, u_bar(x2-h2/2, h2, t_n-dt/np.sqrt(2)), label="ubar^\\star")
            #     plt.plot(x2, u_bar(x2-h2/2, h2, t_n+dt), label="exact ubar")
            #     plt.plot(x_flux, phi2_0, label="phi")
            #     plt.plot(x_flux, flux(x_flux, t_initial), label="exact phi")
            #     x2_accurate = np.linspace(0, 1, 1000, endpoint=False)
            #     #plt.plot(x2_accurate, u_real(x2_accurate, t_n+dt), label="exact u_real")
            #     #plt.plot(x2_accurate,
            #     #        [scheme.reconstruction_spline(phi2_0, u2_0, upper_domain=True, x=x) for x in x2_accurate], label="accurate reconstruction")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()


        ret += [np.linalg.norm(u_bar(x2-h2/2, h2, t_n+dt) - u2_0)/np.linalg.norm(u_bar(x2-h2/2, h2, t_n+dt))]
    # FIGURE OF VALIDATION ORDRE 2:
    import matplotlib.pyplot as plt
    plt.loglog(dts, np.array(dts)**2, "k--", label="y=dt^2")
    plt.loglog(dts, ret, '+', label="error on u=cos(x+t), with reaction")
    plt.xlabel("dt")
    plt.ylabel("error")
    plt.legend()
    plt.show()
    return np.log(ret[0]/ret[1])/np.log(2)

