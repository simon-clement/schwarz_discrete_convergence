import numpy as np
from numpy import cos, sin
from numpy.random import random
import tests.test_cases
from tests.test_cases import intricated_spacetime, linear_time, cosinus_time, exp_space_quad_time, const_space_quad_time, const_space_const_time


def launch_all_tests():
    from discretizations.space.FD_bulk import FiniteDifferencesBulk as FD_bulk
    print("Test of Finite Differences schemes.")
    print("Testing Naive Neumann interface.")

    print("Test integration finite differences with interface fluxes:", test_integrate_one_step(FD_bulk))
    return "ok"

def complete_test_schwarz():
    from tests.test_schwarz import schwarz_convergence
    ecart = schwarz_convergence(fdifference)
    assert ecart[-1] < 1e-10

    return "ok"

def test_integrate_one_step(space_scheme_FD, CORRECTIVE_TERM=False):
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.Manfredi import Manfredi
    def verify_order(time_scheme, function, order, **kwargs): # /!\ give order in space !
        try:
            experimental_order = test_any_time_scheme_domain1(time_scheme, space_scheme_FD, function,  **kwargs)
            if abs(experimental_order - order) < .4:
                return
            else:
                print("Error for ", time_scheme.__name__, function.__name__,
                        ": should be space order", order, "but is:", experimental_order)
        except(KeyboardInterrupt):
            print("Interrupted test", time_scheme.__name__, function.__name__, "by user interruption.")
        except:
            raise

    print("BACKWARD EULER SCHEME:")
    verify_order(BackwardEuler, linear_time, 2) # exact in time so the only error is space
    verify_order(BackwardEuler, const_space_quad_time, 2) # exact in space or almost so the only error is time
    # Backward Euler is order 1 in time, so 2 in space (constant Courant parabolic number)
    verify_order(BackwardEuler, intricated_spacetime, 2) #Naive : =2.8
    verify_order(BackwardEuler, cosinus_time, 2) # Naive: =1.28
    verify_order(BackwardEuler, exp_space_quad_time, 2)

    print("MANFREDI SCHEME:")
    # Manfredi is a particular scheme. It makes an order 1 error when using a rhs
    verify_order(Manfredi, intricated_spacetime, 2)
    verify_order(Manfredi, linear_time, 2)
    verify_order(Manfredi, cosinus_time, 2)
    verify_order(Manfredi, exp_space_quad_time, 2)

    print("THETA SCHEME:")
    verify_order(ThetaMethod, intricated_spacetime, 2)
    verify_order(ThetaMethod, linear_time, 2)
    verify_order(ThetaMethod, cosinus_time, 2)
    verify_order(ThetaMethod, exp_space_quad_time, 2)

    if not CORRECTIVE_TERM:
        print("RK2 SCHEME:")
        verify_order(RK2, intricated_spacetime, 2)
        verify_order(RK2, linear_time, 2)
        verify_order(RK2, cosinus_time, 2)
        verify_order(RK2, exp_space_quad_time, 2)

        # 4th orders time scheme are limited by space error:
        print("RK4 SCHEME:")
        verify_order(RK4, intricated_spacetime, 2)
        verify_order(RK4, linear_time, 2)
        verify_order(RK4, cosinus_time, 2)
        verify_order(RK4, exp_space_quad_time, 2)

    return "ok"

# This is not the same function as test_finite_volumes*.test_any_time_scheme_domain1
# because the right hand side is not the same, the comparison is not the same,
# we integrate in time u and not the fluxes
def test_any_time_scheme_domain1(time_scheme, space_scheme, u_flux_f=linear_time, first_M=8):
    """
    Simplest Case : a=c=0
    on se place en (-1, 0)
    Neumann en -1
    """
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 4.
    Lambda = 1.
    Courant = .2 # 0.16 is RK2 experimental CFL condition
    D = 20.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_2 = Lambda
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or r!=0, the rhs must take them into account
    builder.R = 0.
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_flux_f = u_flux_f(builder.A, builder.R, D)
    scheme = builder.build(time_scheme, space_scheme)

    ret = []
    # Loop to compare different settings:
    for M in (first_M, first_M*2):
        dt = 1/(M*M)*Courant/D
        scheme.DT = dt

        N = int(T/dt)
        T = N * dt

        scheme.M1 = M
        scheme.M2 = M
        h1, h2 = scheme.get_h()
        h = h1[0]
        assert h1[0]<0
        
        t_initial = 0.
        t_final = t_initial + T

        x1 = np.cumsum(np.concatenate(([h1[0] / 2], (h1[1:] + h1[:-1]) / 2)))
        x_du = np.concatenate((x1-h/2, [-builder.SIZE_DOMAIN_1]))

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
        u1_0 = u_real(x1, t_initial)
        additional = [u1_0]
        uprime1_0 = flux(x_du, t_initial)/D
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(-1, t_n + dt*time)

            def neumann(time): return flux(-1, t_n + dt*time)/D

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f1(time): return f(x1, t_n + dt*time)

            uprime_np1, _, _, *additional = scheme.integrate_one_step(f=f1,
                                                 bd_cond=neumann,
                                                 u_nm1=uprime1_0,
                                                 u_interface=u_int,
                                                 phi_interface=phi_int,
                                                 additional=additional,
                                                 upper_domain=False)
            u_np1 = additional[0]

            nb_plots = 4
            # if int(4 * N * (t_n - t_initial) / T) % int(4*N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #    
            #     #plt.plot(x_du, uprime_np1, label="approximation flux, should be 0")
            #     #plt.plot(x_du, flux(x_du, t_n+dt)/D, "m--", label="real flux")
            #     plt.plot(x_du, flux(x_du, t_n+dt)/D - uprime_np1, label="approximation flux, should be 0")
            #     #plt.plot(x_du, flux(x_du, t_n+dt)/D, "m--", label="real flux")
            #     plt.plot(x1, u_real(x1, t_n+dt) - u_np1, "g", label="approximation np1")
            #     #plt.plot(x1, u_real(x1, t_n+dt), "y--", label="solution np1")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

            u1_0 = u_np1
            uprime1_0 = uprime_np1

        ret += [np.linalg.norm(u_real(x1, t_n+dt) - u1_0)/np.linalg.norm(u_real(x1, t_n+dt))]

    print(ret)
    # print(np.log(ret[0]/ret[1])/np.log(2)) # <- space order. Time order is same but divided by 2
    return np.log(ret[0]/ret[1])/np.log(2)

# This is not the same function as test_finite_volumes*.test_any_time_scheme_domain1
# because the right hand side is not the same, the comparison is not the same,
# we integrate in time u and not the fluxes
def test_any_time_scheme_domain2(time_scheme, space_scheme, u_flux_f=linear_time, first_M=8):
    """
    Simplest Case : a=c=0
    on se place en ( 0, 1)
    Dirichlet en 1
    """
    from figures import Builder
    from progressbar import ProgressBar
    builder = Builder()

    ###########################################
    # DEFINITION OF THE SETTING:
    ###########################################
    T = 4.
    Lambda = -10.
    Courant = .2 # 0.16 is RK2 experimental CFL condition
    D = .2

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_2 = Lambda
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or r!=0, the rhs must take them into account
    builder.R = 0.
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_flux_f = u_flux_f(builder.A, builder.R, D)
    scheme = builder.build(time_scheme, space_scheme)

    ret = []
    # Loop to compare different settings:
    for M in (first_M, first_M*2):
        dt = 1/(M*M)*Courant/D
        scheme.DT = dt

        N = int(T/dt)
        T = N * dt

        scheme.M1 = M
        scheme.M2 = M
        h1, h2 = scheme.get_h()
        h = h2[0]
        
        t_initial = 0.
        t_final = t_initial + T

        x_u = np.cumsum(np.concatenate(([h2[0] / 2], (h2[1:] + h2[:-1]) / 2)))
        x_du = np.concatenate((x_u-h/2, [builder.SIZE_DOMAIN_2]))

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
        additional = [u2_0]
        uprime2_0 = flux(x_du, t_initial)/D
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(1, t_n + dt*time)

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f2(time): return f(x_u, t_n + dt*time)

            uprime_np1, _, _, *additional = scheme.integrate_one_step(f=f2,
                                                 bd_cond=dirichlet,
                                                 u_nm1=uprime2_0,
                                                 u_interface=u_int,
                                                 phi_interface=phi_int,
                                                 additional=additional,
                                                 upper_domain=True)
            u_np1 = additional[0]

            nb_plots = 4
            # if int(4 * N * (t_n - t_initial) / T) % int(4*N/nb_plots) == 0:
            #     import matplotlib.pyplot as plt
            #    
            #     #plt.plot(x_du, uprime_np1, label="approximation flux, should be 0")
            #     plt.plot(x_du, uprime_np1 - flux(x_du, t_n+dt)/D, label="approximation flux, should be 0")
            #     #plt.plot(x_u, flux(x_u, t_n+dt)/D, label="real flux")
            #     plt.plot(x_u, u_np1 - u_real(x_u, t_n+dt), "g", label="approximation np1")
            #     # plt.plot(x_u, u_real(x_u, t_n+dt), "y--", label="solution np1")
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

            u2_0 = u_np1
            uprime2_0 = uprime_np1

        ret += [np.linalg.norm(u_real(x_u, t_n+dt) - u2_0)/np.linalg.norm(u_real(x_u, t_n+dt))]

    print(ret)
    # print(np.log(ret[0]/ret[1])/np.log(2)) # <- space order. Time order is same but divided by 2
    return np.log(ret[0]/ret[1])/np.log(2)

