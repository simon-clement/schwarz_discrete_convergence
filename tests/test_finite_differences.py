import numpy as np
from numpy import cos, sin
from numpy.random import random
import tests.test_cases
from tests.test_cases import intricated_spacetime, linear_time, cosinus_time, exp_space_quad_time


def launch_all_tests():
    from discretizations.space.FD_naive import FiniteDifferencesNaive as FD_naive
    from discretizations.space.FD_corr import FiniteDifferencesCorr as FD_corr
    from discretizations.space.FD_extra import FiniteDifferencesExtra as FD_extra
    print("Test of Finite Differences schemes.")
    print("Testing Naive Neumann interface.")
    print("Test integration finite differences corr=0:", test_integrate_one_step(FD_naive))
    print("Testing Neumann interface with a corrective term.")
    print("Test integration finite differences corr=1:", test_integrate_one_step(FD_corr, CORRECTIVE_TERM=True))
    print("Testing Neumann interface with an extrapolation of the flux.")
    print("Test integration finite differences extra:", test_integrate_one_step(FD_extra))
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

    print("MANFREDI SCHEME:")
    # Manfredi is a particular scheme. It makes an order 1 error when using a rhs
    verify_order(Manfredi, intricated_spacetime, 2)
    verify_order(Manfredi, linear_time, 2)
    verify_order(Manfredi, cosinus_time, 2)
    verify_order(Manfredi, exp_space_quad_time, 2)

    print("BACKWARD EULER SCHEME:")
    # Backward Euler is order 1 in time, so 2 in space (constant Courant parabolic number)
    verify_order(BackwardEuler, intricated_spacetime, 2) #Naive : =2.8
    verify_order(BackwardEuler, linear_time, 2) # exact in time so the only error is space
    verify_order(BackwardEuler, cosinus_time, 2) # Naive: =1.28
    verify_order(BackwardEuler, exp_space_quad_time, 2)

    # the 2nd order time schemes gives 4rth order in space error.

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
    T = 4.
    Lambda = 1e9
    Courant = .2 # 0.16 is RK2 experimental CFL condition
    D = 1.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_2 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
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

            def dirichlet(time): return u_real(-1, t_n + dt*time)

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
            #     plt.legend()
            #     plt.show()
            #     print("enter to continue, or ctrl-C to stop")
            #     input()

            u1_0 = u_np1

        ret += [np.linalg.norm(u_real(x_u, t_n+dt) - u1_0)/np.linalg.norm(u_real(x_u, t_n+dt))]

    print(ret)
    # print(np.log(ret[0]/ret[1])/np.log(2)) # <- space order. Time order is same but divided by 2
    return np.log(ret[0]/ret[1])/np.log(2)

