import numpy as np
from numpy import cos, sin
from numpy.random import random


def launch_all_tests():
    from discretizations.space.FD_naive import FiniteDifferencesNaive as FD_naive
    from discretizations.space.FD_corr import FiniteDifferencesCorr as FD_corr
    from discretizations.space.FD_extra import FiniteDifferencesExtra as FD_extra
    print("Test integration finite differences corr=0:", test_integrate_one_step(FD_naive))
    #print("Test integration finite differences corr=1:", test_integrate_one_step(FD_corr))
    #print("Test integration finite differences extra:", test_integrate_one_step(FD_extra))
    #print("Test complete finite differences:", complete_test_schwarz())
    return "ok"

def complete_test_schwarz():
    from tests.test_schwarz import schwarz_convergence
    ecart = schwarz_convergence(fdifference)
    assert ecart[-1] < 1e-10

    return "ok"

def test_integrate_one_step(space_scheme_FD):
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.Manfredi import Manfredi
    u_flux_f = const_space_quad_time
    print(test_all_time_schemes_domain1(ThetaMethod, space_scheme_FD, u_flux_f))
    print(test_all_time_schemes_domain1(BackwardEuler, space_scheme_FD, u_flux_f))
    print(test_all_time_schemes_domain1(Manfredi, space_scheme_FD, u_flux_f))
    #print(test_all_time_schemes_domain1(RK2, space_scheme_FD, u_flux_f))
    #print(test_all_time_schemes_domain1(RK4, space_scheme_FD, u_flux_f))

    return "ok"

def linear_time(a, c, D):
    def u_real(x, t): return np.sin(x) + t
    def flux(x, t): return D * np.cos(x)
    def f(x, t): return 1 - (- D * np.sin(x))
    return u_real, flux, f

def cosinus_time(a, c, D):
    def u_real(x, t): return np.sin(x) + np.cos(t)
    def flux(x, t): return D * np.cos(x)
    def f(x, t): return -np.sin(t) - (- D * np.sin(x))
    return u_real, flux, f

def intricated_spacetime(a, c, D):
    def u_real(x, t): return np.sin(x + t)
    def flux(x, t): return D * np.cos(x + t)
    def f(x, t): return np.cos(x+t) - ( - D * np.sin(x+t))
    return u_real, flux, f

def exp_spacetime(a, c, D):
    def u_real(x, t): return np.exp(x + t)
    def flux(x, t): return D * np.exp(x + t)
    def f(x, t): return np.exp(x+t) - D * np.exp(x+t)
    return u_real, flux, f

def exp_space_quad_time(a, c, D):
    def u_real(x, t): return np.exp(x)*t**2
    def flux(x, t): return D * np.exp(x)*t**2
    def f(x, t): return 2*t*np.exp(x) - D * np.exp(x)*t**2
    return u_real, flux, f

def const_space_quad_time(a, c, D):
    def u_real(x, t): return t**2 + np.zeros_like(x)
    def flux(x, t): return 0 + np.zeros_like(x)
    def f(x, t): return 2*t + np.zeros_like(x)
    return u_real, flux, f

def solution_no_forcing(a, c, D):
    def u_real(x, t): return np.exp(x/np.sqrt(D) + t)
    def flux(x, t): return np.sqrt(D)*np.exp(x/np.sqrt(D) + t)
    def f(x, t): return np.zeros_like(t) + np.zeros_like(x)
    return u_real, flux, f

# This is not the same function as test_finite_differences.test_all_time_schemes_domain1
# because the right hand side is not the same, the comparison is not the same,
# we integrate in time u and not the fluxes
def test_all_time_schemes_domain1(time_scheme, space_scheme, u_flux_f=linear_time):
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
    Courant = 10 # 0.16 is RK2 experimental CFL condition
    D = 1.

    builder.COURANT_NUMBER = Courant
    builder.LAMBDA_1 = Lambda
    builder.D1 = D
    builder.D2 = D
    builder.A = 0. # warning: note that if a!=0 or c!=0, the rhs must take them into account
    builder.C = 0.
    builder.SIZE_DOMAIN_1 = 1.
    builder.SIZE_DOMAIN_2 = 1.

    u_flux_f = u_flux_f(builder.A, builder.C, D)
    scheme = builder.build(time_scheme, space_scheme)

    ret = []
    # Loop to compare different settings:
    for M in (8, 16):
        dt = 1/M**2*Courant/D
        M = 3
        scheme.DT = dt

        N = int(T/dt)
        T = N * dt
        print(T)

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

        u_real, flux, f = u_flux_f

        ###########################
        # TIME INTEGRATION:
        ############################

        # The following loop is valid independently of the u chosen.
        u1_0 = u_real(x_u, t_initial)
        progress = ProgressBar()
        for t_n in progress(np.linspace(t_initial, t_final, N, endpoint=False)):

            def dirichlet(time): return u_real(-1, t_n + dt*time)

            def phi_int(time): return flux(0, t_n + dt*time)

            def u_int(time): return u_real(0, t_n + dt*time)

            def f1(time): return f(x_u, t_n + dt*time)

            u_np1, real_u_interface, real_phi_interface = scheme.integrate_one_step(f=f1,
                                                                             bd_cond=dirichlet,
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
    else:
        return time_scheme.__name__ + " time order is: " + str(np.log(ret[0]/ret[1])/np.log(2) / 2)

