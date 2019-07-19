#!/usr/bin/python3
"""
    This module is a container of the generators of more maths figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
    use this module with ./cv_rate.py try fun_name
"""
import numpy as np
from numpy import pi
from discretizations.finite_difference import FiniteDifferences
from discretizations.finite_volumes import FiniteVolumes
from discretizations.finite_difference_no_corrective_term \
        import FiniteDifferencesNoCorrectiveTerm
from discretizations.finite_difference_naive_neumann \
        import FiniteDifferencesNaiveNeumann
import functools
import cv_rate
from cv_rate import continuous_analytic_rate_robin_neumann
from cv_rate import continuous_analytic_rate_robin_robin
from cv_rate import analytic_robin_robin, interface_errors
from cv_rate import rate_fast
from cv_rate import raw_simulation
from cv_rate import frequency_simulation, frequency_simulation_slow
from memoisation import memoised, FunMem
import matplotlib.pyplot as plt
from figures import DEFAULT

def inverse_Y(Y):
    return np.linalg.inv(np.diag(Y_0, k=-1)+np.diag(Y_1) + np.diag(Y_2,k=1))

def local_in_time():
    N=1
    dis = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    a, c, dt = dis.get_a_c_dt(None, None, None)
    M1 = DEFAULT.M1
    M2 = DEFAULT.M2
    h1 = -DEFAULT.SIZE_DOMAIN_1 / (M1-1)
    h2 = DEFAULT.SIZE_DOMAIN_2 / (M2-1)
    D1 = DEFAULT.D1
    D2 = DEFAULT.D2
    Lambda_1 = 0.5
    Lambda_2 = -0.5
    e_simu = np.abs(interface_errors(dis, 1, Lambda_1=Lambda_1, Lambda_2=Lambda_2,
                              a=a, c=c, dt=dt, M1=M1, M2=M2, NUMBER_IT=4, seed=0))
    rho_simu = e_simu[1:] / e_simu[:-1]
    np.random.seed(0)
    all_u1_interface = 2 * (np.random.rand(1) - 0.5)
    all_phi1_interface = 2 * (np.random.rand(1) - 0.5)

    T = np.diag(np.ones(N)) + np.diag(-np.ones(N-1), k=-1)

    first_guess = T @ (Lambda_2 * all_u1_interface + all_phi1_interface)

    I_0_1 = np.array([1] + [0]*(M1-1))
    I_0_2 = np.array([1] + [0]*(M2-1))
    I_1_2 = np.array([Lambda_1 - D2/h2] + [D2/h2] + [0]*(M2-2))
    I_1_1 = np.array([Lambda_2 - D1/h1] + [D1/h1] + [0]*(M1-2))

    Y1 = dis.precompute_Y(M=M1, h=h1, D=D1, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_1, upper_domain=False)
    Y2 = dis.precompute_Y(M=M2, h=h2, D=D2, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_2, upper_domain=True)

    Y_inv_2 = inverse_Y(Y2)
    Y_inv_1 = inverse_Y(Y1)
    mat2 = T@np.reshape(I_1_2@Y_inv_2@I_0_2, (1,1))
    mat1 = T@np.reshape(I_1_1@Y_inv_1@I_0_1, (1,1))
    mat1_fin = T@np.reshape(I_0_1@Y_inv_1@I_0_1, (1,1))

    print("simu:", e_simu[0], e_simu[1], e_simu[2])
    print("théorique :", all_u1_interface[0], mat1_fin*mat2*first_guess,
            mat1_fin*mat2*mat1*mat2*first_guess)
    #print("taux de convergence:", mat1*mat2)
    #print("rho simu : ", rho_simu)

def best_parameters():
    from scipy.optimize import minimize
    x0 = np.array([.5, -.5])
    print(minimize(method="Nelder-Mead", fun=local_in_time_rate, x0=x0))

def local_in_time_rate(params):
    print(params.flatten())
    params = params.flatten()
    Lambda_1 = params[0]
    Lambda_2 = params[1]
    N=1
    dis = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    a, c, dt = dis.get_a_c_dt(None, None, None)
    M1 = DEFAULT.M1
    M2 = DEFAULT.M2
    h1 = -DEFAULT.SIZE_DOMAIN_1 / (M1-1)
    h2 = DEFAULT.SIZE_DOMAIN_2 / (M2-1)
    D1 = DEFAULT.D1
    D2 = DEFAULT.D2


    I_0_1 = np.array([1] + [0]*(M1-1))
    I_0_2 = np.array([1] + [0]*(M2-1))
    I_1_2 = np.array([Lambda_1 - D2/h2] + [D2/h2] + [0]*(M2-2))
    I_1_1 = np.array([Lambda_2 - D1/h1] + [D1/h1] + [0]*(M1-2))

    Y1 = dis.precompute_Y(M=M1, h=h1, D=D1, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_1, upper_domain=False)
    Y2 = dis.precompute_Y(M=M2, h=h2, D=D2, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_2, upper_domain=True)

    Y_inv_2 = inverse_Y(Y2)
    Y_inv_1 = inverse_Y(Y1)
    mat2 = np.reshape(I_1_2@Y_inv_2@I_0_2, (1,1))
    mat1 = np.reshape(I_1_1@Y_inv_1@I_0_1, (1,1))
    mat1_fin = np.reshape(I_0_1@Y_inv_1@I_0_1, (1,1))
    return (mat1*mat2)**2


def global_in_time():
    N=30
    # Les volumes finis ont l'air teeeellement chiants à gérer ><
    dis = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    a, c, dt = dis.get_a_c_dt(None, None, None)
    M1 = DEFAULT.M1
    M2 = DEFAULT.M2
    h1 = -DEFAULT.SIZE_DOMAIN_1 / (M1-1)
    h2 = DEFAULT.SIZE_DOMAIN_2 / (M2-1)
    D1 = DEFAULT.D1
    D2 = DEFAULT.D2
    Lambda_1 = 0.54272485
    Lambda_2 = -0.4926211
    e_simu = interface_errors(dis, N, Lambda_1=Lambda_1, Lambda_2=Lambda_2,
                              a=a, c=c, dt=dt, M1=M1, M2=M2, NUMBER_IT=4, seed=0)
    np.random.seed(0)
    all_u1_interface = 2 * (np.random.rand(N) - 0.5)
    all_phi1_interface = 2 * (np.random.rand(N) - 0.5)

    R1 = np.diag(np.concatenate(((0,), np.ones(M1-2), (0,))))
    R2 = np.diag(np.concatenate(((0,), np.ones(M2-2), (0,))))

    first_guess = (Lambda_2 * all_u1_interface + all_phi1_interface)

    I_0_1 = np.array([1] + [0]*(M1-1))
    I_0_2 = np.array([1] + [0]*(M2-1))
    I_1_1 = dis.give_robin_projector(M1, h1, D1, a, c, dt, 0, Lambda_2)
    I_1_2 = dis.give_robin_projector(M2, h2, D2, a, c, dt, 0, Lambda_1)

    Y1 = dis.give_Y_for_analysis(M=M1, h=h1, D=D1, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_1, upper_domain=False)
    Y2 = dis.give_Y_for_analysis(M=M2, h=h2, D=D2, a=a, c=c, dt=dt,
                          f=None, bd_cond=None,
                          Lambda=Lambda_2, upper_domain=True)

    Y_inv_2 = np.linalg.inv(Y2)
    Y_inv_1 = np.linalg.inv(Y1)
    Z1_fin = []
    Z2_fin = []
    Z1 = []
    Z2 = []
    for n in range(1, N+1):
        bloc1 = []
        bloc1_fin = []
        bloc2 = []
        bloc2_fin = []
        for i in range(N):
            if i < n:
                bloc1_fin += [I_0_1.T @ np.linalg.matrix_power(Y_inv_1 @ R1, n - i-1) @Y_inv_1 @ I_0_1]
                bloc1 += [I_1_1.T @ np.linalg.matrix_power(Y_inv_1 @ R1, n - i-1) @Y_inv_1 @ I_0_1]
                bloc2_fin += [np.reshape(np.linalg.matrix_power(Y_inv_2 @ R2, n - i-1) @Y_inv_2 @ I_0_2, (-1, 1))]
                bloc2 += [I_1_2.T @ np.linalg.matrix_power(Y_inv_2 @ R2, n - i-1) @Y_inv_2 @ I_0_2]
            else:
                bloc1_fin += [np.zeros(1)]
                bloc2_fin += [np.zeros_like(np.reshape(I_0_2, (-1,1)))]
                bloc1 += [np.zeros(1)]
                bloc2 += [np.zeros(1)]
        Z1_fin += [np.hstack(bloc1_fin)]
        Z2_fin += [np.hstack(bloc2_fin)]
        Z1 += [np.hstack(bloc1)]
        Z2 += [np.hstack(bloc2)]
    Z1_fin = np.vstack(Z1_fin)
    Z2_fin = np.vstack(Z2_fin)
    Z1 = np.vstack(Z1)
    Z2 = np.vstack(Z2)

    """
    # WARNING ! 2h/DT TO HAVE A CORRECT MATRIX
    # -> we multiply u by I3 = 2h/dt * (0, 1, .. 1, 0)
    err1 = Y_inv_2@I_0_2*first_guess[0]
    print("1st temps:")
    print(err1)
    print("erreur 2nd temps (should be correct):")
    print(Y_inv_2 @ R2 @ Y_inv_2@I_0_2*first_guess[0] + Y_inv_2 @ I_0_2 * first_guess[1])
    """

    print("Z1_fin@Z2*first_guess:", Z1_fin@Z2@first_guess)
    print("error simu:", e_simu[1])
    #print(Z1_fin@Z2@Z1@Z2@first_guess)
    #print("Z1@Z2:", Z1@Z2)

    #print("erreur :", all_u1_interface, Z2_fin@first_guess)
    #print("rho_simu :", np.array(e_simu[2]) / np.array(e_simu[1]))
    #print("rho_fast :", (Z1_fin@Z2@Z1@Z2@first_guess) / (Z1_fin@Z2@first_guess))


all_functions = {}
##########################################################
# Filling the dictionnary with the functions             #
##########################################################
# First take all globals defined in this module:
for key, glob in globals().copy().items():
    # Note that we don't check if it is a function,
    # So that a user can give a callable (for example, with functools.partial)
    # Ok the problem is that we also can call variables... But the user is smart, right ?
    all_functions[key] = glob
