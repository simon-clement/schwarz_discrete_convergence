#!/usr/bin/python3
import numpy as np
from numpy import pi, cos, sin
from discretizations.finite_difference import FiniteDifferences
from discretizations.finite_volumes import FiniteVolumes
import functools

def launch_all_tests():
    # increasing M2 and M1 won't impact performances in this test.
    print("verifying consistency...", discrete_tends_to_continuous())

    dt=.1
    finite_volumes = FiniteVolumes(A_DEFAULT=0.0, C_DEFAULT=1e-10,
            D1_DEFAULT=.6, D2_DEFAULT=.54, LAMBDA_2_DEFAULT=0.,
            SIZE_DOMAIN_1=200, SIZE_DOMAIN_2=200,
            M1_DEFAULT=8000000, M2_DEFAULT=800000,
            DT_DEFAULT=dt)

    finite_difference = FiniteDifferences(A_DEFAULT=0.0, C_DEFAULT=1e-10,
            D1_DEFAULT=.6, D2_DEFAULT=.54, LAMBDA_2_DEFAULT=0.,
            SIZE_DOMAIN_1=200, SIZE_DOMAIN_2=200,
            M1_DEFAULT=8000000, M2_DEFAULT=800000,
            DT_DEFAULT=dt)

    lambda_min = 1e-9
    lambda_max = 10
    steps = 100
    assert error_theoric_rate(finite_volumes, 2.3, pi/dt) < 1e-6
    assert error_theoric_rate(finite_difference, 2.3, pi/dt) < 1e-6
    
    print("verified theoric_rate... ok")

def theoric_optimal_lambda(discretization, TIME_WINDOW_LEN_DEFAULT):
    dt = discretization.DT_DEFAULT
    T = dt*TIME_WINDOW_LEN_DEFAULT
    
    sqD1 = np.sqrt(discretization.D1_DEFAULT)
    sqD2 = np.sqrt(discretization.D2_DEFAULT)
    sqw1 = np.sqrt(pi/T)
    sqw2 = np.sqrt(pi/dt)
    return 1/(2*np.sqrt(2)) * ((sqD2-sqD1)*(sqw1+sqw2) + np.sqrt((sqD2-sqD1)**2 * (sqw1 + sqw2)**2 + 8*sqD1*sqD2*sqw1*sqw2))


def error_theoric_rate(discretization, Lambda_1, w):
    rho_continuous = continuous_analytic_rate(discretization, Lambda_1, w)
    discrete_rho = discretization.analytic_robin_robin(s=w*1j, Lambda_1=Lambda_1)
    return np.abs(rho_continuous-discrete_rho)

def continuous_analytic_rate(discretization, Lambda_1, w):
    D1 = discretization.D1_DEFAULT
    D2 = discretization.D2_DEFAULT
    # sig1 is \sigma^1_{+}
    sig1 = np.sqrt(np.abs(w)/(2*D1)) * (1 + np.abs(w)/w * 1j)
    # sig2 is \sigma^2_{-}
    sig2 = -np.sqrt(np.abs(w)/(2*D2)) * (1 + np.abs(w)/w * 1j)
    return np.abs(D1*sig1*(D2*sig2+Lambda_1) / (D2*sig2*(D1*sig1+Lambda_1)))

def numerical_optimization(discretization, lambda_min, lambda_max):
    lambda_1 = np.linspace(lambda_min, lambda_max, 100)
    from scipy.optimize import minimize_scalar
    TIME_WINDOW_LEN_DEFAULT=5
    dt=discretization.DT_DEFAULT
    T = TIME_WINDOW_LEN_DEFAULT*dt

    dt = discretization.DT_DEFAULT
    T = dt*TIME_WINDOW_LEN_DEFAULT
    rho = []
    for t in np.linspace( dt, T, TIME_WINDOW_LEN_DEFAULT):
        rho += [[discretization.analytic_robin_robin(s=pi/t*1j, Lambda_1=i) \
                                                for i in lambda_1]]
    max_rho = np.max(np.array(rho),axis=0)

    # for each lambda we need to maximize in frequency
    def to_minimize_discrete(l):
        f = functools.partial(discretization.analytic_robin_robin, Lambda_1=l)
        #g = lambda x:-f(x*1j)
        # minimize_scalar with bounded domain is not good when using concave function
        #ret = -minimize_scalar(fun=g, method="bounded", bounds=(-pi/dt, -pi/T)).fun
        return max([f(pi/t*1j) for t in np.linspace(dt, T, TIME_WINDOW_LEN_DEFAULT)])

    def to_minimize_continuous(l):
        cont = functools.partial(continuous_analytic_rate,discretization, l)
        #mcont = lambda x:-cont(x)
        # minimize_scalar with bounded domain is not good when using concave function
        #ret_cont = -minimize_scalar(fun=mcont, method="bounded", bounds=(-pi/dt, -pi/T)).fun
        ret_cont = np.max([cont(pi/t) for t in np.linspace(dt, T, TIME_WINDOW_LEN_DEFAULT)])
        return ret_cont

    ret_continuous = minimize_scalar(fun=to_minimize_continuous)
    ret_discrete = minimize_scalar(fun=to_minimize_discrete)
    assert np.abs(ret_continuous.x - ret_discrete.x) < 1e-5
    assert np.abs(ret_continuous.fun - ret_discrete.fun) < 1e-5

    theoric_best = theoric_optimal_lambda(discretization, TIME_WINDOW_LEN_DEFAULT)
    assert np.abs(ret_continuous.x - theoric_best) < 1e-8
    assert np.abs(ret_discrete.x - theoric_best) < 1e-8
    assert np.abs(lambda_1[np.argmin(max_rho)]- theoric_best) < .1

def discrete_tends_to_continuous():
    # increasing M2 and M1 won't impact performances in this test.
    dt=.1
    finite_volumes = FiniteVolumes(A_DEFAULT=0.0, C_DEFAULT=1e-10,
            D1_DEFAULT=.6, D2_DEFAULT=.54, LAMBDA_2_DEFAULT=0.,
            SIZE_DOMAIN_1=200, SIZE_DOMAIN_2=200,
            M1_DEFAULT=80000000, M2_DEFAULT=8000000,
            DT_DEFAULT=0.1)

    finite_difference = FiniteDifferences(A_DEFAULT=0.0, C_DEFAULT=1e-10,
            D1_DEFAULT=.6, D2_DEFAULT=.54, LAMBDA_2_DEFAULT=0.,
            SIZE_DOMAIN_1=200, SIZE_DOMAIN_2=200,
            M1_DEFAULT=80000000, M2_DEFAULT=8000000,
            DT_DEFAULT=0.1)

    lambda_min = 1e-9
    lambda_max = 10
    assert error_theoric_rate(finite_volumes, 2.3, pi/dt) < 1e-6
    assert error_theoric_rate(finite_difference, 2.3, pi/dt) < 1e-6
    lambda_min = 1e-9
    numerical_optimization(finite_difference, lambda_min, lambda_max)
    numerical_optimization(finite_volumes, lambda_min, lambda_max)
    return "ok"
