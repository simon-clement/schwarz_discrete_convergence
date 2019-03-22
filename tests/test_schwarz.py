import numpy as np
from numpy import cos, sin
from numpy.random import random
from cv_rate import solve_u_time_domain
from utils_numeric import integration
from utils_linalg import solve_linear
from finite_difference import get_Y, get_Y_star

def test_integration():

    # Our domain is [0,1]
    M = 401
    x = np.linspace(0,1,M)**6
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2])/2
    D = 1.0
    a = 0.0
    c = 0.0
    Y = get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)

    # Simplest case: f=0, u* = x
    f = np.zeros(M)
    f[1:M-1] = 0 * (hm1 + hm)
    f[-1] = np.pi
    assert np.linalg.norm(np.pi*x - solve_linear(Y, f)) < 1e-10

    dt = 0.1
    number_time_steps = 5000
    f_x0 = np.zeros(number_time_steps+1)
    u0 = np.zeros_like(f)
    u = integration(u0, Y, f, f_x0, dt=dt, theta=1) #implicit scheme
    assert (np.linalg.norm(np.pi*x - u[-1])) < 1e-9

    M = 10001
    a = 1.2
    c = 0.3
    d = 1.2

    x = np.linspace(0,1,M)**3
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2])/2

    D = lambda x: x
    D_prime = lambda x: 1
    D_star = D((x[1:] + x[:-1])/2)


    function_f = lambda x : d*d*D(x)*sin(d*x) - d*D_prime(x)*cos(d*x) \
        + d*a*cos(d*x) + c*sin(d*x)
    f = np.zeros(M)
    f[1:M-1] = function_f(x[1:M-1]) * (hm1 + hm)
    f[-1] = d*cos(d*middle_grid_last)

    Y = get_Y_star(M_star=M, h_star=h, D_star=D_star, a=a, c=c)
    assert np.linalg.norm(sin(d*x) - solve_linear(Y, f)) < 1e-7

    dt = 100
    number_time_steps = 5000
    f_x0 = np.zeros(number_time_steps+1)
    u0 = np.zeros_like(f)
    u = integration(u0, Y, f, f_x0, dt=dt, theta=1) # Implicit scheme
    assert np.linalg.norm(sin(d*x) - u[-1]) < 1e-7

    dt = 100
    number_time_steps = 5000
    f_x0 = np.zeros(number_time_steps+1)
    u0 = np.zeros_like(f)
    u = integration(u0, Y, f, f_x0, dt=dt, theta=0.5) # Crank-Nicholson scheme
    assert np.linalg.norm(sin(d*x) - u[-1]) < 1e-3

    return "ok"

def test_time_domain():
    # Our domain is [-1,1]
    # Omega_2 is [0,1]
    # Omega_1 is [-1,0]
    M = 11
    a = 1.2
    c = 0.3
    d = 1.2

    Lambda1 = 1.0
    Lambda2 = 2.0

    x2 = np.linspace(0,1,M)**3
    h2 = np.diff(x2)
    h2m1 = h2[:-1]
    h2m = h2[1:]
    middle_grid_last2 = (x2[-1] + x2[-2])/2
    D2 = lambda x: 1.1-x*x
    D2_prime = lambda x: -2*x
    D2_send = D2((x2[1:] + x2[:-1])/2)
    function_f2 = lambda x : d*d*D2(x)*sin(d*x) - d*D2_prime(x)*cos(d*x) \
        + d*a*cos(d*x) + c*sin(d*x)

    # Omega_1 values:
    x1 = np.linspace(0,-1,M)**3
    h1 = np.diff(x1)
    h1m1 = h1[:-1]
    h1m = h1[1:]
    D1 = lambda x: 1.1-x*x
    D1_prime = lambda x: -2*x
    D1_send = D1((x1[1:] + x1[:-1])/2)
    function_f1 = lambda x : d*d*D1(x)*sin(d*x) - d*D1_prime(x)*cos(d*x) \
        + d*a*cos(d*x) + c*sin(d*x)

    f1 = np.zeros(M)
    f1[1:M-1] = function_f1(x1[1:M-1]) * (h1m1 + h1m)
    f1[-1] = sin(d*x1[-1]) # Dirichlet bd condition

    f2 = np.zeros(M)
    f2[1:M-1] = function_f2(x2[1:M-1]) * (h2m1 + h2m)
    f2[-1] = d*cos(d*middle_grid_last2) # Neumann bd condition
    f2[0] = d*cos(d*x2[1]/2)*D2_send[0] # Robin bd condition, u(0) = 0

    f1[0] = D1_send[0] * d*cos(d*x1[1]/2)

    u1_init = np.zeros_like(f1)
    u2_init = np.zeros_like(f2)

    nb_time_steps = 2000
    u = solve_u_time_domain(u1_init, u2_init,
                            f_star_0=function_f1(0)*(x2[1] - x1[1]),
                        f1=f1, f2=f2,
                        Lambda_1=Lambda1, Lambda_2=Lambda2,
                        D1=D1_send, D2=D2_send,
                        h1=h1, h2=h2, a=a, c=c, dt=1,
                        number_time_steps=nb_time_steps)


    M = 2*M - 1
    x = np.linspace(-1,1,M)**3
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2])/2

    D = lambda x: 1.1 - x*x
    D_prime = lambda x: -2*x
    D_star = D((x[1:] + x[:-1])/2)


    function_f = lambda x : d*d*D(x)*sin(d*x) - d*D_prime(x)*cos(d*x) \
        + d*a*cos(d*x) + c*sin(d*x)
    f = np.zeros(M)
    f[1:M-1] = function_f(x[1:M-1]) * (hm1 + hm)
    f[-1] = d*cos(d*middle_grid_last)
    f[0] = sin(d*x[0]) # Dirichlet bd condition
    f0_on_time = np.ones(nb_time_steps+1) * f[0]
    Y = get_Y_star(M_star=M, h_star=h, D_star=D_star, a=a, c=c)
    ureal = solve_linear(Y, f)

    assert np.linalg.norm(ureal - u[-1]) < 1e-10
    return "ok"


"""
    Simple test of the Schwarz method
"""
def test_schwarz():
    # Our domain is [-1,1]
    # Omega_2 is [0,1]
    # Omega_1 is [-1,0]
    M = 10001
    a = 1.2
    c = 0.3
    d = 1.2
    dt=1000.0
    #TODO ajouter terme correctif

    Lambda1 = 1.0
    Lambda2 = 2.0

    x2 = np.linspace(0,1,M)**3
    h2 = np.diff(x2)
    h2m1 = h2[:-1]
    h2m = h2[1:]
    middle_grid_last2 = (x2[-1] + x2[-2])/2
    D2 = lambda x: x
    D2_prime = lambda x: 1
    D2_send = D2((x2[1:] + x2[:-1])/2)
    function_f2 = lambda x : d*d*D2(x)*sin(d*x) - d*D2_prime(x)*cos(d*x) \
        + d*a*cos(d*x) + c*sin(d*x)
    Y2 = get_Y(M=M, Lambda=Lambda2, h=h2, D=D2_send, dt=dt,
               a=a, c=c, upper_domain=True)

    # Omega_1 values:
    x1 = np.linspace(0,-1,M)**3
    h1 = np.diff(x1)
    h1m1 = h1[:-1]
    h1m = h1[1:]
    D1 = lambda x: -x
    D1_prime = lambda x: -1
    D1_send = D1((x1[1:] + x1[:-1])/2)
    function_f1 = lambda x : d*d*D1(x)*sin(d*x) - d*D1_prime(x)*cos(d*x) \
        + d*a*cos(d*x) + c*sin(d*x)

    Y1 = get_Y(M=M, Lambda=Lambda1, h=h1, D=D1_send,
               a=a, c=c, upper_domain=False)

    f1 = np.zeros(M)
    f1[1:M-1] = function_f1(x1[1:M-1]) * (h1m1 + h1m)
    f1[-1] = sin(d*x1[-1]) # Dirichlet bd condition

    f2 = np.zeros(M)
    f2[1:M-1] = function_f2(x2[1:M-1]) * (h2m1 + h2m)
    f2[-1] = d*cos(d*middle_grid_last2) # Neumann bd condition

    f2[0] = d*cos(d*x2[1]/2)*D2_send[0] # Robin bd condition, u(0) = 0
    assert np.linalg.norm(sin(d*x2) - solve_linear(Y2, f2)) < 1e-7

    f1[0] = D1_send[0] * d*cos(d*x1[1]/2)
    assert np.linalg.norm(sin(d*x1) - solve_linear(Y1, f1)) < 2e-6

    initial_error1 = np.linalg.norm(sin(d*x1) - solve_linear(Y1, f1))
    initial_error2 = np.linalg.norm(sin(d*x2) - solve_linear(Y2, f2))

    f1[0] = f2[0] = -3
    #TODO ajuster terme correctif

    for i in range(1, 100):
        u1 = solve_linear(Y1, f1)

        f2[0] = D1_send[0] * (u1[1] - u1[0]) / h1[0] + Lambda2 * u1[0]
        u2 = solve_linear(Y2, f2)

        f1[0] = D2_send[0] * (u2[1] - u2[0]) / h2[0] + Lambda1 * u2[0]

    # we tolerate 0.1% additional error (before it was 5%)
    tol_err = 1.001 * max(initial_error1, initial_error2)
    assert np.linalg.norm(sin(d*x1) - solve_linear(Y1, f1)) < tol_err
    assert np.linalg.norm(sin(d*x2) - solve_linear(Y2, f2)) < tol_err

    return "ok"

def launch_all_tests():
    print("Test Schwarz method:", test_schwarz())
    print("Test time integration:", test_integration())
    print("Test time domain solution:", test_time_domain())
