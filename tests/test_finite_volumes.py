import numpy as np
"""
import matplotlib
import matplotlib.pyplot as plt
"""
from numpy import cos, sin, pi
from numpy.random import random
from utils_numeric import solve_linear
from finite_volumes import get_Y_star, integrate_one_step_star


"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are constant, a and c are != 0.
    D is discontinuous over the interface.
    Right now the test is not passing because of the advection term.
    We should make an advection-like with fluxes rather than working
    with the discontinuous derivative of u
"""
def test_integrate_one_step_with_a_c():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 10, 10
    h1, h2 = 1/M1, 1/M2
    dt = 10000000
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = 0.0

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2/D1,
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    f = c * u_theoric + a*partial_xu

    u0 = np.zeros_like(u_theoric)
    u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f=f,
                neumann=neumann, dirichlet=dirichlet, u0=u0)

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"


"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are constant, a and c are zeros.
    D is discontinuous over the interface.
    The functions used are extremely simple because it's not trivial
    enough to have an analytical solution
"""
def test_integrate_one_step_simplest():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 20, 20
    h1, h2 = 1/M1, 1/M2
    dt = 100000
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))


    D1 = 2.4
    D2 = 1.4
    a = 0.0
    c = 0.0

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    neumann = pi
    dirichlet = -pi
    f = np.zeros_like(x)

    u0 = np.zeros_like(x)
    u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f=f,
                neumann=neumann, dirichlet=dirichlet, u0=u0)

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"

def test_integrate_one_step():
    #TODO assert "ok" == test_integrate_one_step_simplest()
    assert "ok" == test_integrate_one_step_with_a_c()
    return "ok"

def launch_all_tests():
    print("Test of compact scheme.")
    print("Test integration with finite volumes:", test_integrate_one_step())

