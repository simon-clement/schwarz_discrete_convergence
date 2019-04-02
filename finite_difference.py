"""
    Finite differences for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""

import numpy as np
from utils_numeric import solve_linear

DT_DEFAULT = -1e-4
H_DEFAULT = -1e-4
D_DEFAULT = -1.0
A_DEFAULT = -0.0
C_DEFAULT = -0.0

"""
    Entry point in the module.
    Provided equation parameters M, h, D, a, c, dt, f;
    Provided boundary condition bd_cond, phi_interface, u_interface, Lambda;
    Provided former state of the equation u_nm1;
    Returns (u_n, u_interface, phi_interface)
    u_n is the next state vector, {u, phi}_interface are the
    values of the state vector at interface,
    necessary to compute Robin conditions.

    If upper_domain is True, the considered domain is Omega_2 (atmosphere)
    bd_cond is then the Neumann condition of the top of the atmosphere.
    If upper_domain is False, the considered domain is Omega_1 (ocean)
    bd_cond is then the Dirichlet condition of the bottom of the ocean.
    h, D, f and u_nm1 have their first values ([0,1,..]) at the interface
    and their last values ([..,M-2, M-1]) at
    the top of the atmosphere (Omega_2) or bottom of the ocean (Omega_1)

    M is int
    a, c, dt, bd_cond, Lambda, u_interface, phi_interface are float
    h, D, f can be float or np.ndarray of dimension 1.
    if h is a ndarray: its size must be M
    if f is a ndarray: its size must be M
    if D is a ndarray: its size must be M+1
    u_nm1 must be a np.ndarray of dimension 1 and size M

"""
def integrate_one_step(M, h, D, a, c, dt, f, bd_cond, Lambda, u_nm1,
        u_interface, phi_interface, upper_domain=True):
    a, c, dt, bd_cond, Lambda, u_interface, phi_interface = float(a), \
            float(c), float(dt), float(bd_cond), float(Lambda), \
            float(u_interface), float(phi_interface)

    # Broadcasting / verification of type:
    D = np.zeros(M-1) + D
    h = np.zeros(M-1) + h
    f = np.zeros(M) + f

    assert type(u_nm1) == np.ndarray and u_nm1.ndim == 1 and u_nm1.shape[0] == M
    assert upper_domain is True or upper_domain is False

    #TODO include integration in time in Y, so that
    # it can be uniformized with finite_volumes.get_Y ?
    Y = get_Y(M=M, h=h, D=D, a=a, c=c, dt=dt,
            Lambda=Lambda, upper_domain=upper_domain)
    Y[1][1:-1] += (np.ones(M-2) / dt) * (h[1:] + h[:-1]) 

    #TODO right hand side of the right shape
    rhs = (f[1:-1] + u_nm1[1:-1] / dt) * (h[1:] + h[:-1]) 

    cond_robin = Lambda * u_interface + phi_interface \
            - h[0] / 2 * (u_nm1[0] / dt + f[0])

    rhs = np.concatenate(([cond_robin], rhs, [bd_cond]))

    u_n = solve_linear(Y, rhs)

    new_u_interface = u_n[0]
    # Finite difference approx with the corrective term:
    new_phi_interface = D[0]/h[0] * (u_n[1] - u_n[0]) \
        - h[0] / 2 * ((u_n[0]-u_nm1[0])/dt + a*(u_n[1]-u_n[0])/h[0] \
                      + c * u_n[0] - f[0])


    assert u_n.shape[0] == M
    return u_n, new_u_interface, new_phi_interface

"""
    See integrate_one_step. This function integrates in time
    the full system: should work better if D1[0] == D2[0].
    h1 should be negative, and h1[0] is the interface.
"""
def integrate_one_step_star(M1, M2, h1, h2, D1, D2, a, c, dt, f1, f2,
        neumann, dirichlet, u_nm1):
    a, c, dt, neumann, dirichlet = float(a), float(c), float(dt), \
            float(neumann), float(dirichlet)
    # Theses assertions cannot be used because of the unit tests:
    # for arg, name in zip((a, c, neumann, dirichlet), 
    #         ("a", "c", "neumann", "dirichlet")):
    #     assert arg >= 0, name + " should be positive !"
    assert dt > 0, "dt should be strictly positive"

    assert type(M2) is int and type(M1) is int
    assert M2 > 0 and M1 > 0
    if type(D1) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D1 = float(D1)
    if type(D2) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D2 = float(D2)
    for arg, name in zip((h2, D1, D2), ("h1", "h2", "D1", "D2")):
        assert type(arg) is float or \
                type(arg) is np.float64 or \
                type(arg) is np.ndarray and \
                arg.ndim == 1, name
        assert (np.array(arg) > 0).all(), name + " is negative or 0 !"

    assert (np.array(h1) < 0).all(), "h1 is positive or 0 ! should be <0."

    #Broadcasting / verification of types:
    D1 = np.zeros(M1-1) + D1
    D2 = np.zeros(M2-1) + D2
    h1 = np.zeros(M1-1) + h1
    h2 = np.zeros(M2-1) + h2
    f1 = np.zeros(M1) + f1
    f2 = np.zeros(M2) + f2
    # Flipping arrays to have [0] at low altitudes rather than interface
    h1f, f1f, D1f = np.flipud(h1), np.flipud(f1), np.flipud(D1)
    h1f = -h1f # return to positive h1

    D = np.concatenate((D1f, D2))
    h = np.concatenate((h1f, h2))
    f = np.concatenate((f1f[:-1],
        [f1f[-1]*h1f[-1]/(h1f[-1]+h2[0]) + f2[0]*h2[0]/(h1f[-1]+h2[0])],
        f2[1:]))
    M = M1 + M2 - 1

    Y = get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)


    rhs = np.concatenate(([dirichlet],
        (h[1:] + h[:-1]) * (f[1:-1] + u_nm1[1:-1] / dt),
        [neumann]))

    Y[1][1:-1] += (h[1:] + h[:-1]) * np.ones(M-2) / dt

    u_n = solve_linear(Y, rhs)

    u1_n = np.flipud(u_n[:M1])
    u2_n = u_n[M1-1:]

    assert u2_n.shape[0] == M2

    phi_interface = (D1[0] + D2[0]) * (u2_n[1] - u1_n[1]) / (h2[0]-h1[0])

    return u_n, u1_n[0], phi_interface

"""
    Returns the tridiagonal matrix Y in the shape asked by solve_linear.
    (Does not actually returns a np matrix, it returns (Y_0, Y_1, Y_2).
    For details, see the documentation of @solve_linear)

    The returned matrix is of dimension MxM
    Lambda is the free parameter of Robin boundary conditions

    The following parameters can either be given as floats or np.ndarray

    h: step size (>0 in Omega_2, <0 in Omega_1) (size: M-1)
    D: diffusivity (always positive) (size: M-1)
        Note: if D is a np.ndarray, it should be given on the half-steps,
                i.e. D[m] is D_{m+1/2}
    a: advection coefficient (should be positive) (size: M-2)
    c: reaction coefficient (should be positive) (size: M-2)

    If upper_domain is True, returns Y2; otherwise returns Y1
    (The upper domain is \Omega_2)

    It is assumed that we use an implicit Euler discretisation in time.
    However, contrary to finite_volumes.get_Y the time discretisation
    is not contained in the returned matrix.
    (The assumption is for the corrective term)
    /!\ WARNING : DUPLICATE CODE between get_Y_star and get_Y (for lisibility)

"""
def get_Y(M, Lambda, h=H_DEFAULT, D=D_DEFAULT, a=A_DEFAULT,
                        c=C_DEFAULT, dt=DT_DEFAULT, upper_domain=True):
    assert type(a) is float or type(a) is int
    assert type(c) is float or type(c) is int
    if type(h) is int:
        print("Warning: type of step size is int. casting to float...")
        h = float(h)
    if type(D) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D = float(D)
    assert type(M) is int
    assert M > 0
    assert type(Lambda) is float
    for arg in (h, D):
        assert type(arg) is float or \
                type(arg) is np.ndarray and \
                arg.ndim == 1 and arg.shape[0] == M-1
    assert type(upper_domain) is bool

    # Broadcast or verification of size:
    D = np.zeros(M-1) + D
    h = np.zeros(M-1) + h

    if (np.array(h) <= 0).any() and upper_domain:
        print("Warning : h should not be negative in Omega_2")
    if (np.array(h) >= 0).any() and not upper_domain:
        print("Warning : h should not be positive in Omega_1")
    if (np.array(D) < 0).any():
        print("Warning : D should never be negative")
    if a < 0:
        print("Warning : a should probably not be negative")
    if c < 0:
        print("Warning : c should probably not be negative")

    h_m = h[1:] # h_{m}
    h_mm1 = h[:-1] # h_{m-1}
    sum_both_h = h_m + h_mm1 # (h_{m-1} + h_{m})
    D_mp1_2 = D[1:] # D_{m+1/2}
    D_mm1_2 = D[:-1] # D_{m-1/2}
    ######## MAIN DIAGONAL
    Y_1 = np.empty(M)

    Y_1[1:M-1] = c*sum_both_h + 2*(h_mm1*D_mp1_2 + h_m* D_mm1_2) / \
            (h_m*h_mm1)

    corrective_term = h[0] / 2 * (1 / dt + c) - a / 2
    Y_1[0] = Lambda - D[0] / h[0] - corrective_term # Robin bd conditions at interface
    Y_1[M-1] = 1 # Neumann bd conditions for \Omega_2, Dirichlet for \Omega_1
    if upper_domain:
        Y_1[M-1] /= h[-1]

    ######## RIGHT DIAGONAL
    Y_2 = np.empty(M-1)
    Y_2[1:] = -2 * D_mp1_2 / h_m

    Y_2[0] = D[0] / h[0] - a / 2
    Y_2[1:] += a

    ######## LEFT DIAGONAL
    Y_0 = np.empty(M-1)
    Y_0[:-1] = -2 * D_mm1_2 / h_mm1
    Y_0[:-1] -= a
    if not upper_domain:
        Y_0[-1] = 0 # In \Omega_1 we have Dirichlet bd conditions
    else:
        Y_0[-1] = -1/h[-1] # In \Omega_2 we have Neumann bd conditions

    return (Y_0, Y_1, Y_2)


"""
    Returns the tridiagonal matrix Y* in the shape asked by solve_linear.
    Y* is the matrix we need to inverse to solve the full domain.
    This function is useful to compute u*, solution of Y*u*=f*

    (Does not actually return a np matrix, it returns (Y_0, Y_1, Y_2).
    For details, see the documentation of @solve_linear)

    The returned matrix is of dimension M_starxM_star
    To compare with the coupled system and get:
        u*[0:M] = u1[0:M] (i.e. for all m, u*[m] = u1[m]
        u*[M-1:2M-1] = u2[0:M] (i.e. for all m, u*[M + m] = u2[m]
    We should have:
        - M_star = 2M - 1
        - D_star[0:M] = D1[0:M]
        - h_star[0:M] = h1[0:M]
        - D_star[M-1:2M-1] = D2[0:M]
        - h_star[M-1:2M-1] = h2[0:M]

    The following parameters can either be given as floats or np.ndarray

    h: step size (always positive) (size: M-1)
    D: diffusivity (always positive) (size: M-1)
        Note: if D is a np.ndarray, it should be given on the half-steps,
                i.e. D[m] is D_{m+1/2}
    a: advection coefficient (should be positive) (size: M-2)
    c: reaction coefficient (should be positive) (size: M-2)

    /!\ WARNING : DUPLICATE CODE between get_Y_star and get_Y (for lisibility)

"""
def get_Y_star(M_star, h_star=H_DEFAULT, D_star=D_DEFAULT,
        a=A_DEFAULT, c=C_DEFAULT):
    M = M_star
    h = h_star
    D = D_star
    assert type(a) is float or type(a) is int
    assert type(c) is float or type(c) is int
    if type(h) is int:
        print("Warning: type of step size is int. casting to float...")
        h = float(h)
    if type(D) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D = float(D)
    assert type(M) is int
    assert M > 0
    for arg in (h, D):
        assert type(arg) is float or \
                type(arg) is np.ndarray and \
                arg.ndim == 1
    if (np.array(h) < 0).any():
        print("Warning : h should never be negative")
    if (np.array(D) < 0).any():
        print("Warning : D should never be negative")
    if a < 0:
        print("Warning : a should probably not be negative")
    if c < 0:
        print("Warning : c should probably not be negative")

    # Broadcast or verification of size:
    D = np.zeros(M-1) + D
    h = np.zeros(M-1) + h
    # In the notations h is negative when considering \omega_1:

    h_m = h[1:] # h_{m}
    h_mm1 = h[:-1] # h_{m-1}
    sum_both_h = h_m + h_mm1 # (h_{m-1} + h_{m})
    D_mp1_2 = D[1:] # D_{m+1/2}
    D_mm1_2 = D[:-1] # D_{m-1/2}
    ######## MAIN DIAGONAL
    Y_1 = np.empty(M)

    Y_1[1:M-1] = c*sum_both_h + 2*(h_mm1*D_mp1_2 + h_m* D_mm1_2) / \
            (h_m*h_mm1)
    Y_1[0] = 1 # Dirichlet
    Y_1[M-1] = 1/h[-1] # Neumann

    ######## RIGHT DIAGONAL
    Y_2 = np.empty(M-1)
    Y_2[1:] = -2 * D_mp1_2 / h_m + a
    Y_2[0] = 0

    ######## LEFT DIAGONAL
    Y_0 = np.empty(M-1)
    Y_0[:-1] = -2 * D_mm1_2 / h_mm1 - a
    Y_0[-1] = -1/h[-1] # Neumann bd conditions on top

    return (Y_0, Y_1, Y_2)

if __name__ == "__main__":
    from tests import test_finite_differences
    test_finite_differences.launch_all_tests()

