import numpy as np

H_DEFAULT = 1e-4
D_DEFAULT = 1.0
A_DEFAULT = 0.0
C_DEFAULT = 0.0

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

    /!\ WARNING : DUPLICATE CODE between get_Y_star and get_Y (for lisibility)

"""
def get_Y(M, Lambda, h=H_DEFAULT, D=D_DEFAULT, a=A_DEFAULT,
                        c=C_DEFAULT, upper_domain=True):
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
    Y_1[0] = Lambda - D[0] / h[0] # Robin bd conditions at interface
    Y_1[M-1] = 1 # Neumann bd conditions for \Omega_2, Dirichlet for \Omega_1
    if upper_domain:
        Y_1[M-1] /= h[-1]

    ######## RIGHT DIAGONAL
    Y_2 = np.empty(M-1)
    Y_2[1:] = -2 * D_mp1_2 / h_m
    Y_2[0] = D[0] / h[0]
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
                arg.ndim == 1 and arg.shape[0] == M-1
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
