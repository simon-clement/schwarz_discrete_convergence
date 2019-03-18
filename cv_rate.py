import numpy as np

H_DEFAULT = 1e-4
D_DEFAULT = 1.0
A_DEFAULT = 0.0
C_DEFAULT = 0.0

"""
    Solve the equation with full system:
    \partial t u* + Y* u* = f*
    and return u* at each time step.

    f_star_0 is the value of f* in x0.
    The other values of f_star are deduced from f1 and f2

    This function should be useless in the future but it is a good example
    of how to use the integrator in time.
"""
def solve_u_time_domain(u1_init, u2_init,
                        f_star_0, f1, f2,
                      Lambda_1, Lambda_2,
                      D1, D2,
                      h1, h2,
                      a, c, dt, number_time_steps):
    assert type(float(f_star_0)) == float
    assert type(float(dt)) == float
    assert u1_init.ndim == u2_init.ndim == f1.ndim == f2.ndim == 1
    M1 = u1_init.shape[0]
    M2 = u2_init.shape[0]
    assert f1.shape[0] == M1 and f2.shape[0] == M2
    assert type(float(a)) == float
    assert type(float(c)) == float
    assert type(float(Lambda_1)) == float
    assert type(float(Lambda_2)) == float
    # Broadcasting of h and D:
    h1 = np.ones(M1-1) * h1
    h2 = np.ones(M2-1) * h2
    D1 = np.ones(M1-1) * D1
    D2 = np.ones(M2-1) * D2
    assert (h1 < 0).all()
    assert (h2 > 0).all()
    assert (D1 > 0).all()
    assert (D2 > 0).all()
    assert h1.ndim == D1.ndim == h2.ndim == D2.ndim == 1

    # Compute Y matrices:
    M_star = M1 + M2 - 1
    h_star = np.concatenate((-h1[::-1], h2))
    D_star = np.concatenate((D1[::-1], D2))
    Y_star = get_Y_star(M_star=M_star, h_star=h_star, D_star=D_star, a=a, c=c)

    f_star = np.concatenate(([f1[-1]],-f1[-2:0:-1], [f_star_0], f2[1:]))
    #we use -f1 because f1 is f*(h^1_m + h^1_{m-1}) and h^1<0
    u0_star = np.concatenate((u1_init[:0:-1], u2_init))
    f0_on_time = np.array([f1[-1] for _ in range(number_time_steps)])

    return integration(u0_star, Y_star, f_star, f0_on_time, dt, 1)

"""
    Performs integration in time with a theta-shema.
    u0 is the initial condition
    Y is the matrix returned by get_Y or get_Y_star, such that
    \partial t u + Yu = f
    f cannot evolve in time except its first coordinate f_x0:
    f must have the shape of u
    f_x0 must be a 1-D array
    the number of time_steps is (f_x0.shape[0] - 1)
"""
def integration(u0, Y, f_const, f_x0, dt, theta):
    f = np.copy(f_const)
    assert Y[0].ndim == Y[1].ndim == Y[2].ndim == 1
    assert f_const.ndim == f.ndim == u0.ndim == 1
    M = Y[1].shape[0]
    assert f_const.shape[0] == u0.shape[0] == M
    assert Y[0].shape[0] == Y[2].shape[0] == M - 1
    assert 0 <= theta <= 1

    u = np.empty((f_x0.shape[0], u0.shape[0]))
    u[0] = u0
    if theta != 0:
        for step in range(f_x0.shape[0] - 1):
            f[0] = (1-theta) * f_x0[step] + theta * f_x0[step+1]
            Y_implicit = (theta*Y[0], theta*Y[1] + np.ones(M)/dt, theta*Y[2])
            Yu = multiply(Y, u[step])
            u[step+1] = solve_linear(Y_implicit,
                                          f + u[step] / dt - (1-theta)*Yu)
    else: #explicit case:
        for step in range(f_x0.shape[0] - 1):
            f[0] = f_x0[step]
            Yu = multiply(Y, u[step])
            u[step+1] = u[step] + dt*(f - Yu)
    return u

"""
    Returns "Y * u"
    equivalent code :
    return (np.diag(Y[1])+np.diag(Y[0], k=-1) + np.diag(Y[2], k=1)) @ u
    Y is a tridiagonal matrix returned by get_Y or get_Y_star
"""
def multiply(Y, u):
    assert len(Y) == 3
    assert u.ndim == Y[0].ndim == Y[1].ndim == Y[2].ndim == 1
    assert Y[1].shape[0] == Y[0].shape[0] + 1 == u.shape[0]
    assert Y[0].shape[0] == Y[2].shape[0]
    return np.concatenate(([0], Y[0]*u[:-1])) + Y[1] * u + \
            np.concatenate((Y[2]*u[1:], [0]))


"""
    Solve the linear system Yu = f and returns u.
    This function is just a wrapper over scipy
    Y is a tuple (Y_0, Y_1, Y_2) containing respectively:
    - The left diagonal of size M-1
    - The main diagonal of size M
    - The right diagonal of size M-1
    f is an array of size M.
    f[0] is the condition on the bottom of the domain
    f[-1] is the condition on top of the domain
    /!\ f[1:-1] should be equal to f * (hm + hmm1) /!\

    Y is returned by the functions get_Y and get_Y_star
"""
def solve_linear(Y, f):
    # We note Y_1 the main diagonal, Y_2 the right diag, Y_0 the left diag
    Y_0, Y_1, Y_2 = Y
    assert Y_0.ndim == Y_1.ndim == Y_2.ndim == f.ndim == 1
    assert Y_1.shape[0] - 1 == Y_2.shape[0] == Y_0.shape[0] == f.shape[0] - 1

    # solve_banded function requires to put the diagonals in the following form:
    Y_2 = np.concatenate(([0], Y_2))
    Y_0 = np.concatenate((Y_0, [0]))
    Y = np.vstack((Y_2, Y_1, Y_0))
    from scipy.linalg import solve_banded
    return solve_banded((1, 1), Y, f)

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

if __name__ == "__main__":
    import tests.test_linear_sys
    import tests.test_schwarz
    tests.test_linear_sys.launch_all_tests()
    tests.test_schwarz.launch_all_tests()

