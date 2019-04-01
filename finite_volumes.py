"""
    Finite volume for diffusion advection reaction equation.
    The functions to use are integrate_one_step and integrate_one_step_star.
    Theses function make a single step in time. It is not efficient
    (need to compute the matrix each time) but it is simple.
"""
import numpy as np
from utils_numeric import solve_linear


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
    D = np.zeros(M+1) + D
    h = np.zeros(M) + h
    f = np.zeros(M) + f

    assert type(u_nm1) == np.ndarray and u_nm1.ndim == 1 and u_nm1.shape[0] == M
    assert upper_domain is True or upper_domain is False

    if not upper_domain: # from now h, D, f, u_nm1 are 0 at bottom of the ocean.
        h, D, f, u_nm1 = np.flipud(h), np.flipud(D), np.flipud(f), \
                np.flipud(u_nm1)

    Y = get_Y(M=M, h=h, D=D, a=a, c=c, dt=dt,
            Lambda=Lambda, upper_domain=upper_domain)

    rhs = dt / (1+dt*c) * (f[1:] - f[:-1] + (u_nm1[1:] - u_nm1[:-1]) /dt)
    if upper_domain: # Neumann condition: user give derivative but I need flux
        cond_robin = Lambda * u_interface + phi_interface \
                - Lambda * dt /(1+dt*c) * (f[0] + u_nm1[0]/dt)

        cond_M = bd_cond * D[-1]
        rhs = np.concatenate(([cond_robin], rhs, [cond_M]))
    else: # Dirichlet condition: user gives value, rhs is more complicated
        cond_robin = Lambda * u_interface + phi_interface \
                - Lambda * dt /(1+dt*c) * (f[-1] + u_nm1[-1]/dt)

        cond_M = bd_cond - dt / (1+dt*c) * (f[0] + u_nm1[0]/dt)
        rhs = np.concatenate(([cond_M], rhs, [cond_robin]))

    phi_ret = solve_linear(Y, rhs)
    d = phi_ret / D # We take the derivative of u

    d_kp1 = d[1:]
    d_km1 = d[:-1]
    D_mp1_2 = D[1:]
    D_mm1_2 = D[:-1]

    u_n = dt / (1+dt*c) * ( f + u_nm1 / dt \
            + (D_mp1_2*d_kp1 - D_mm1_2*d_km1)/h \
            - a * (d_kp1 + d_km1) / 2 )

    assert u_n.shape[0] == M
    if upper_domain:
        u_interface = u_n[0] - h[0]*d[1]/6 - h[0]*d[0]/3
        phi_interface = phi_ret[0]
    else:
        u_interface = u_n[-1] + h[-1]*d[-2]/6 + h[-1]*d[-1]/3
        phi_interface = phi_ret[-1]
        u_n = np.flipud(u_n)

    return u_n, u_interface, phi_interface

"""
    Same as integrate_one_step, but with full domain. The parameters are
    more or less the same

    Provided equation parameters (M, h, D, f){1,2}, a, c, dt;
    Provided boundary condition neumann, dirichlet
    Provided former state of the equation u_nm1;
    Returns (u_n, u_interface, phi_interface)
    u_n is the next state vector, {u, phi}_interface are the
    values of the state vector at interface,
    necessary to compare with Robin conditions.

    neumann is the Neumann condition of the top of the atmosphere.
    dirichlet is the Dirichlet condition of the bottom of the ocean.
    h{1,2}, D{1,2}, f{1,2} have their first values ([0,1,..]) at the interface
    and their last values ([..,M-2, M-1]) at 
    the top of the atmosphere (h2, D2, f2) or bottom of the ocean (f1, D1, f1)
    u_nm1[0] is the bottom of the ocean 
    and u_nm1[M1 + M2 - 1] is the top of the atmosphere

    M{1, 2} is int
    a, c, dt, neumann, dirichlet are float
    h{1,2}, D{1,2}, f{1,2} can be float or np.ndarray of dimension 1.
    if h is a ndarray: its size must be M{1,2}
    if f is a ndarray: its size must be M{1,2}
    if D is a ndarray: its size must be M+1
    u_nm1 must be a np.ndarray of dimension 1 and size M

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
    assert type(u_nm1) == np.ndarray and u_nm1.ndim == 1 and u_nm1.shape[0] == M1+M2

    assert type(M2) is int and type(M1) is int
    assert M2 > 0 and M1 > 0
    if type(D1) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D1 = float(D1)
    if type(D2) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D2 = float(D2)
    for arg, name in zip((h1, h2, D1, D2), ("h1", "h2", "D1", "D2")):
        assert type(arg) is float or \
                type(arg) is np.float64 or \
                type(arg) is np.ndarray and \
                arg.ndim == 1, name
        assert (np.array(arg) > 0).all(), name + " is negative or 0 !"

    #Broadcasting / verification of types:
    D1 = np.zeros(M1+1) + D1
    D2 = np.zeros(M2+1) + D2
    h1 = np.zeros(M1) + h1
    h2 = np.zeros(M2) + h2
    f1 = np.zeros(M1) + f1
    f2 = np.zeros(M2) + f2

    # Flipping arrays to have [0] at low altitudes rather than interface
    D1 = np.flipud(D1)
    h1 = np.flipud(h1)
    f1 = np.flipud(f1)
    Y = get_Y_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1, D2=D2, a=a, c=c, dt=dt)

    f = np.concatenate((f1, f2))

    rhs = dt / (1+dt*c) * (f[1:] - f[:-1] + (u_nm1[1:] - u_nm1[:-1]) /dt)
    dirichlet = dirichlet - dt / (1+dt*c) * (f[0] + u_nm1[0]/dt)
    neumann = neumann * D2[-1]
    rhs = np.concatenate(([dirichlet], rhs, [neumann]))

    phi_ret = solve_linear(Y, rhs)

    d1 = phi_ret[:M1+1] / D1 # we go until interface
    d2 = phi_ret[M1:] / D2 # we start from interface

    d1_kp1 = d1[1:]
    d2_kp1 = d2[1:]
    d1_km1 = d1[:-1]
    d2_km1 = d2[:-1]
    D1_kp1_2 = D1[1:]
    D1_km1_2 = D1[:-1]
    D2_kp1_2 = D2[1:]
    D2_km1_2 = D2[:-1]

    u1_n = dt / (1+dt*c) * ( f[:M1] + u_nm1[:M1] / dt \
            + (D1_kp1_2*d1_kp1 - D1_km1_2*d1_km1)/ h1 \
            - a * (d1_kp1 + d1_km1) / 2 )

    u2_n = dt / (1+dt*c) * ( f[M1:] + u_nm1[M1:] / dt \
            + (D2_kp1_2*d2_kp1 - D2_km1_2*d2_km1)/h2 \
            - a * (d2_kp1 + d2_km1) / 2 )

    assert u1_n.shape[0] == M1
    assert u2_n.shape[0] == M2

    u2_interface = u2_n[0] - h2[0]*d2[1]/6 - h2[0]*d2[0]/3
    u1_interface = u1_n[-1] + h1[-1]*d1[-2]/6 + h1[-1]*d1[-1]/3
    u1_bottom = u1_n[0] - h1[0]*d1[1]/6 - h1[0]*d1[0]/3
    assert abs(u1_interface - u2_interface) < 1e-5
    phi_interface = phi_ret[M1]

    return np.concatenate((u1_n, u2_n)), u1_interface, phi_interface

"""
    Returns the tridiagonal matrix Y in the shape asked by solve_linear.
    Y is the matrix we need to inverse to solve one of the half-domains.
    This function is useful to compute u_{1,2}, solution of Yu=f

    (Does not actually return a np matrix, it returns (Y_0, Y_1, Y_2).
    For details, see the documentation of utils_numeric.solve_linear)

    The returned matrix is of dimension MxM
    To compare with the full system:
        u*[0:M] = u1[0:M] (i.e. for all m, u*[m] = u1[m]
        u*[M-1:2M-1] = u2[0:M] (i.e. for all m, u*[M + m] = u2[m]

    h and D have their first values ([0]) for low altitudes
    (bottom of the ocean for Omega_1, interface for Omega_2)
    and their last values ([..M-2, M-1]) for high altitudes
    (interface for Omega_1, top of the atmosphere for Omega_2)
    /!\ it is different than in the notations or in integrate_* functions.

    h: step size (always positive) (float or ndarray, size: M)
    D: diffusivity (always positive) (float or ndarray, size: M+1)
        Note: if D is a np.ndarray, it should be given on the half-steps,
                i.e. D[m] is D_{m+1/2}
    a: advection coefficient (should be positive) (float)
    c: reaction coefficient (should be positive) (float)
"""

def get_Y(M, h, D, a, c, dt, Lambda, upper_domain=True):
    a, c, dt, Lambda = float(a), float(c), float(dt), float(Lambda)
    D = np.zeros(M+1) + D
    h = np.zeros(M) + h
    assert type(M) is int
    assert upper_domain is True or upper_domain is False

    # We first use our great function get_Y_star:
    if upper_domain:
        Y_0, Y_1, Y_2 = get_Y_star(M1=1, M2=M,
                h1=1.0, h2=h, D1=D[0], D2=D, a=a, c=c, dt=dt)
        Y_0 = Y_0[1:]
        Y_1 = Y_1[1:]
        Y_2 = Y_2[1:]

        # Now we have the tridiagonal matrices, except for the Robin bd condition
        dirichlet_cond_extreme_point = -dt/(1+dt*c) * (1/h[0] +a/(2*D[0])) - h[0]/(3*D[0])
        dirichlet_cond_interior_point = dt/(1+dt*c) * (1/h[0] -a/(2*D[1])) - h[0]/(6*D[1])
        # Robin bd condition are Lambda * Dirichlet + Neumann:
        # Except we work with fluxes:
        # Neumann condition is actually a Dirichlet bd condition
        # and Dirichlet is just a... pseudo-differential operator
        Y_1[0] = Lambda * dirichlet_cond_extreme_point + 1
        Y_2[0] = Lambda * dirichlet_cond_interior_point
    else:
        Y_0, Y_1, Y_2 = get_Y_star(M1=M, M2=1,
                h1=h, h2=1.0, D1=D, D2=D[0], a=a, c=c, dt=dt)
        # Here Y_0 and Y_2 are inverted because we need to take the symmetric
        Y_0 = Y_0[:-1]
        Y_1 = Y_1[:-1]
        Y_2 = Y_2[:-1]
        # Now we have the tridiagonal matrices, except for the Robin bd condition
        dirichlet_cond_extreme_point = dt/(1+dt*c) * (1/h[-1] -a/(2*D[-1])) + h[-1]/(3*D[-1])
        dirichlet_cond_interior_point = dt/(1+dt*c) * (-1/h[-1] -a/(2*D[-2])) + h[-1]/(6*D[-2])
        # Robin bd condition are Lambda * Dirichlet + Neumann:
        # Except we work with fluxes:
        # Neumann condition is actually a Dirichlet bd condition
        # and Dirichlet is just a... pseudo-differential operator
        Y_1[-1] = Lambda * dirichlet_cond_extreme_point + 1
        Y_0[-1] = Lambda * dirichlet_cond_interior_point
        # We take the flipped, symmetric of the matrix:
    return (Y_0, Y_1, Y_2)

"""
    Returns the tridiagonal matrix Y* in the shape asked by solve_linear.
    Y* is the matrix we need to inverse to solve the full domain.
    This function is useful to compute u*, solution of Y*u*=f*
    It is also used in get_Y, with one of the arguments M_{1 | 2} = 1

    (Does not actually return a np matrix, it returns (Y_0, Y_1, Y_2).
    For details, see the documentation of utils_numeric.solve_linear)

    The returned matrix is of dimension M_starxM_star
    To compare with the coupled system and get:
        u*[0:M] = u1[0:M] (i.e. for all m, u*[m] = u1[m]
        u*[M-1:2M-1] = u2[0:M] (i.e. for all m, u*[M + m] = u2[m]
    We should have:
        - M_star = M1+M2 - 1
        - D_star[0:M+1] = D1[0:M+1]
        - h_star[0:M] = h1[0:M]
        - D_star[M-1:2M-1] = D2[0:M]
        - h_star[M-1:2M-1] = h2[0:M]

    D1[-1] and D2[0] should both be the diffusivity at interface.
    (2 values because it can be discontinuous, so D1[-1] is D(0^{-}) )

    h{1,2}: step size (always positive) (float or ndarray, size: M{1,2})
    D{1,2}: diffusivity (always positive) (float or ndarray, size: M{1,2}+1)
        Note: if D{1,2} is a np.ndarray, it should be given on the half-steps,
                i.e. D{1,2}[m] is D{1,2}_{m+1/2}
    a{1,2}: advection coefficient (should be positive) (float)
    c{1,2}: reaction coefficient (should be positive) (float)

"""
def get_Y_star(M1, M2, h1, h2, D1, D2, a, c, dt):
    a, c, dt = float(a), float(c), float(dt)
    if a < 0:
        print("Warning : a should probably not be negative")
    if c < 0:
        print("Warning : c should probably not be negative")

    assert type(M2) is int and type(M1) is int
    assert M2 > 0 and M1 > 0
    if type(D1) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D1 = float(D1)
    if type(D2) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D2 = float(D2)
    for arg, name in zip((h1, h2, D1, D2), ("h1", "h2", "D1", "D2")):
        assert type(arg) is float or \
                type(arg) is np.float64 or \
                type(arg) is np.ndarray and \
                arg.ndim == 1, name
        assert (np.array(arg) > 0).all(), name + " is negative or 0 !"

    # Broadcast or verification of size:
    D1 = np.zeros(M1+1) + D1
    h1 = np.zeros(M1) + h1
    D2 = np.zeros(M2+1) + D2
    h2 = np.zeros(M2) + h2
    # In the notations h is negative when considering \omega_1:

    #assert (h1 == h1[::-1]).all() # Warning with this,
    # it means they are constant... The code is not purely consistant
    # with itself though, so for now let's keep theses assertions
    #assert (D1 == D1[::-1]).all()
    #D1 = D1[::-1]

    # D_minus means we take the value of D1 for the interface
    D_minus = np.concatenate((D1[1:], D2[1:-1]))
    # D_plus means we take the value of D1 for the interface
    D_plus = np.concatenate((D1[1:-1], D2[:-1]))
    # D_mm1_2 is a D_plus means we take the value of D1 for the interface
    D_mm1_2 = np.concatenate((D1[:-1], D2[:-2]))
    # D_mm1_2 is a D_minus means we take the value of D1 for the interface
    D_mp3_2 = np.concatenate((D1[2:], D2[1:]))
    h = np.concatenate((h1, h2))
    h_m = h[:-1]
    h_mp1 = h[1:]

    ######## LEFT DIAGONAL: phi_{1/2} -> phi_{M+1/2}
    Y_0 = -dt/(1+dt*c) * (1/h_m + a/(2*D_mm1_2)) + h_m/(6*D_mm1_2)
    # Now we can put Neumann bd condition:
    Y_0 = np.concatenate((Y_0, [0])) # Neumann bd condition 
    # (actually Dirichlet bd because we work on the fluxes...)

    ######## RIGHT DIAGONAL:
    Y_2 = -dt/(1+dt*c) * (1/h_mp1 - a/(2*D_mp3_2)) + h_mp1/(6*D_mp3_2)
    Y_2_bd = dt/(1+dt*c) * (1/h[0] - a/(2*D1[1])) - h[0]/(6*D1[1])
    Y_2 = np.concatenate(([Y_2_bd], Y_2))

    ######## MAIN DIAGONAL:
    Y_1 = dt/(1+dt*c) * (1 / h_m + 1/h_mp1 + a/2 * (1/D_plus - 1/D_minus)) \
            + (h_m/D_minus + h_mp1/D_plus)/3
    Y_1_bd = -dt/(1+dt*c) * (1/h[0] + a/(2*D1[0])) - h[0]/(3*D1[0])
    Y_1 = np.concatenate(([Y_1_bd], Y_1, [1]))

    assert Y_1.shape[0] == M1 + M2 + 1
    assert Y_0.shape[0] == M1 + M2 == Y_2.shape[0]

    return (Y_0, Y_1, Y_2)


if __name__ == "__main__":
    from tests import test_finite_volumes
    test_finite_volumes.launch_all_tests()

