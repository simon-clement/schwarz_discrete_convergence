import numpy as np
from tests import test_finite_volumes
from utils_numeric import solve_linear

def get_Y(M, h, D, a, c, dt, Lambda, upper_domain=True):
    # We first use our great function get_Y_star:
    if upper_domain:
        Y_0, Y_1, Y_2 = get_Y_star(M1=1, M2=M,
                h1=1.0, h2=h, D1=D[0], D2=D, a=a, c=c, dt=dt)
        Y_0 = Y_0[1:]
        Y_1 = Y_1[1:]
        Y_2 = Y_2[1:]
    else:
        Y_0, Y_1, Y_2 = get_Y_star(M1=M, M2=1,
                h1=h, h2=1.0, D1=D, D2=D[0], a=a, c=c, dt=dt)
        # Here Y_0 and Y_2 are inverted because we need to take the symmetric
        Y_2, Y_0 = Y_0[-2::-1], Y_2[-2::-1]
        Y_1 = Y_1[-2::-1]
    # Now we have the tridiagonal matrices, except for the Robin bd condition
    dirichlet_cond_extreme_point = -dt/(1+dt*c) * (1/h[0] +a/(2*D[0])) - h[0]/(3*D[0])
    dirichlet_cond_interior_point = dt/(1+dt*c) * (1/h[0] -a/(2*D[1])) - h[0]/(6*D[1])
    # Robin bd condition are Lambda * Dirichlet + Neumann:
    # Except we work with fluxes:
    # Neumann condition is actually a Dirichlet bd condition
    # and Dirichlet is just a... pseudo-differential operator
    Y_1[0] = Lambda * dirichlet_cond_extreme_point + 1
    Y_2[0] = Lambda * dirichlet_cond_interior_point
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
        - D_star[0:M] = D1[M-1::-1]
        - h_star[0:M] = h1[M-1::-1]
        - D_star[M-1:2M-1] = D2[0:M]
        - h_star[M-1:2M-1] = h2[0:M]

    D1[0] and D2[0] should both be the diffusivity at interface.

    The following parameters can either be given as floats or np.ndarray

    h: step size (always positive) (size: M-1)
    D: diffusivity (always positive) (size: M-1)
        Note: if D is a np.ndarray, it should be given on the half-steps,
                i.e. D[m] is D_{m+1/2}
    a: advection coefficient (should be positive) (size: M-2)
    c: reaction coefficient (should be positive) (size: M-2)

    /!\ WARNING : DUPLICATE CODE between get_Y_star and get_Y (for lisibility)

"""
def get_Y_star(M1, M2, h1, h2, D1, D2, a, c, dt):
    a = float(a)
    c = float(c)
    assert type(a) is float
    assert type(c) is float
    if type(h1) is int:
        print("Warning: type of step size is int. casting to float...")
        h1 = float(h1)
    if type(D1) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D1 = float(D1)
    assert type(M2) is int
    assert M2 > 0
    for arg in (h1, D1):
        assert type(arg) is float or \
                type(arg) is np.float64 or \
                type(arg) is np.ndarray and \
                arg.ndim == 1
    if (np.array(h1) < 0).any():
        print("Warning : h2 should never be negative")
    if (np.array(D1) < 0).any():
        print("Warning : D2 should never be negative")

    if type(h2) is int:
        print("Warning: type of step size is int. casting to float...")
        h2 = float(h2)
    if type(D2) is int:
        print("Warning: type of diffusivity is int. casting to float...")
        D2 = float(D2)
    assert type(M2) is int
    assert M2 > 0
    for arg in (h2, D2):
        assert type(arg) is float or \
                type(arg) is np.float64 or \
                type(arg) is np.ndarray and \
                arg.ndim == 1
    if (np.array(h2) < 0).any():
        print("Warning : h2 should never be negative")
    if (np.array(D2) < 0).any():
        print("Warning : D2 should never be negative")

    if a < 0:
        print("Warning : a should probably not be negative")
    if c < 0:
        print("Warning : c should probably not be negative")
    assert M1 > 0
    assert M2 > 0

    # Broadcast or verification of size:
    D1 = np.zeros(M1+1) + D1
    h1 = np.zeros(M1) + h1
    D2 = np.zeros(M2+1) + D2
    h2 = np.zeros(M2) + h2
    # In the notations h is negative when considering \omega_1:

    h1 = h1[::-1]
    D1 = D1[::-1]
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

def integrate_one_step_star(M1, M2, h1, h2, D1, D2, a, c, dt, f1, f2,
        neumann, dirichlet, u0):

    D1 = np.zeros(M1+1) + D1
    D2 = np.zeros(M2+1) + D2
    h1 = np.zeros(M1) + h1
    h2 = np.zeros(M2) + h2

    f1 = np.zeros(M1) + f1
    f2 = np.zeros(M2) + f2

    Y = get_Y_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1, D2=D2, a=a, c=c, dt=dt)

    D1 = D1[::-1]
    h1 = h1[::-1]
    f1 = f1[::-1]
    f = np.concatenate((f1, f2))

    rhs = dt / (1+dt*c) * (f[1:] - f[:-1] + (u0[1:] - u0[:-1]) /dt)
    dirichlet = dirichlet - dt / (1+dt*c) * (f[0] + u0[0]/dt)
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

    u1_np1 = dt / (1+dt*c) * ( f[:M1] + u0[:M1] / dt \
            + (D1_kp1_2*d1_kp1 - D1_km1_2*d1_km1)/ h1 \
            - a * (d1_kp1 + d1_km1) / 2 )

    u2_np1 = dt / (1+dt*c) * ( f[M1:] + u0[M1:] / dt \
            + (D2_kp1_2*d2_kp1 - D2_km1_2*d2_km1)/h2 \
            - a * (d2_kp1 + d2_km1) / 2 )

    assert u1_np1.shape[0] == M1
    assert u2_np1.shape[0] == M2

    u2_interface = u2_np1[0] - h2[0]*d2[1]/6 - h2[0]*d2[0]/3
    u1_interface = u1_np1[-1] + h1[-1]*d1[-2]/6 + h1[-1]*d1[-1]/3
    u1_bottom = u1_np1[0] - h1[0]*d1[1]/6 - h1[0]*d1[0]/3
    assert abs(u1_interface - u2_interface) < 1e-5
    phi_interface = phi_ret[M1]

    return np.concatenate((u1_np1, u2_np1)), u1_interface, phi_interface


def integrate_one_step(M, h, D, a, c, dt, f, bd_cond, Lambda, u_nm1,
        u_interface, phi_interface, upper_domain=True):
    assert type(u_nm1) == np.ndarray and u_nm1.ndim == 1 and u_nm1.shape[0] == M

    D = np.zeros(M+1) + D
    h = np.zeros(M) + h
    f = np.zeros(M) + f

    Y = get_Y(M=M, h=h, D=D, a=a, c=c, dt=dt,
            Lambda=Lambda, upper_domain=upper_domain)

    cond_robin = Lambda * u_interface + phi_interface \
            - Lambda * dt /(1+dt*c) * (f[0] + u_nm1[0]/dt)

    rhs = dt / (1+dt*c) * (f[1:] - f[:-1] + (u_nm1[1:] - u_nm1[:-1]) /dt)
    if upper_domain: # Neumann condition: user give derivative but I need flux
        cond_M = bd_cond * D[-1]
    else: # Dirichlet condition: user gives value, rhs is more complicated
        cond_M = bd_cond - dt / (1+dt*c) * (f[-1] + u_nm1[-1]/dt)

    rhs = np.concatenate(([cond_robin], rhs, [cond_M]))

    phi_ret = solve_linear(Y, rhs)

    d = phi_ret / D # We take the derivative

    d_kp1 = d[1:]
    d_km1 = d[:-1]
    D_kp1_2 = D[1:]
    D_km1_2 = D[:-1]

    u_np1 = dt / (1+dt*c) * ( f + u_nm1 / dt \
            + (D_kp1_2*d_kp1 - D_km1_2*d_km1)/h \
            - a * (d_kp1 + d_km1) / 2 )

    assert u_np1.shape[0] == M
    if upper_domain:
        u_interface = u_np1[0] - h[0]*d[1]/6 - h[0]*d[0]/3
    else: #TODO be sure...
        u_interface = u_np1[0] + h[0]*d[1]/6 + h[0]*d[0]/3
    phi_interface = phi_ret[0]

    return u_np1, u_interface, phi_interface


if __name__ == "__main__":
    test_finite_volumes.launch_all_tests()

