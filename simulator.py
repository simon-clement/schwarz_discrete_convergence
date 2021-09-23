#!/usr/bin/python3
"""
    This module aims to provide functions to simulate 1D atmosphere column toy model
"""
import numpy as np
import matplotlib.pyplot as plt

def stationary_case(z, NUMBER_ITERATION=10):
    """
        Parameters: z: numpy array, first should be 0 and last is h_{bl}
        levels of grid points at full levels

        viscosity K is K(z) = \kappa u* z/h_{bl} (h_{bl} - z) + K_mol
        u_star is prescribed for now, but it may change in the future
    """
    assert z[0] == 0
    M = z.shape[0]
    h_bl = z[-1]
    l = 200.
    z_half = (z[1:] + z[:-1] ) / 2
    h_full = z[1:] - z[:-1] # distance between full levels
    # since it is around z_{k+1/2}, we note it h_{k+1/2}
    h_half = (h_full[1:] + h_full[:-1])/2 # distance between half levels
    # since it is around z_k, we note it h_k
    f_coriolis = 1e-4 # Coriolis parameter
    R_0, alpha, xi_0, xi_1, rho, C_p = 1e-5, 0.2, 12., 15., 1., 1e-3

    rhs = R_0*(-alpha*np.exp(-z/xi_0)/xi_0 + -(1-alpha)*np.exp(-z/xi_1)/xi_1) / (rho*C_p)
    # geostrophy = 10.
    # rhs = 1j*f_coriolis*np.ones_like(z) * geostrophy
    # rhs[-1] = geostrophy
    # rhs[0] = 0.
    rhs[0] = rhs[-1] = 0.
    K_mol = 1e-5

    u = np.copy(z) # first guess is u(z)=identity

    error = []
    for ITERATION in range(NUMBER_ITERATION):
        H_sl = 50.
        l = np.sqrt(z_half*np.maximum(H_sl-z_half, np.zeros_like(z_half)))
        K_half = l**2 * np.abs((u[1:] - u[:-1])/h_full) + K_mol
        # Implicit part of equation is: 
        # 1j*f_coriolis * u_j - 1/h_k * (K_half[1:] * d_kp1_2 - K_half[:-1] * dkm1_2) = rhs
        # where d_kp1_2 = (u_jp1 - u_j) / (h_half[1:])
        # where d_km1_2 = (u_j - u_jm1) / (h_half[:-1])
        diagonal = 1j*f_coriolis - 1/h_half * \
                (K_half[1:] * (-1)/h_full[1:] - K_half[:-1] / h_full[:-1])
        upper_diag = - 1/h_half * K_half[1:] /h_full[1:]
        lower_diag = - 1/h_half *( - K_half[:-1] * (-1) / h_full[:-1])

        # boundary conditions:
        diagonal = np.concatenate(([1], diagonal, [1]))
        upper_diag = np.concatenate(([0], upper_diag))
        lower_diag = np.concatenate((lower_diag, [0]))
        from utils_linalg import solve_linear
        new_u = solve_linear((lower_diag, diagonal, upper_diag), rhs)
        error += [np.linalg.norm(new_u - u)]
        u = new_u
        
    return error, u, K_half
