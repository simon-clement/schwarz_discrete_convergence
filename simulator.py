#!/usr/bin/python3
"""
    This module aims to provide functions to simulate 1D atmosphere column toy model
"""
import numpy as np
import matplotlib.pyplot as plt


def stationary_case_MOST(z, H_cfl, H_bl, NUMBER_ITERATION=10):
    """
        Parameters: z: numpy array, first should be 0 and last is h_{bl}
        levels of grid points at full levels
        H_cfl is the height of the constant flux layer

        u_star is prescribed for now, but it may change in the future
    """
    assert z[0] == 0
    M = z.shape[0]
    h_bl = z[-1]
    z_half = (z[1:] + z[:-1]) / 2
    h_full = z[1:] - z[:-1]  # distance between full levels
    # since it is around z_{k+1/2}, we note it h_{k+1/2}
    h_half = (h_full[1:] + h_full[:-1])/2  # distance between half levels
    # since it is around z_k, we note it h_k
    f_coriolis = 1e-4  # Coriolis parameter
    R_0, alpha, xi_0, xi_1, rho, C_p = 1e-5, 0.2, 12., 15., 1., 1e-3

    rhs = R_0*(-alpha*np.exp(-z/xi_0)/xi_0 + -(1-alpha)
               * np.exp(-z/xi_1)/xi_1) / (rho*C_p)
    # geostrophy = 10.
    # rhs = 1j*f_coriolis*np.ones_like(z) * geostrophy
    # rhs[-1] = geostrophy
    # rhs[0] = 0.
    rhs[-1] = 0.
    K_mol = 1e-5
    C_m = 0.0667
    c_eps = 0.7
    kappa = 0.4
    C_D = 1e-3

    u = np.zeros_like(z)  # first guess is u(z)=0
    z_0 = .1
    l = np.sqrt(np.abs(z_half - H_cfl) *
                np.maximum(H_bl-z_half, np.zeros_like(z_half)))
    # abs in the previous function is only here to prevent sqrt from having negative input
    # the case z_half<H_cfl is taken care of in the next line
    l_MOST = (c_eps*C_m)**(1/4) * kappa*(z_half[z_half < H_cfl]+z_0)/C_m
    z_half_above_cfl = z_half[z_half >= H_cfl]
    l = np.concatenate((l_MOST, l_MOST[-1]*np.linspace(1, 0, z_half[z_half >= H_cfl].shape[0])**10
                        + np.sqrt((z_half[z_half >= H_cfl] - H_cfl)*np.maximum(H_bl-z_half[z_half >= H_cfl], 0.))))
    error = []
    from bisect import bisect
    index_10m = bisect(z, 10.)
    for ITERATION in range(NUMBER_ITERATION):
        K_half = l**2 * np.abs((u[1:] - u[:-1])/h_full) * \
            C_m * np.sqrt(C_m/c_eps) + K_mol
        # u_star = np.sqrt(C_D) * np.abs(u[index_10m])
        # np.copyto(K_half, kappa*u_star*(z_half+z_0), where=z_half<H_cfl) # MOST part
        # Implicit part of equation is:
        # 1j*f_coriolis * u_j - 1/h_k * (K_half[1:] * d_kp1_2 - K_half[:-1] * dkm1_2) = rhs
        # where d_kp1_2 = (u_jp1 - u_j) / (h_half[1:])
        # where d_km1_2 = (u_j - u_jm1) / (h_half[:-1])
        diagonal = 1j*f_coriolis - 1/h_half * \
            (K_half[1:] * (-1)/h_full[1:] - K_half[:-1] / h_full[:-1])
        upper_diag = - 1/h_half * K_half[1:] / h_full[1:]
        lower_diag = - 1/h_half * (- K_half[:-1] * (-1) / h_full[:-1])

        # boundary conditions:
        diagonal = np.concatenate(([1j*f_coriolis + K_half[0] / h_full[0]], diagonal, [1]))
        upper_diag = np.concatenate(([-K_half[0] / h_full[0]], upper_diag)) # (K d) at interface is set to 0
        lower_diag = np.concatenate((lower_diag, [0]))
        from utils_linalg import solve_linear
        new_u = solve_linear((lower_diag, diagonal, upper_diag), rhs)
        error += [np.linalg.norm(new_u - u)]
        u = new_u

    return error, u, K_half, l


def stationary_case(z, NUMBER_ITERATION=10):
    """
        Parameters: z: numpy array, first should be 0 and last is h_{bl}
        levels of grid points at full levels

        u_star is prescribed for now, but it may change in the future
    """
    assert z[0] == 0
    M = z.shape[0]
    h_bl = z[-1]
    l = 200.
    z_half = (z[1:] + z[:-1]) / 2
    h_full = z[1:] - z[:-1]  # distance between full levels
    # since it is around z_{k+1/2}, we note it h_{k+1/2}
    h_half = (h_full[1:] + h_full[:-1])/2  # distance between half levels
    # since it is around z_k, we note it h_k
    f_coriolis = 1e-4  # Coriolis parameter
    R_0, alpha, xi_0, xi_1, rho, C_p = 1e-5, 0.2, 12., 15., 1., 1e-3

    rhs = R_0*(-alpha*np.exp(-z/xi_0)/xi_0 + -(1-alpha)
               * np.exp(-z/xi_1)/xi_1) / (rho*C_p)
    # geostrophy = 10.
    # rhs = 1j*f_coriolis*np.ones_like(z) * geostrophy
    # rhs[-1] = geostrophy
    # rhs[0] = 0.
    rhs[0] = rhs[-1] = 0.
    K_mol = 1e-5

    u = np.copy(z)  # first guess is u(z)=identity

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
        upper_diag = - 1/h_half * K_half[1:] / h_full[1:]
        lower_diag = - 1/h_half * (- K_half[:-1] * (-1) / h_full[:-1])

        # boundary conditions:
        diagonal = np.concatenate(([1], diagonal, [1]))
        upper_diag = np.concatenate(([0], upper_diag))
        lower_diag = np.concatenate((lower_diag, [0]))
        from utils_linalg import solve_linear
        new_u = solve_linear((lower_diag, diagonal, upper_diag), rhs)
        error += [np.linalg.norm(new_u - u)]
        u = new_u

    return error, u, K_half

def stationary_case_KPP(z, H_cfl, H_bl, NUMBER_ITERATION=10, TKE_instead=True):
    """
        Parameters: z: numpy array, first should be >0 and last is >h_{bl}.
            height where the collocation points u are computed.
        H_cfl is the height of the constant flux layer
        h_bl is the height of the planetary boundary layer (nothing interesting above)

        viscosity K_kpp is K(z) = \kappa u* z(1 - z/h_{bl})^2 + K_mol
        if TKE_instead is True, then 1 prognostic equation is used for TKE

        no forcing leads to no interesting profiles.
    """
    assert z[0] > 0
    M = z.shape[0]
    # u, theta, e are computed on z levels
    z_u = z
    # the viscosities and the derivatives are in z_visc: starting at 0
    z_visc = np.concatenate(([0],  (z[1:] + z[:-1]) / 2))
    h_u = z_u[1:] - z_u[:-1]  # distance between full levels
    # since it is around z_{k+1/2}, we note it h_{k+1/2}
    h_visc = z_visc[1:] - z_visc[:-1] # distance between visc levels
    # since it is around z_k, we note it h_k
    f_coriolis = 0. # 1e-4  # Coriolis parameter
    R_0, alpha, xi_0, xi_1, rho, C_p = 1e-5, 0.2, 12., 15., 1., 1e-3

    rhs = np.ones_like(R_0*(-alpha*np.exp(-z/xi_0)/xi_0 + -(1-alpha)
               * np.exp(-z/xi_1)/xi_1) / (rho*C_p))
    # geostrophy = 10.
    # rhs = 1j*f_coriolis*np.ones_like(z) * geostrophy
    # rhs[-1] = geostrophy
    # rhs[0] = 0.
    rhs[-1] = 0.
    K_mol = 1e-5
    C_m = 0.0667
    c_eps = 0.7
    kappa = 0.4

    u = z  # first guess is u(z)=identity
    z_0 = .1

    ############# Compute mixing length l (unused) ################
    l = np.sqrt(np.abs(z_visc - H_cfl) *
                np.maximum(H_bl-z_visc, np.zeros_like(z_visc)))

    # abs in the previous function is only here to prevent sqrt from having negative input
    # the case z_visc<H_cfl is taken care of in the next line
    l_MOST = (c_eps*C_m)**(1/4) * kappa*(z_visc[z_visc < H_cfl]+z_0)/C_m
    l = np.concatenate((l_MOST, l_MOST[-1]*np.linspace(1, 0, z_visc[z_visc >= H_cfl].shape[0])**10
                        + np.sqrt((z_visc[z_visc >= H_cfl] - H_cfl)*np.maximum(H_bl-z_visc[z_visc >= H_cfl], 0.))))
    ######### mixing length l is finally set to 1 everywhere
    # l[:] = 400
    
    # profile function of KPP
    G_z = kappa * z_visc * (1 - z_visc/H_bl)**2
    G_z[z_visc > H_bl] = 0.
    u_star = 1. # friction scale
    K_kpp = u_star * G_z + K_mol
    # first guess of TKE
    TKE = np.ones_like(z_u)
    K_neutral_TKE = K_kpp
    flux_0 = 0

    error = []
    from bisect import bisect
    from utils_linalg import solve_linear
    index_10m = bisect(z_u, 10.)
    for ITERATION in range(NUMBER_ITERATION):
        #### TKE computation ########
        K = K_neutral_TKE
        mixing_length_diagonal = (l[2:] + l[1:-1])/2
        diagonal = - 1/h_visc[1:] * (K[2:] * (-1)/h_u[1:] - K[1:-1] / h_u[:-1])
        upper_diag = - 1/h_visc[1:] * K[2:] / h_u[1:]
        lower_diag = - 1/h_visc[1:] * (- K[1:-1] * (-1) / h_u[:-1])

        # boundary conditions: e has Dirichlet conditions on both sides
        diagonal = np.concatenate(([1.], diagonal, [1]))
        upper_diag = np.concatenate(([0], upper_diag))
        lower_diag = np.concatenate((lower_diag, [0]))

        # wind shear is K||dzu||^2, aligned with K.
        wind_shear = K[1:] * (np.abs(u[1:] - u[:-1])/h_u)**2

        rhs_TKE = np.concatenate(([(np.abs(flux_0)**2 / K[0]*l[0]/c_eps)**(2/3)],
                                  (wind_shear[1:] + wind_shear[:-1])/2. \
                                          - c_eps * TKE[1:-1]**(3/2) / mixing_length_diagonal, [0]))
        TKE = solve_linear((lower_diag, diagonal, upper_diag), rhs_TKE)
        TKE = np.maximum(TKE, 1e-10)
        K_neutral_TKE = K_mol + C_m * l * np.concatenate(([np.sqrt(TKE[0])],
                                                np.sqrt((TKE[1:] + TKE[-1:])/2)))


        if TKE_instead:
            K = K_neutral_TKE
        else:
            K = K_kpp

        diagonal = 1j*f_coriolis - 1/h_visc[1:] * \
            (K[2:] * (-1)/h_u[1:] - K[1:-1] / h_u[:-1])
        upper_diag = - 1/h_visc[1:] * K[2:] / h_u[1:]
        lower_diag = - 1/h_visc[1:] * (- K[1:-1] * (-1) / h_u[:-1])

        # boundary conditions:
        diagonal = np.concatenate(([1j*f_coriolis + K[0] / h_u[0]], diagonal, [1]))
        upper_diag = np.concatenate(([-K[0] / h_u[0]], upper_diag)) # (K d) at interface is set to 0
        lower_diag = np.concatenate((lower_diag, [0]))
        rhs[0] = flux_0
        new_u = solve_linear((lower_diag, diagonal, upper_diag), rhs)
        error += [np.linalg.norm(new_u - u)]
        u = new_u

    return error, u, K
