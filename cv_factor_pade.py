#!/usr/bin/python3
"""
    This module computes theoric convergence rates and
    provide functions to observe real convergence rate.
"""
import time
import functools
import numpy as np
from numpy import pi

#########################################################################
# THEORIC PART : RETURN RATES YIELDED BY ANALYSIS IN FREQUENTIAL DOMAIN #
#########################################################################
def default_gamma(z):
    b = 1+1/np.sqrt(2)
    return z - b*(z-1) - b/2 * (z-1)**2


def rho_Pade_c(builder, w, gamma=default_gamma):
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.R
    nu_1 = builder.D1
    nu_2 = builder.D2
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2

    def get_z_s(w):
        z = np.exp(-1j*w*dt)
        return z, (z - 1)/(z*dt)

    def square_root_interior(w):
        z, s = get_z_s(w)
        return 1j*np.sqrt(-1*(1+(a*dt*s)**2 - (a**2+1)*dt*s))

    def sigma_plus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s +a**2*dt*r + square_root_interior(w))/(a*np.sqrt(dt*nu))

    def sigma_minus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s +a**2*dt*r - square_root_interior(w))/(a*np.sqrt(dt*nu))

    sigma_1 = sigma_minus(w, nu_1)
    sigma_2 = - sigma_minus(w, nu_2)
    sigma_3 = sigma_plus(w, nu_1)
    sigma_4 = -sigma_plus(w, nu_2)
    assert (np.real(sigma_1) > 0).all()
    assert (np.real(sigma_2) < 0).all()
    assert (np.real(sigma_3) > 0).all()
    assert (np.real(sigma_4) < 0).all()

    z, s = get_z_s(w)
    mu_1 = z*(1 + r*dt*b - b*dt*nu_1*sigma_1**2)
    mu_2 = z*(1 + r*dt*b - b*dt*nu_2*sigma_2**2)
    mu_3 = z*(1 + r*dt*b - b*dt*nu_1*sigma_3**2)
    mu_4 = z*(1 + r*dt*b - b*dt*nu_2*sigma_4**2)
    assert (np.linalg.norm(mu_1 - mu_2) < 1e-10) # mu_1 == mu_2
    assert (np.linalg.norm(mu_3 - mu_4) < 1e-10) # mu_3 == mu_4
    gamma_t = (mu_1 - gamma(z))/(mu_1 - mu_3)

    varrho = ((L1 + nu_2*sigma_2)/(L2 + nu_2*sigma_2) * (1 - gamma_t) + \
             (L1 + nu_2*sigma_4)/(L2 + nu_2*sigma_4) * gamma_t) * \
             ((L2 + nu_1*sigma_1)/(L1 + nu_1*sigma_1) * (1 - gamma_t) + \
             (L2 + nu_1*sigma_3)/(L1 + nu_1*sigma_3) * gamma_t)

    return np.abs(varrho)



def lambda_gamma_Pade_FD(builder, w, gamma=default_gamma):
    """
        returns lambda_1, lambda_2, lambda_3, lambda_4, gamma_t1, gamma_t2
        used in the computation of rho^{Pade, FD}.
    """
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.R
    nu_1 = builder.D1
    nu_2 = builder.D2

    def get_z(w):
        return np.exp(-1j*w*dt)

    ##################################
    # DISCRETE CASE, discrete in time ofc
    ##################################

    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)

    def sqrt_g2_4f(w, nu):
        """
            computes the value sqrt(g^2 - 4f).
            This value must be carefully computed because
            it is the square root of a complex:
            a factorization of (g^2-4f) is needed.
        """
        z = get_z(w)
        # f is (q + d / z)/Gamma_b^2
        # g is (v / z - c)/Gamma_b
        q = (1 + r * dt * b)**2
        d = - 1 - a*r*dt
        v = a / b
        c = 2*(1+r*dt*b)
        Gamma_b = Gamma(b, nu)
        # now we have g^2-4f = (pol_a /z^2 + pol_b / z + pol_c)/Gamma_b^2
        pol_a = v**2
        pol_b = - (4*d + 2*c*v)
        pol_c = c**2 - 4*q

        first_term = np.sqrt((1/z - (-pol_b + np.sqrt(pol_b**2 - 4*pol_a*pol_c))/(2*pol_a)))
        second_term = np.sqrt((1/z - (-pol_b - np.sqrt(pol_b**2 - 4*pol_a*pol_c))/(2*pol_a)))

        return np.sqrt(pol_a) * first_term * second_term / Gamma_b

    def Gamma(ab, nu):
        return ab*dt*nu/h**2

    def get_f_g(w, nu):
        z = get_z(w)
        Gamma_a, Gamma_b = Gamma(a, nu), Gamma(b, nu)
        return ((1+b*r*dt)**2 - (1+a*r*dt)/z) / (Gamma_b**2), \
                (Gamma_a - 2*z*Gamma_b*(1 + r*dt*b)) / (z*Gamma_b**2)

    def lambda_pp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + sqrt_g2_4f(w, nu)/4 + np.sqrt((-(g-4)*(sqrt_g2_4f(w, nu) -g)/2 - f))/2

    def lambda_pm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + sqrt_g2_4f(w, nu)/4 - np.sqrt((-(g-4)*(sqrt_g2_4f(w, nu) -g)/2 - f))/2

    def lambda_mp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (sqrt_g2_4f(w, nu)/4 + np.sqrt(((g-4)*(sqrt_g2_4f(w, nu) +g)/2 - f))/2)

    def lambda_mm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (sqrt_g2_4f(w, nu)/4 - np.sqrt(((g-4)*(sqrt_g2_4f(w, nu) +g)/2 - f))/2)

    lambda_1 = lambda_mp(w, nu_1) # bon en fait normalement on pourrait utiliser pp et mm
    lambda_2 = lambda_mp(w, nu_2) # mais faudrait utiliser partout lambda_1^{-1} au lieu
    lambda_3 = lambda_pm(w, nu_1) # de lambda_1. Ca vaut pas vraiment le coup
    lambda_4 = lambda_pm(w, nu_2)

    chi_1 = h**2 * (r+1j*w)/nu_1
    chi_2 = h**2 * (r+1j*w)/nu_2
    lambda1_c = 1+chi_1/2 - np.sqrt(chi_1*(chi_1+4))/2 # Je comprends pas pourquoi c'est pas un +
    lambda2_c = 1+chi_2/2 - np.sqrt(chi_2*(chi_2+4))/2

    def mu_FD(w, nu_i, lambda_i):
        z = get_z(w)
        return z*(1 + r*dt*b - Gamma(b, nu_i)*(lambda_i - 2 + 1/lambda_i))

    z = get_z(w)
    mu_1FD = mu_FD(w, nu_1, lambda_1)
    mu_2FD = mu_FD(w, nu_2, lambda_2)
    mu_3FD = mu_FD(w, nu_1, lambda_3)
    mu_4FD = mu_FD(w, nu_2, lambda_4)

    gamma_t1 = (mu_1FD - gamma(z))/(mu_1FD - mu_3FD)
    gamma_t2 = (mu_2FD - gamma(z))/(mu_2FD - mu_4FD)
    return lambda_1, lambda_2, lambda_3, lambda_4, gamma_t1, gamma_t2

def rho_Pade_FD_corr0(builder, w, gamma=default_gamma):
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2
    nu_1 = builder.D1
    nu_2 = builder.D2
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)
    lambda_1, lambda_2, lambda_3, lambda_4, gamma_t1, gamma_t2 = lambda_gamma_Pade_FD(builder, w, gamma=gamma)

    # naive interface:
    eta_22 = nu_2 * (lambda_2-1)/h
    eta_24 = nu_2 * (lambda_4-1)/h
    eta_11 = nu_1 * (1-lambda_1)/h
    eta_13 = nu_1 * (1-lambda_3)/h

    # RR:
    varrho = ((L1 + eta_22)/(L2 + eta_22) * (1-gamma_t2) + \
             (L1 + eta_24)/(L2 + eta_24) * (gamma_t2)) * \
             ((L2 + eta_11)/(L1 + eta_11) * (1-gamma_t1) + \
             (L2 + eta_13)/(L1 + eta_13) * (gamma_t1))

    return np.abs(varrho)

def rho_Pade_FD_extra(builder, w, gamma=default_gamma):
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2
    nu_1 = builder.D1
    nu_2 = builder.D2
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)
    lambda_1, lambda_2, lambda_3, lambda_4, gamma_t1, gamma_t2 = lambda_gamma_Pade_FD(builder, w, gamma=gamma)

    # extrapolation:
    eta_11 = -nu_1/h * (lambda_1 - 1) * (3/2 - lambda_1/2)
    eta_13 = -nu_1/h * (lambda_3 - 1) * (3/2 - lambda_3/2)
    eta_22 = nu_2/h * (lambda_2 - 1) * (3/2 - lambda_2/2)
    eta_24 = nu_2/h * (lambda_4 - 1) * (3/2 - lambda_4/2)
    # DN: varrho = ((1 - gamma_t1) * eta_11 + gamma_t1 * eta_13) * ((1 - gamma_t2) / eta_22 + gamma_t2 / eta_24)
    # RR:
    varrho = ((L1 + eta_22)/(L2 + eta_22) * (1-gamma_t2) + \
             (L1 + eta_24)/(L2 + eta_24) * (gamma_t2)) * \
             ((L2 + eta_11)/(L1 + eta_11) * (1-gamma_t1) + \
             (L2 + eta_13)/(L1 + eta_13) * (gamma_t1))

    return np.abs(varrho)

def rho_Pade_FD_corr1(builder, w, gamma=default_gamma):
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2
    nu_1 = builder.D1
    nu_2 = builder.D2
    dt = builder.DT
    z = np.exp(-1j*w*dt)
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)
    r = builder.R
    b = 1 + 1/np.sqrt(2)
    a = 1 + np.sqrt(2)

    lambda_1, lambda_2, lambda_3, lambda_4, gamma_t1, gamma_t2 = lambda_gamma_Pade_FD(builder, w, gamma=gamma)

    def mu_FD(nu_i, lambda_i):
        return z*(1 + r*dt*b - b*dt*nu_i/h**2 *(lambda_i - 2 + 1/lambda_i))

    mu_1 = mu_FD(nu_1, lambda_1)
    mu_2 = mu_FD(nu_2, lambda_2)
    mu_3 = mu_FD(nu_1, lambda_3)
    mu_4 = mu_FD(nu_2, lambda_4)

    # naive interface:
    eta_22 = nu_2 * (lambda_2-1)/h
    eta_24 = nu_2 * (lambda_4-1)/h
    eta_11 = nu_1 * (1-lambda_1)/h
    eta_13 = nu_1 * (1-lambda_3)/h

    #is always the same h, and not h and -h !
    # the sign is taken into account in the matrix inversion.

    zeta_11 = nu_1 * (b*mu_1 - a) * (lambda_1 - 1) / h - h / 2 * ((mu_1 - 1) / dt + r*(b*mu_1 - a))
    zeta_12 = nu_2 * (b*mu_2 - a) * (lambda_2 - 1) / h - h / 2 * ((mu_2 - 1) / dt + r*(b*mu_2 - a))
    zeta_13 = nu_1 * (b*mu_3 - a) * (lambda_3 - 1) / h - h / 2 * ((mu_3 - 1) / dt + r*(b*mu_3 - a))
    zeta_14 = nu_2 * (b*mu_4 - a) * (lambda_4 - 1) / h - h / 2 * ((mu_4 - 1) / dt + r*(b*mu_4 - a))

    zeta_21 = nu_1 * (lambda_1 - 1) / h - h / 2 * ((z - mu_1) / (z*b*dt) + r)
    zeta_22 = nu_2 * (lambda_2 - 1) / h - h / 2 * ((z - mu_2) / (z*b*dt) + r)
    zeta_23 = nu_1 * (lambda_3 - 1) / h - h / 2 * ((z - mu_3) / (z*b*dt) + r)
    zeta_24 = nu_2 * (lambda_4 - 1) / h - h / 2 * ((z - mu_4) / (z*b*dt) + r)


    psi_11 = nu_1 * (b*gamma(z) - a) *(1 - lambda_1) / h + h/2 * ((gamma(z) - 1)/dt + r*(b*gamma(z) - a))
    psi_12 = nu_2 * (b*gamma(z) - a) *(1 - lambda_2) / h + h/2 * ((gamma(z) - 1)/dt + r*(b*gamma(z) - a))
    psi_13 = nu_1 * (b*gamma(z) - a) *(1 - lambda_3) / h + h/2 * ((gamma(z) - 1)/dt + r*(b*gamma(z) - a))
    psi_14 = nu_2 * (b*gamma(z) - a) *(1 - lambda_4) / h + h/2 * ((gamma(z) - 1)/dt + r*(b*gamma(z) - a))

    psi_21 = nu_1 * (1 - lambda_1) / h + h/2 * ((z - gamma(z))/(z*b*dt) + r)
    psi_22 = nu_2 * (1 - lambda_2) / h + h/2 * ((z - gamma(z))/(z*b*dt) + r)
    psi_23 = nu_1 * (1 - lambda_3) / h + h/2 * ((z - gamma(z))/(z*b*dt) + r)
    psi_24 = nu_2 * (1 - lambda_4) / h + h/2 * ((z - gamma(z))/(z*b*dt) + r)

    #warning of the axis: matrices of arrays...
    bold_psi1 = np.array( [ [psi_11 + L1 * (b*gamma(z)-a), psi_13 + L1 * (b*gamma(z)-a)],
            [psi_21 + L1, psi_23 + L1]])
    # The index on bold matrix is for L1 or L2
    bold_psi2 = np.array([
            [-psi_12 + L2 * (b*gamma(z)-a), - psi_14 + L2 * (b*gamma(z)-a)],
            [- psi_22 + L2, -psi_24 + L2]])


    bold_zeta1 = np.array(
        [[zeta_12 + L1 * (b*mu_2-a), zeta_14 + L1 * (b*mu_4-a)],
        [zeta_22 + L1, zeta_24 + L1]])
    bold_zeta2 = np.array(
        [[-zeta_11 + L2 * (b*mu_1-a), -zeta_13 + L2 * (b*mu_3-a)],
        [-zeta_21 + L2, -zeta_23 + L2]])

    bold_zeta1 = bold_zeta1.transpose((2,0,1))
    bold_zeta2 = bold_zeta2.transpose((2,0,1))
    bold_psi1 = bold_psi1.transpose((2,0,1))
    bold_psi2 = bold_psi2.transpose((2,0,1))
    print(np.linalg.inv(bold_zeta2).shape)
    # the matrix ret should be 2x2 (being 2x2 does not mean it is correct tho)
    ret = np.linalg.inv(bold_zeta2) @ bold_psi1 @ np.linalg.inv(bold_zeta1) @ bold_psi2
    return np.linalg.eig(ret)[0].transpose() # returns couple of two arrays of eigenvalues
if __name__ == "__main__":
    import main
    main.main()
