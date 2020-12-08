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


def lambda_Pade_FV(builder, w):
    """
        returns lambda_1, lambda_2, lambda_3, lambda_4, gamma_t1, gamma_t2
        used in the computation of rho^{Pade, FV}.
    """
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.R
    nu_1 = builder.D1
    nu_2 = builder.D2

    def get_z(w):
        return np.exp(-1j*w*dt)
    z = get_z(w)

    ##################################
    # DISCRETE CASE, discrete in time ofc
    ##################################

    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)

    def Gamma(ab, nu):
        return ab*dt*(nu/h**2 - r/6) - 1/6


    def lambda_pm_pm(sign_first, sign_second, nu):
        """ solves c (-1 + x)^4 + b (-1 + x)^2 x + a x^2 = 0
        a = z - 1 - (a - 2 z b - z b^2 dt r) dt r
        b = gamma_a - (1+b dt r) 2 z gamma_b
        c = z gamma_b ^2

        we name pa, pb, pc to avoid confusion with a, b letters of Pade
        """
        gamma_a = Gamma(a, nu)
        gamma_b = Gamma(b, nu)
        assert r == 0
        pa = z - 1 - (a - 2*z*b - z* b**2 * dt*r)*dt*r
        pb = gamma_a - (1 + b*dt*r)*2*z*gamma_b
        pc = z*gamma_b**2
        """ solves (-1 + x)^4 + f (-1 + x)^2 x + d x^2 = 0 """
        d = pa / pc
        f = pb / pc

        sqrt_f2_m_4d = np.sqrt(pc*(f**2 - 4*d))/np.sqrt(pc)
        return 1+ 1/4 *(sign_second*sqrt_f2_m_4d -f + sign_first * \
                np.sqrt(pc*(2*(f-4)*(-sign_second*sqrt_f2_m_4d + f) - 4*d))/np.sqrt(pc))

    lambda_1 = lambda_pm_pm(1, -1, nu_1)
    lambda_2 = lambda_pm_pm(-1, -1, nu_2)
    lambda_3 = lambda_pm_pm(1, 1, nu_1)
    lambda_4 = lambda_pm_pm(-1, 1, nu_2)

    nu = nu_1
    gamma_a = Gamma(a, nu)
    gamma_b = Gamma(b, nu)
    assert r == 0
    pa = z - 1 - (a - 2*z*b - z* b**2 * dt*r)*dt*r
    pb = gamma_a - (1 + b*dt*r)*2*z*gamma_b
    pc = z*gamma_b**2

    # TODO tout n'est pas égal à 0: problème ou c'est juste que nu est pas le bon ?
    # print(max(lambda_1**2 * pa + (lambda_1-1)**2*lambda_1*pb + pc*(lambda_1 - 1)**4))
    # print(max(lambda_2**2 * pa + (lambda_2-1)**2*lambda_2*pb + pc*(lambda_2 - 1)**4))
    # print(max(lambda_3**2 * pa + (lambda_3-1)**2*lambda_3*pb + pc*(lambda_3 - 1)**4))
    # print(max(lambda_4**2 * pa + (lambda_4-1)**2*lambda_4*pb + pc*(lambda_4 - 1)**4))

    return lambda_1, lambda_2, lambda_3, lambda_4

def rho_Pade_FV(builder, w, gamma=default_gamma):
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

    lambda_1, lambda_2, lambda_3, lambda_4 = lambda_Pade_FV(builder, w)

    def Gamma(ab, nu):
        return ab*dt*(nu/h**2 - r/6) - 1/6

    ratio_u = 1 / ((1+b*dt*r)**2*z - (1 + a*dt*r))
    ratio_ustar = z*(1+b*dt*r)*ratio_u

    def eta_etastar(lambda_i, nu_j):
        sigma_i = np.log(lambda_i) / h # we let sigma be with a real part negative 
        mu_i = z  * ( 1 + b * dt * r - Gamma(b, nu_j) * (lambda_i - 2 + 1/lambda_i))

        eta_i = -sigma_i * (h/6 * (2 + lambda_i) + ratio_u* nu_j / h * dt * (b*z*(1+b*dt*r) - a + b*mu_i) * (1 - lambda_i))
        eta_istar = -sigma_i * (h/6 * mu_i * (2 + lambda_i) + ratio_ustar * nu_j / h * dt * (b*(1+a*dt*r)/(1+b*dt*r) - a + b*mu_i) * (1 - lambda_i))
        return eta_i, eta_istar

    eta1, eta1star = eta_etastar(lambda_1, nu_1)
    eta2, eta2star = eta_etastar(lambda_2, nu_2)
    eta3, eta3star = eta_etastar(lambda_3, nu_1)
    eta4, eta4star = eta_etastar(lambda_4, nu_2)

    sigma_1 = np.log(lambda_1) / h
    sigma_2 = np.log(lambda_2) / h
    sigma_3 = np.log(lambda_3) / h
    sigma_4 = np.log(lambda_4) / h

    mu_1 = z  * ( 1 + b * dt * r - Gamma(b, nu_1) * (lambda_1 - 2 + 1/lambda_1))
    mu_2 = z  * ( 1 + b * dt * r - Gamma(b, nu_2) * (lambda_2 - 2 + 1/lambda_2))
    mu_3 = z  * ( 1 + b * dt * r - Gamma(b, nu_1) * (lambda_3 - 2 + 1/lambda_3))
    mu_4 = z  * ( 1 + b * dt * r - Gamma(b, nu_2) * (lambda_4 - 2 + 1/lambda_4))

    # zeta_step is the matrix multiplying (A_k, A_k')^T,(B_k, B_k')^T when they are prognosed.
    # psi_step is the matrix multiplying (A_k, A_k')^T,(B_k, B_k')^T when they are diagnosed.
    zeta_1 = np.array((( L1*eta1    +nu_1*sigma_1,      L1*eta3    +nu_1*sigma_3),
                        (L1*eta1star+nu_1*sigma_1*mu_1, L1*eta3star+nu_1*sigma_3*mu_3)))
    zeta_2 = np.array((( L2*eta2    +nu_2*sigma_2,      L2*eta4    +nu_2*sigma_4),
                        (L2*eta2star+nu_2*sigma_2*mu_2, L2*eta4star+nu_2*sigma_4*mu_4)))
    psi_1 = np.array((( L1*eta1    +nu_1*sigma_1,      L1*eta3    +nu_1*sigma_3),
                        ((L1*eta1+nu_1*sigma_1)*gamma(z), (L1*eta3+nu_1*sigma_3)*gamma(z))))
    psi_2 = np.array((( L2*eta2    +nu_2*sigma_2,        L2*eta4   +nu_2*sigma_4),
                      ((L2*eta2    +nu_2*sigma_2)*gamma(z),(L2*eta4   +nu_2*sigma_4)*gamma(z))))

    zeta_1 = zeta_1.transpose((2,0,1))
    zeta_2 = zeta_2.transpose((2,0,1))
    psi_1 = psi_1.transpose((2,0,1))
    psi_2 = psi_2.transpose((2,0,1))

    gamma_t1 = (eta1star*eta3 - gamma(z)*eta1*eta3) / (eta1star*eta3 - eta3star*eta1)
    gamma_t2 = (mu_2 - gamma(z))/ (mu_2 - mu_4)
    #gammma_t1 = gamma_t2 = 0 # TO IMITATE CONTINUOUS IN TIME
    cv_rate = nu_1/nu_2 * (sigma_1 * (1 - gamma_t1) / eta1 + sigma_3 * gamma_t1/eta3) * \
            ((1 - gamma_t2)/sigma_2 * eta2 + gamma_t2/sigma_4*eta4)

    from discretizations.time.PadeLowTildeGamma import PadeLowTildeGamma
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    dis = builder.build(PadeLowTildeGamma, QuadSplinesFV)
    cv_rate_c = dis.analytic_robin_robin_modified(w=w,
            order_time=0, order_operators=float('inf'),
            order_equations=float('inf'))
    #cv_rate_c = rho_Pade_c(builder, w, gamma=gamma)


    # The convergence matrix is thus zeta_1^{-1} psi_2 zeta_2^{-1} psi_1
    #ret = np.linalg.inv(zeta_1) @ psi_2 @ np.linalg.inv(zeta_2) @ psi_1
    #plt.plot(np.linalg.eig(ret)[0].transpose()[0])
    #plt.plot(np.linalg.eig(ret)[0].transpose()[1])
    import matplotlib.pyplot as plt
    lam_1m, lam_2m, lam_1p, lam_2p = dis.lambda_1_2_pm(1j*w)
    etadir, etaneu = dis.eta_dirneu(1, lam_1m, lam_1p, s=1j*w)

    lambda_i = lam_1m
    nu_j = nu_1
    #sigma_i = np.log(lambda_i) / h # we let sigma be with a real part negative 
    #mu_i = z  * ( 1 + b * dt * r - Gamma(b, nu_j) * (lambda_i - 2 + 1/lambda_i))
    import matplotlib.pyplot as plt
    plt.semilogx(w, lam_1m, "--", label="lam1_c")
    plt.semilogx(w, lam_1p, "--", label="lam2_c")
    plt.semilogx(w, lambda_1, label="lam1")
    plt.semilogx(w, lambda_2, label="lam2")
    #plt.semilogx(w, lambda_3, label="lam3")
    #plt.semilogx(w, lambda_4, label="lam4")
    plt.legend()
    plt.show()

    #return np.linalg.eig(ret)[0].transpose()
    return np.abs(cv_rate)

if __name__ == "__main__":
    import main
    main.main()
