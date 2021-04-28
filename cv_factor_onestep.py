#!/usr/bin/python3
"""
    This module computes theoric convergence rates and
    provide functions to observe real convergence rate.
"""
from numpy import sqrt, exp

#########################################################################
# THEORIC PART : RETURN RATES YIELDED BY ANALYSIS IN FREQUENTIAL DOMAIN #
#########################################################################

def rho_c_c(builder, w, overlap_L=0):
    return rho_c_s(builder, 1j*w, overlap_L=overlap_L)

def rho_BE_c(builder, w, overlap_L=0):
    return rho_c_s(builder, BE_s(builder.DT, w), overlap_L=overlap_L)

def rho_BE_FV(builder, w, overlap_M=0):
    return rho_FV_s(builder, BE_s(builder.DT, w), overlap_M=overlap_M)

def rho_BE_FD(builder, w, k_c=0, overlap_M=0):
    return rho_FD_s(builder, BE_s(builder.DT, w), k_c, overlap_M=overlap_M)

def rho_c_FV(builder, w, overlap_M=0):
    return rho_FV_s(builder, 1j*w, overlap_M=overlap_M)

def rho_c_FD(builder, w, k_c=0, overlap_M=0):
    return rho_FD_s(builder, 1j*w, k_c, overlap_M=overlap_M)

#######################################################################
# Internal functions
#######################################################################

def rho_RR(eta_1_dir,eta_1_neu, eta_2_dir, eta_2_neu, p1, p2):
    """ convergence factor without overlap """
    return (eta_2_neu + p1*eta_2_dir) / (eta_1_neu + p1*eta_1_dir) \
            * (eta_1_neu + p2*eta_1_dir) / (eta_2_neu + p2*eta_2_dir)

def rho_c_s(builder, s, overlap_L):
    eta_1_neu = sqrt(builder.D1)*sqrt(builder.R + s)
    eta_2_neu = - sqrt(builder.D2)*sqrt(builder.R + s)
    overlap_term = exp(-overlap_L * (sqrt((builder.R + s)/builder.D1) \
                                    + sqrt((builder.R + s)/builder.D2)))
    return overlap_term * rho_RR(eta_1_dir=1, eta_1_neu=eta_1_neu,
            eta_2_dir=1, eta_2_neu=eta_2_neu,
            p1=builder.LAMBDA_1, p2=builder.LAMBDA_2)

def rho_FD_s(builder, s, k_c, overlap_M):
    h1 = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    h2 = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    chi_1 = h1**2 * (builder.R + s) / builder.D1 
    chi_2 = h2**2 * (builder.R + s) / builder.D2
    eta_1_neu = builder.D1/(2*h1)*(chi_1*(k_c-1) + sqrt(chi_1)*sqrt(chi_1+4))
    eta_2_neu = builder.D2/(2*h2)*(chi_2*(1 - k_c) - sqrt(chi_2)*sqrt(chi_2+4))
    lambda_1 = 1+ (chi_1 - sqrt(chi_1)*sqrt(chi_1 + 4))/2
    lambda_2 = 1+ (chi_2 - sqrt(chi_2)*sqrt(chi_2 + 4))/2
    return lambda_1**overlap_M * lambda_2**overlap_M * \
            rho_RR(eta_1_dir=1, eta_1_neu=eta_1_neu, eta_2_dir=1, eta_2_neu=eta_2_neu,
            p1=builder.LAMBDA_1, p2=builder.LAMBDA_2)

def rho_FV_s(builder, s, overlap_M):
    h1 = builder.SIZE_DOMAIN_1 / (builder.M1 - 1)
    h2 = builder.SIZE_DOMAIN_2 / (builder.M2 - 1)
    chi_1 = h1**2 * (builder.R + s) / builder.D1
    chi_2 = h2**2 * (builder.R + s) / builder.D2
    eta_1_dir = sqrt(1+chi_1/12)
    eta_2_dir = sqrt(1+chi_2/12)
    eta_1_neu = builder.D1/h1*sqrt(chi_1)
    eta_2_neu = -builder.D2/h2*sqrt(chi_2)
    lambda_1 = (1/chi_1 + 1/3 - sqrt(1/chi_1 + 1/12)) / (1/chi_1 - 1/6)
    lambda_2 = (1/chi_2 + 1/3 - sqrt(1/chi_2 + 1/12)) / (1/chi_2 - 1/6)
    return lambda_1**overlap_M * lambda_2**overlap_M * \
            rho_RR(eta_1_dir=eta_1_dir, eta_1_neu=eta_1_neu,
            eta_2_dir=eta_2_dir, eta_2_neu=eta_2_neu,
            p1=builder.LAMBDA_1, p2=builder.LAMBDA_2)

def BE_s(dt, w):
    z = exp(-1j*w*dt)
    return (z - 1)/(z*dt)

if __name__ == "__main__":
    import main
    main.main()
