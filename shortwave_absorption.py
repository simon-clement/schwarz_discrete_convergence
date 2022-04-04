"""
    This module defines how the shortwave radiative fluxes
    are absorbed in the ocean.
"""
import numpy as np
import scipy.integrate
from numba import jit

@jit(nopython=True)
def shortwave_fractional_decay(M, h_full):
    """
        To compute the solar radiation penetration
        we need this function that was
        directly translated from fortran.
    """
    mu1 = [0.35, 0.6, 1.0, 1.5, 1.4]
    mu2 = [23.0, 20.0, 17.0, 14.0, 7.9]
    r1 = [0.58, 0.62, 0.67, 0.77, 0.78, ]
    Jwt=0
    attn1=-1./mu1[Jwt]
    attn2=-1./mu2[Jwt]

    swdk1=r1[Jwt]
    swdk2=1.-swdk1
    swr_frac = np.zeros(M)
    swr_frac[-1]=1.
    for k in range(M-1, -1, -1):
        xi1=attn1*h_full[k]
        if xi1 > -20.:
            swdk1 *= np.exp(xi1)
        else:
            swdk1 = 0.
        xi2 = attn2*h_full[k]
        if xi2 > -20.:
            swdk2 *= np.exp(xi2)
        else:
            swdk2 = 0.
        swr_frac[k]=swdk1+swdk2
    return swr_frac

@jit(nopython=True)
def shortwave_frac_sl(z):
    """
        Paulson and Simpson, 1981
    """
    A_i = np.array([.237, .360, .179, .087, .08, .0246,
        .025, .007, .0004])
    k_i = 1./np.array([34.8, 2.27, 3.15e-2,
        5.48e-3, 8.32e-4, 1.26e-4, 3.13e-4, 7.82e-5, 1.44e-5])
    return np.sum(A_i * np.exp(np.outer(z, k_i)), axis=-1)

@jit(nopython=True)
def to_integrate_swfrac_sl(z: float, inv_L_MO: float):
    zeta = -z*inv_L_MO
    Ch = np.cbrt(1-25*zeta) # in shortwave_absorption.py
    phi_h = 5*zeta + 1 if zeta >= 0 else 1/Ch
    return phi_h * shortwave_frac_sl(z) / z

def integrated_shortwave_frac_sl(inv_L_MO: float, z: float) -> float:
    """
        int_z^0 { 1/z'(phi_h(-z'/L_MO) * sum(Ai exp(Ki z'))) dz'}
        returns E(z)
    """
    return scipy.integrate.quad(to_integrate_swfrac_sl, z, z*1e-5,
            args=(inv_L_MO,))[0]

def Qsw_E(z: float, SL, turhocp):
    """
        SHORTWAVE RADIATIVE FLUX:
        returns Qsw / (t*u*rho cp) * E(z)
    """
    return SL.Q_sw * integrated_shortwave_frac_sl(SL.inv_L_MO, z) \
            / turhocp

