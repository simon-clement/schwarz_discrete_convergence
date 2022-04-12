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
def to_integrate_swfrac_sl(z: float, inv_L_MO: float) -> float:
    zeta: float = -z*inv_L_MO
    Ch: float = np.cbrt(1-25*zeta) # in shortwave_absorption.py
    phi_h: float = 5*zeta + 1 if zeta >= 0 else 1/Ch
    sw_frac: float = shortwave_frac_sl(z)[0]
    return phi_h * sw_frac / z

@jit(nopython=True)
def integrated_shortwave_frac_sl(z: float, inv_L_MO: float) -> float:
    """
        int_z^0 { 1/z'(phi_h(-z'/L_MO) * sum(Ai exp(Ki z'))) dz'}
        returns E(z)
    """
    if abs(z) < 1e-5:
        return 0.
    n: int = 30
    s: float = 0.
    for z_prim in np.linspace(z*1e-5, z, n):
        s += to_integrate_swfrac_sl(z_prim, inv_L_MO)
    return -z * s / n

@jit(nopython=True)
def Qsw_E(z: float, SL):
    """
        SHORTWAVE RADIATIVE FLUX:
        returns Qsw * E(z)
    """
    return SL.Q_sw * integrated_shortwave_frac_sl(z, SL.inv_L_MO)

