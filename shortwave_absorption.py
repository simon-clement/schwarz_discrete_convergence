"""
    This module defines how the shortwave radiative fluxes
    are absorbed in the ocean.
"""
import numpy as np
import scipy.integrate
from numba import jit
from scipy.special import exp1

@jit(nopython=True)
def shortwave_fractional_decay(M, h_half):
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
    swr_frac = np.zeros(M+1)
    swr_frac[M]=1.
    for k in range(M-1, -1, -1):
        xi1=attn1*h_half[k]
        if xi1 > -20.:
            swdk1 *= np.exp(xi1)
        else:
            swdk1 = 0.
        xi2 = attn2*h_half[k]
        if xi2 > -20.:
            swdk2 *= np.exp(xi2)
        else:
            swdk2 = 0.
        swr_frac[k]=swdk1+swdk2
    return swr_frac

# A_i = np.array([.237, .360, .179, .087, .08, .0246,
#     .025, .007, .0004])
# k_i = 1./np.array([34.8, 2.27, 3.15e-2,
#     5.48e-3, 8.32e-4, 1.26e-4, 3.13e-4, 7.82e-5, 1.44e-5])
# replacing by the four first values for numerical stability:
A_i = np.array([.237, .360, .179, .087+.08+.0246+.025+.007+.0004])
k_i = 1./np.array([34.8, 2.27, 3.15e-2, 5.48e-3])

@jit(nopython=True)
def shortwave_frac_sl(z):
    """
        Paulson and Simpson, 1981
    """
    return np.sum(A_i * np.exp(np.outer(z, k_i)), axis=-1)

@jit(nopython=True)
def to_integrate_swfrac_sl(z: float, inv_L_MO: float) -> float:
    """
        returns 1/z'(phi_h(-z'/L_MO) * sum(Ai exp(Ki z')))
    """
    zeta: float = -z*inv_L_MO
    Ch: float = np.cbrt(1-25*zeta) # in shortwave_absorption.py
    phi_h: float = 5*zeta + 1 if zeta >= 0 else 1/Ch
    sw_frac: float = shortwave_frac_sl(z)[0]
    return phi_h * sw_frac / z

def integrated_shortwave_frac_sl(z: float, inv_L_MO: float,
        z0H: float) -> float:
    """
        int_z^0 { 1/z'(phi_h(-z'/L_MO) * sum(Ai exp(Ki z'))) dz'}
        returns E(z)
    """
    if abs(z) < 1e-5:
        return 0.
    n: int=30
    zprim = np.linspace(z*1e-7, z, n)
    ki_z = np.squeeze(np.outer(z, k_i))
    ki_zprim = np.outer(zprim, k_i) # note: zprim is 1D here
    ki_z0H = np.minimum(k_i * z0H, 700.) # avoids overflow in exp
    # ki_z0H > 700 only happens when bulk fails and z0H>>1.
    # it is associated with small A_i anyway.
    exp1_m_exp1 = exp1(ki_z0H) - exp1(ki_z0H-ki_z)
    left_part = np.sum(A_i * np.exp(ki_z0H) * exp1_m_exp1, axis=-1)
    # left part should be roughly equivalent to:
    # integral betweeen z and 0 of (sum_i A_i e^(k_i z) / (-z+z0H))

    right_part = np.sum(A_i * np.exp(ki_zprim), axis=-1)

    zet = -zprim*inv_L_MO
    phi_h = np.where(zet>=0, 5*zet+1, 1/np.cbrt(1-25*zet))
    right_part *= (1 - phi_h) / (z0H-zprim)

    return left_part - (-z) * np.sum(right_part) / n

def Qsw_E(z: float, SL):
    """
        SHORTWAVE RADIATIVE FLUX:
        returns Qsw * E(z)
    """
    return SL.Q_sw * integrated_shortwave_frac_sl(z, SL.inv_L_MO,
            SL.z_0H)

