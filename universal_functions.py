"""
This module defines several universal functions:
    all functions of this module should return tuples of functions
    (phi_m, phi_h, psi_m, psi_h, Psi_m, Psi_h)
    (See Nishizawa and Kitamura, 2018)
    phi_m, phi_h are the universal functions.
    psi_m, psi_h are their integral form.
    Psi_m, Psi_h are the primitive of psi_m, psi_h.
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def large_phi_m(zeta: np.ndarray) -> np.ndarray:
    Cm = np.cbrt(1-14*zeta)
    return np.where(zeta>=0, 5*zeta+1, 1/Cm)
@jit(nopython=True)
def large_phi_h(zeta: np.ndarray) -> np.ndarray: # warning: there are duplicates of this function
    Ch = np.cbrt(1-25*zeta) # in shortwave_absorption.py
    return np.where(zeta>=0, 5*zeta+1, 1/Ch)
@jit(nopython=True)
def large_psi_m(zeta: np.ndarray) -> np.ndarray:
    Cm = np.cbrt(1-14*zeta)
    sq3 = np.sqrt(3)
    return np.where(zeta>=0, -5*zeta,
        sq3 * np.arctan(sq3) - \
        sq3 * np.arctan(sq3/3*(2*Cm+1)) + 1.5 * \
        np.log((Cm**2 + Cm + 1)/3))
@jit(nopython=True)
def large_psi_h(zeta: np.ndarray) -> np.ndarray:
    sq3 = np.sqrt(3)
    Ch = np.cbrt(1-25*zeta)
    return np.where(zeta>=0, -5*zeta,
        sq3 * np.arctan(sq3) - \
            sq3 * np.arctan(sq3/3*(2*Ch+1)) + 1.5 * \
            np.log((Ch**2 + Ch + 1)/3))
@jit(nopython=True)
def large_Psi_m(zeta: np.ndarray) -> np.ndarray:
    Cm = np.cbrt(1-14*zeta)
    sq3 = np.sqrt(3)
    return np.where(zeta>=0, -5*zeta/2,
            sq3 * np.arctan(sq3) - \
            sq3 * np.arctan(sq3/3*(2*Cm+1)) + 1.5 * \
            np.log((Cm**2 + Cm + 1)/3) - (2*Cm+1)*(Cm-1) / \
            2/(Cm**2 + Cm + 1))

@jit(nopython=True)
def large_Psi_h(zeta: np.ndarray) -> np.ndarray:
    sq3 = np.sqrt(3)
    Ch = np.cbrt(1-25*zeta)
    return np.where(zeta>=0, -5*zeta/2,
            sq3 * np.arctan(sq3) - \
            sq3 * np.arctan(sq3/3*(2*Ch+1)) + 1.5 * \
            np.log((Ch**2 + Ch + 1)/3) - (2*Ch+1)*(Ch-1) / \
            2/(Ch**2 + Ch + 1))

# a = 4.7
# Pr = 0.74
a = 4.8
Pr = 4.8/7.8
@jit(nopython=True)
def businger_phi_m(zeta: np.ndarray) -> np.ndarray:
    fm = (1-15*np.minimum(zeta, 0.))**(1/4)
    return np.where(zeta>=0, a*zeta+1, 1/fm)
@jit(nopython=True)
def businger_phi_h(zeta: np.ndarray) -> np.ndarray:
    fh = (1-9*np.minimum(zeta, 0.))**(1/2)
    return np.where(zeta>=0, a*zeta/Pr+1, 1/fh)
@jit(nopython=True)
def businger_psi_m(zeta: np.ndarray) -> np.ndarray:
    fm = (1-15*np.minimum(zeta, 0.))**(1/4)
    zeta_neg = np.log((1+fm)**2*(1+fm**2)/8) - \
                    2*np.arctan(fm) + np.pi/2
    return np.where(zeta>=0, -a*zeta, zeta_neg)
@jit(nopython=True)
def businger_psi_h(zeta: np.ndarray) -> np.ndarray:
    fh = (1-9*np.minimum(zeta, 0.))**(1/2)
    return np.where(zeta>=0, -a*zeta/Pr,
        2*np.log((1+fh)/2))
def businger_Psi_m(zeta: np.ndarray) -> np.ndarray:
    fm = (1-15*np.minimum(zeta, 0.))**(1/4)
    cond_list = [np.logical_and(zeta>=0, np.abs(zeta)<1e-6),
            np.logical_and(zeta<0, np.abs(zeta)<1e-6),
            zeta>=0, zeta<0]
    choice_list = [-a*zeta/2, -15*zeta/8, -a*zeta/2,
            np.log((1+fm)**2*(1+fm**2)/8) - \
            2*np.arctan(fm) + \
            (1-fm**3)/12/np.minimum(zeta, -1e-20) + \
            np.pi/2 - 1]
    return np.select(cond_list, choice_list)
def businger_Psi_h(zeta: np.ndarray) -> np.ndarray:
    fh = (1-9*np.minimum(zeta, 0.))**(1/2)
    cond_list = [np.logical_and(zeta>=0, np.abs(zeta)<1e-6),
            np.logical_and(zeta<0, np.abs(zeta)<1e-6),
            zeta>=0, zeta<0]
    choice_list = [-a*zeta/2/Pr, -9*zeta/4, -a*zeta/2/Pr,
            2*np.log((1+fh)/2) + 2*(1-fh)/9/np.minimum(zeta, -1e-20) \
                    - 1 ]
    return np.select(cond_list, choice_list)

Large_et_al_2019 = (large_phi_m, large_phi_h,
            large_psi_m, large_psi_h,
            large_Psi_m, large_Psi_h)
Businger_et_al_1971 = (businger_phi_m, businger_phi_h,
            businger_psi_m, businger_psi_h,
            businger_Psi_m, businger_Psi_h)

