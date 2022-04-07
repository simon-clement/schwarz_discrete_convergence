#!/usr/bin/python3
"""
    This module is here to simulate the OA coupling.
"""
from typing import NamedTuple, List
import numpy as np
from scipy import interpolate
from atm1DStratified import Atm1dStratified
from ocean1DStratified import Ocean1dStratified


class State(NamedTuple):
    """
        all the output variables of the atmosphere (resp. ocean) model
        A part of them will be used by the ocean (resp. atmosphere).
    """
    u_delta: np.ndarray # SL Momentum at all time steps
    t_delta: np.ndarray # SL Temperature at all time steps

class NumericalSetting(NamedTuple):
    """
        all the input variables of both the atmosphere and ocean
    """
    T: float # Integration time (in seconds)
    sf_scheme_a: str # should be accepted by simulator_oce
    sf_scheme_o: str # should be accepted by simulator_atm
    delta_sl_a: float # height of the top of ASL
    delta_sl_o: float # height of the bottom of OSL (<0)


def schwarz_coupling(simulator_oce: Ocean1dStratified,
        simulator_atm: Atm1dStratified,
        parameters: NumericalSetting)-> (List[State], List[State]):
    """
        computes the coupling between the two models
        Atm1dStratified and Ocean1dStratified.
    """
    atm_state, oce_state = [initialization_atmosphere(parameters)], []
    NUMBER_SCHWARZ_ITERATION = 3
    for _ in range(NUMBER_SCHWARZ_ITERATION):
        oce_state += [compute_ocean(simulator_oce,
            atm_state[-1], parameters)]
        atm_state += [compute_atmosphere(simulator_atm,
            oce_state[-1], parameters)]
    return atm_state, oce_state

def initialization_atmosphere(numer_set: NumericalSetting) -> State:
    """
    returns a State that can be used by ocean model for integration.
    """

def compute_ocean(simulator_oce: Ocean1dStratified,
        atm_state: State,
        numer_set: NumericalSetting) -> State:
    """
        Integrator in time of the ocean
    """
    T0 = 16. # Reference temperature
    N = int(numer_set.T/simulator_oce.dt) # Number of time steps
    theta_0 = T0 - simulator_oce.N0**2 * \
            np.abs(simulator_oce.z_half[:-1]) \
            / simulator_oce.alpha / 9.81 # Initial temperature
    u_0 = np.zeros(simulator_oce.M)
    phi_0 = np.zeros(simulator_oce.M+1)
    dz_theta_0 = np.ones(simulator_oce.M+1) * simulator_oce.N0**2 \
            / simulator_oce.alpha / 9.81

    Qsw, Qlw = numer_set.Q_sw, numer_set.Q_lw
    delta_sl = numer_set.delta_sl_o
    sf_scheme = numer_set.sf_scheme_o
    wind_10m = projection(atm_state.u_delta, N)
    temp_10m = projection(atm_state.t_delta, N)

    if sf_scheme in {"FV free", "FV2"}:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                simulator_oce.initialization(\
                np.zeros(simulator_oce.M)+0j, # u_0
                np.copy(theta_0), # theta_0
                delta_sl, wind_10m[0], temp_10m[0], Qsw[0], Qlw[0],
                10., sf_scheme)
    else:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                u_0, phi_0, theta_0, dz_theta_0, 0., T0

    if sf_scheme[:2] == "FV":
        u_current, phi, tke, all_u_star, theta, \
                    dz_theta, l_eps, SL, viscosity =simulator_oce.FV(\
                u_t0=u_i, phi_t0=phi_i, theta_t0=theta_i,
                dz_theta_t0=dz_theta_i, Q_sw=Qsw, Q_lw=Qlw,
                delta_sl_a=numer_set.delta_sl_a,
                u_delta=u_delta, t_delta=t_delta, delta_sl=delta_sl,
                heatloss=None, wind_10m=wind_10m, TEST_CASE=0,
                temp_10m=temp_10m, sf_scheme=sf_scheme)

    elif sf_scheme[:2] == "FD":
        u_currentFD, tke, all_u_star, thetaFD, \
                    l_eps, viscosityFD = simulator_oce.FD(\
                u_t0=u_0, theta_t0=theta_0, TEST_CASE=0,
                delta_sl_a=numer_set.delta_sl_a,
                Q_sw=Qsw, Q_lw=Qlw, wind_10m=wind_10m,
                temp_10m=temp_10m,
                heatloss=None, sf_scheme=sf_scheme)
        # TODO return the good data for atmosphere
    else:
        raise NotImplementedError("Cannot infer discretization " + \
                "from surface flux scheme name " + sf_scheme)

def compute_atmosphere(simulator_atm: Atm1dStratified,
        oce_state: State,
        numer_set: NumericalSetting) -> State:
    """
        Integrator in time of the atmosphere
    """
    N = int(numer_set.T/simulator_atm.dt) # Number of time steps
    M = simulator_atm.M # Number of grid points
    u_0 = 8*np.ones(M) + 0j
    phi_0 = np.zeros(M+1) + 0j
    theta_0 = 280*np.ones(M)
    dz_theta_0 = np.zeros(M+1)
    forcing = 1j*simulator_atm.f*simulator_atm.u_g*np.ones((N+1, M))

    delta_sl = numer_set.delta_sl_a
    sf_scheme = numer_set.sf_scheme_a
    uo_delta = projection(oce_state.u_delta, N)
    to_delta = projection(oce_state.t_delta, N)
    Q_sw = projection(numer_set.Q_sw, N)
    Q_lw = projection(numer_set.Q_lw, N)
    u_deltasl = u_0[0]
    z_constant = 15

    if sf_scheme in {"FV free", "FV2"}:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                simulator_atm.initialization(\
                u_0, phi_0, theta_0, dz_theta_0, delta_sl, uo_delta,
                to_delta, Q_sw[0], Q_lw[0],
                z_constant, numer_set.delta_sl_o)
    else:
        u_i, phi_i, theta_i, dz_theta_i, u_delta, t_delta = \
                u_0, phi_0, theta_0, dz_theta_0, u_0[0], theta_0[0]

    u, phi, tke_full, ustar, temperature, dz_theta, l_eps, SL = \
            simulator_atm.FV(u_t0=u_0, phi_t0=phi_0,
                    SST=to_delta, sf_scheme=sf_scheme,
                    u_0=uo_delta, u_delta=u_deltasl,
                    delta_sl_o=numer_set.delta_sl_o,
                    forcing=forcing, delta_sl=delta_sl)
    #TODO return good data for ocean.


def projection(array: np.ndarray, N: int)-> np.ndarray:
    """
        projects an array of size array.shape[0] onto
        an other array of shape N+1.
    """
    return interpolate.interp1d(array, np.linspace(0, 1,
        array.shape[0]))(np.linspace(0, 1, N+1))
