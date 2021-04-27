import numpy as np
from ocean_fd_be_flux import OceanFdBeFlux

def main():
    linear_test()
    sinus_test()

def linear_test():
    size_domain = 200

    # we choose u to be cos (x+t).
    def u_real(x,t):
        return x*np.cos(t)
    def phi_real(x,t): # du / dx
        return np.cos(t) + np.zeros_like(x)
    def forcing_real(x,t): # d u / dt + r u - nu d^2u / dx^2
        return - x*np.sin(t) + r* u_real(x,t)
    error = []
    for M in (6, 11):
        T = 1
        nu = 1e-1
        r = 1e-1
        dt = .001
        N = int(T/dt)
        ocean_integrator = OceanFdBeFlux(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            phi_n, u_n = ocean_integrator.integrate_in_time(phi_n=phi_n, u_n=u_n,
                    interface_flux_next_time=nu*phi_real(x=size_domain, t=n*dt),
                    forcing_next_time=forcing_real(x=x_u, t=n*dt), boundary=u_real(x=h/2, t=n*dt))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    for err in error:
        assert abs(err - 0.046) < 1e-3
    error = []
    for dt in (0.1, 0.01):
        M = 5
        T = 1
        nu = 1e-1
        r = 1e-1
        N = int(T/dt)
        ocean_integrator = OceanFdBeFlux(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            phi_n, u_n = ocean_integrator.integrate_in_time(phi_n=phi_n, u_n=u_n,
                    interface_flux_next_time=nu*phi_real(x=size_domain, t=n*dt),
                    forcing_next_time=forcing_real(x=x_u, t=n*dt), boundary=u_real(x=h/2, t=n*dt))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    assert abs(10 - error[0]/error[1]) < 0.13


def sinus_test():
    size_domain = 2.
    error = []
    for M in (5, 9):
        T = .1
        nu = 1e-1
        r = 1e-1
        dt = .0001
        N = int(T/dt)
        ocean_integrator = OceanFdBeFlux(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        # we choose u to be cos (x+t).
        def u_real(x,t):
            return np.sin(x)*np.cos(t)
        def phi_real(x,t): # du / dx
            return np.cos(x)*np.cos(t)
        def forcing_real(x,t): # d u / dt + r u - nu d^2u / dx^2
            return - np.sin(x)*np.sin(t) + r* u_real(x,t) + nu*np.sin(x)*np.cos(t)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            phi_n, u_n = ocean_integrator.integrate_in_time(phi_n=phi_n, u_n=u_n,
                    interface_flux_next_time=nu*phi_real(x=size_domain, t=n*dt),
                    forcing_next_time=forcing_real(x=x_u, t=n*dt), boundary=u_real(x=h/2, t=n*dt))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    # when we divide h by 2, we divide error by almost 4:
    # it asks a way too long computation but it works... increasing dt would make an error because
    # we use B-E scheme.
    # if something changes, maybe this order of reduction would be impacted.
    order_of_reduction = error[0]/error[1]
    assert order_of_reduction > 4


