import numpy as np
from atmo_fd_be_flux import AtmoFdBeFlux

def main():
    linear_test()
    sinus_test()

def linear_test():
    size_domain = 200

    # we choose u to be (size_domain-x)*cos (x+t). it is 0 at size_domain !
    def u_real(x,t):
        return (size_domain - x)*np.cos(t)
    def phi_real(x,t): # du / dx
        return - np.cos(t) + np.zeros_like(x)
    def forcing_real(x,t): # d u / dt + r u - nu d^2u / dx^2
        return - (size_domain - x)*np.sin(t) + r* u_real(x,t)

    error = []
    for M in (6, 11):
        T = 1
        nu = 1e-1
        r = 1e-1
        dt = .001
        N = int(T/dt)
        atmo_integrator = AtmoFdBeFlux(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, DT=dt)
        h = atmo_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            phi_n, u_n = atmo_integrator.integrate_in_time(phi_n=phi_n, u_n=u_n,
                    interface_info_next_time=nu*phi_real(x=0., t=n*dt),
                    forcing_next_time=forcing_real(x=x_u, t=n*dt), alpha=1.,
                    theta=0., boundary=u_real(x_u[-1], n*dt))

        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    for err in error:
        assert abs(err - 0.046) < 1e-3
    error = []
    for dt in (0.1, 0.01):
        M = 5
        T = 1
        nu = 1e-1
        r = 1j*1e-1
        N = int(T/dt)
        atmo_integrator = AtmoFdBeFlux(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, DT=dt)
        h = atmo_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            phi_n, u_n = atmo_integrator.integrate_in_time(phi_n=phi_n, u_n=u_n,
                    interface_info_next_time=nu*phi_real(x=0, t=n*dt),
                    forcing_next_time=forcing_real(x=x_u, t=n*dt), alpha=1., theta=0.,
                    boundary=u_real(x=x_u[-1], t=N*dt))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    assert abs(10 - error[0]/error[1]) < 0.1


def sinus_test():
    size_domain = 2
    error = []
    for M in (5, 9):
        T = .1
        nu = 1e-1
        r = 1e-1
        dt = .0001
        N = int(T/dt)
        atmo_integrator = AtmoFdBeFlux(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, DT=dt)
        h = atmo_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        # we choose u to be cos (x+t).
        def u_real(x,t):
            return np.sin(size_domain - x)*np.cos(t)
        def phi_real(x,t): # du / dx
            return -np.cos(size_domain - x)*np.cos(t)
        def forcing_real(x,t): # d u / dt + r u - nu d^2u / dx^2
            return - np.sin(size_domain - x)*np.sin(t) + r* u_real(x,t) + nu*np.sin(size_domain - x)*np.cos(t)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            phi_n, u_n = atmo_integrator.integrate_in_time(phi_n=phi_n, u_n=u_n,
                    interface_info_next_time=nu*phi_real(x=0, t=n*dt),
                    forcing_next_time=forcing_real(x=x_u, t=n*dt), alpha=1.,
                    theta=0., boundary=u_real(x_u[-1], n*dt))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    # when we divide h by 2, we divide error by 4: it seems to be order 2
    assert abs(error[0]/error[1]) > 4.


