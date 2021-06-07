import numpy as np
from ocean_models.ocean_Pade_FV import OceanPadeFV

def main():
    linear_test_large_window()
    sinus_test_large_window()
    linear_test()
    sinus_test()

def linear_test_large_window():
    size_domain = 200

    # we choose u to be cos (x+t).
    def u_real(x,t):
        return x*np.cos(t)
    def u_bar(x,t):
        return x*np.cos(t)
    def phi_real(x,t): # du / dx
        return np.cos(t) + np.zeros_like(x)
    def forcing_real(x,t, h): # d u / dt + r u - nu d^2u / dx^2
        # integration around x +- h/2
        return - x*np.sin(t) + r* u_real(x,t)
    error = []
    for M in (6, 11):
        T = 1
        nu = .1
        r = 1e-1j
        dt = .01
        LAMBDA=1e8
        N = int(T/dt)
        ocean_integrator = OceanPadeFV(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = np.flipud(phi_real(x_phi, 0))
        u_n = np.flipud(u_bar(x_u, 0))

        total_interface = np.array([LAMBDA*u_real(size_domain, n*dt) + nu*phi_real(x=size_domain, t=n*dt)
                for n in range(N+1)])
        def forcing(t):
            return np.flipud(forcing_real(x_u, t, h))

        u_N = ocean_integrator.integrate_large_window(interface=total_interface,
                initial_prognostic=phi_n, initial_diagnostic=u_n,
                forcing=forcing, boundary=np.zeros_like(total_interface),
                DEBUG_LAST_VAL=True)
        error += [np.linalg.norm(u_bar(x=x_u, t=N*dt) - np.flipud(u_N))/np.sqrt(M-1)]
    for err in error:
        assert abs(err - 0.0015) < 1e-4
    error = []
    for dt in (0.1, 0.05):
        M = 10
        T = 1
        nu = 1e-1
        r = 1e-1j
        N = int(T/dt)
        ocean_integrator = OceanPadeFV(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = np.flipud(phi_real(x_phi, 0))
        u_n = np.flipud(u_bar(x_u, 0))

        total_interface = np.array([LAMBDA*u_real(size_domain, n*dt) + nu*phi_real(x=size_domain, t=n*dt)
                for n in range(N+1)])
        def forcing(t):
            return np.flipud(forcing_real(x_u, t, h))

        u_N = ocean_integrator.integrate_large_window(interface=total_interface,
                initial_prognostic=phi_n, initial_diagnostic=u_n,
                forcing=forcing, boundary=np.zeros_like(total_interface),
                DEBUG_LAST_VAL=True)
        error += [np.linalg.norm(u_bar(x=x_u, t=N*dt) - np.flipud(u_N))/np.sqrt(M-1)]

    assert abs(4. - error[0]/error[1]) < 0.4 # divided dt by 2, dividing error by ~4


def sinus_test_large_window():
    size_domain = 2
    error = []
    for M in (20, 41):
        T = 1
        nu = 1e-1
        r = 1e-1
        dt = .001
        LAMBDA=1.
        N = int(T/dt)
        ocean_integrator = OceanPadeFV(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        # we choose u to be cos (x+t).
        def u_real(x,t):
            return np.sin(x)*np.cos(t)
        def u_bar(x,t):
            return (np.cos(x-h/2)-np.cos(x+h/2))/h*np.cos(t)
        def phi_real(x,t): # du / dx
            return np.cos(x)*np.cos(t)
        def forcing_real(x,t, h): # avg(d u / dt + r u - nu d^2u / dx^2) around x+-h/2
            return (np.cos(x-h/2)-np.cos(x+h/2))/h * (- np.sin(t) + r* np.cos(t) + nu*np.cos(t))

        phi_n = np.flipud(phi_real(x_phi, 0))
        u_n = np.flipud(u_bar(x_u, 0))

        total_interface = np.array([LAMBDA*u_real(size_domain, n*dt) + nu*phi_real(x=size_domain, t=n*dt)
                for n in range(N+1)])
        def forcing(t):
            return np.flipud(forcing_real(x_u, t, h))

        u_N = ocean_integrator.integrate_large_window(interface=total_interface,
                initial_prognostic=phi_n, initial_diagnostic=u_n,
                forcing=forcing, boundary=np.zeros_like(total_interface),
                DEBUG_LAST_VAL=True)
        error += [np.linalg.norm(u_bar(x=x_u, t=N*dt) - np.flipud(u_N))/np.sqrt(M-1)]
    # when we divide h by 2, we divide error by almost: it seems to be order 3
    # however, it is a sinus function. so it is more or less attended^^
    # well, it may not be the best test we've got, but at least it is here
    # if something changes, maybe this order of reduction would be impacted.
    order_of_reduction = error[0]/error[1]
    assert order_of_reduction > 4.


def linear_test():
    size_domain = 200

    # we choose u to be cos (x+t).
    def u_real(x,t):
        return x*np.cos(t)
    def u_bar(x,t):
        return x*np.cos(t)
    def phi_real(x,t): # du / dx
        return np.cos(t) + np.zeros_like(x)
    def forcing_real(x,t, h): # d u / dt + r u - nu d^2u / dx^2
        # integration around x +- h/2
        return - x*np.sin(t) + r* u_real(x,t)
    error = []
    for M in (6, 11):
        T = 1
        nu = .1
        r = 1e-1j
        dt = .01
        LAMBDA=1e8
        N = int(T/dt)
        ocean_integrator = OceanPadeFV(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = np.flipud(phi_real(x_phi, 0))
        u_n = np.flipud(u_bar(x_u, 0))

        for n in range(1, N+1):
            n_star = n -1 - 1/np.sqrt(2)
            interface_robin = (LAMBDA*u_real(size_domain,(n-1)*dt) + nu*phi_real(x=size_domain, t=(n-1)*dt),
                    LAMBDA*u_real(size_domain,n_star*dt) + nu*phi_real(x=size_domain, t=n_star*dt),
                    LAMBDA*u_real(size_domain, n*dt) + nu*phi_real(x=size_domain, t=n*dt))
            forcing = (np.flipud(forcing_real(x=x_u, t=(n-1)*dt, h=h)), 
                        np.flipud(forcing_real(x=x_u, t=n_star*dt, h=h)), 
                        np.flipud(forcing_real(x=x_u, t=n*dt, h=h)))
            phi_n, u_n = ocean_integrator.integrate_in_time(prognosed=phi_n, diagnosed=u_n,
                    interface_robin=interface_robin, forcing=forcing, boundary=(0,0,0))
        error += [np.linalg.norm(u_bar(x=x_u, t=N*dt) - np.flipud(u_n))/np.sqrt(M-1)]
    for err in error:
        assert abs(err - 0.0063) < 1e-4
    error = []
    for dt in (0.1, 0.05):
        M = 10
        T = 1
        nu = 1e-1
        r = 1e-1j
        N = int(T/dt)
        ocean_integrator = OceanPadeFV(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = np.flipud(phi_real(x_phi, 0))
        u_n = np.flipud(u_bar(x_u, 0))

        for n in range(1, N+1):
            n_star = n -1 - 1/np.sqrt(2)
            interface_robin = (LAMBDA*u_real(size_domain,(n-1)*dt) + nu*phi_real(x=size_domain, t=(n-1)*dt),
                    LAMBDA*u_real(size_domain,n_star*dt) + nu*phi_real(x=size_domain, t=n_star*dt),
                    LAMBDA*u_real(size_domain, n*dt) + nu*phi_real(x=size_domain, t=n*dt))
            forcing = (np.flipud(forcing_real(x=x_u, t=(n-1)*dt, h=h)), 
                        np.flipud(forcing_real(x=x_u, t=n_star*dt, h=h)), 
                        np.flipud(forcing_real(x=x_u, t=n*dt, h=h)))
            phi_n, u_n = ocean_integrator.integrate_in_time(prognosed=phi_n, diagnosed=u_n,
                    interface_robin=interface_robin, forcing=forcing, boundary=(0,0,0))
        error += [np.linalg.norm(u_bar(x=x_u, t=N*dt) - np.flipud(u_n))/np.sqrt(M-1)]
    assert abs(4. - error[0]/error[1]) < 0.4 # divided dt by 2, dividing error by ~4


def sinus_test():
    size_domain = 2
    error = []
    for M in (20, 41):
        T = 1
        nu = 1e-1
        r = 1e-1
        dt = .001
        LAMBDA=1.
        N = int(T/dt)
        ocean_integrator = OceanPadeFV(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt)
        h = ocean_integrator.h

        x_phi = np.linspace(0,size_domain, M)
        x_u = np.linspace(h/2,size_domain - h/2, M - 1)

        # we choose u to be cos (x+t).
        def u_real(x,t):
            return np.sin(x)*np.cos(t)
        def u_bar(x,t):
            return (np.cos(x-h/2)-np.cos(x+h/2))/h*np.cos(t)
        def phi_real(x,t): # du / dx
            return np.cos(x)*np.cos(t)
        def forcing_real(x,t, h): # avg(d u / dt + r u - nu d^2u / dx^2) around x+-h/2
            return (np.cos(x-h/2)-np.cos(x+h/2))/h * (- np.sin(t) + r* np.cos(t) + nu*np.cos(t))

        phi_n = np.flipud(phi_real(x_phi, 0))
        u_n = np.flipud(u_bar(x_u, 0))

        for n in range(1, N+1):
            n_star = n -1 - 1/np.sqrt(2)
            interface_robin = (LAMBDA*u_real(size_domain,(n-1)*dt) + nu*phi_real(x=size_domain, t=(n-1)*dt),
                    LAMBDA*u_real(size_domain,n_star*dt) + nu*phi_real(x=size_domain, t=n_star*dt),
                    LAMBDA*u_real(size_domain, n*dt) + nu*phi_real(x=size_domain, t=n*dt))
            forcing = (np.flipud(forcing_real(x=x_u, t=(n-1)*dt, h=h)), 
                        np.flipud(forcing_real(x=x_u, t=n_star*dt, h=h)), 
                        np.flipud(forcing_real(x=x_u, t=n*dt, h=h)))
            phi_n, u_n = ocean_integrator.integrate_in_time(prognosed=phi_n, diagnosed=u_n,
                    interface_robin=interface_robin, forcing=forcing, boundary=(0,0,0))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - np.flipud(u_n))/np.sqrt(M-1)]
    # when we divide h by 2, we divide error by almost: it seems to be order 3
    # however, it is a sinus function. so it is more or less attended^^
    # well, it may not be the best test we've got, but at least it is here
    # if something changes, maybe this order of reduction would be impacted.
    order_of_reduction = error[0]/error[1]
    assert order_of_reduction > 4.


