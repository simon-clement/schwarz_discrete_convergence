import numpy as np
from atmosphere_models.atmosphere_BE_FD import AtmosphereBEFD

def main():
    sinus_test_large_window_kc1()
    linear_test_large_window()
    sinus_test_large_window()
    linear_test()
    sinus_test()

def linear_test_large_window():
    size_domain = 200

    # we choose u to be (size_domain-x)*cos (x+t). it is 0 at size_domain !
    def u_real(x,t):
        return (size_domain - x)*np.cos(t)
    def phi_real(x,t): # du / dx
        return - np.cos(t) + np.zeros_like(x)
    def forcing_real(x,t, h): # d u / dt + r u - nu d^2u / dx^2
        return - (size_domain - x)*np.sin(t) + r* u_real(x,t)

    error = []
    for M in (6, 11):
        T = 1
        nu = 1e-1
        r = 1e-1j
        dt = .001
        LAMBDA=1e8
        N = int(T/dt)
        atmo_integrator = AtmosphereBEFD(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt, K_c=0)
        h = atmo_integrator.h

        x_u = np.linspace(0,size_domain, M)
        x_phi = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        total_interface = np.array([LAMBDA*u_real(0, n*dt) + nu*phi_real(x=h/2, t=n*dt)
                for n in range(N+1)])
        def forcing(t):
            return forcing_real(x_u, t, h)

        u_N = atmo_integrator.integrate_large_window(interface=total_interface,
                initial_prognostic=u_n, initial_diagnostic=phi_n,
                forcing=forcing, boundary=np.zeros_like(total_interface),
                DEBUG_LAST_VAL=True)
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_N)/np.sqrt(M-1)]
    for err in error:
        assert abs(err - 0.04) < 1e-2

    error = []
    for dt in (0.1, 0.05):
        M = 5
        T = 1
        nu = 1e-1
        r = 1j*1e-1
        N = int(T/dt)
        atmo_integrator = AtmosphereBEFD(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt, K_c=0)
        h = atmo_integrator.h

        x_u = np.linspace(0,size_domain, M)
        x_phi = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        total_interface = np.array([LAMBDA*u_real(0, n*dt) + nu*phi_real(x=h/2, t=n*dt)
                for n in range(N+1)])
        def forcing(t):
            return forcing_real(x_u, t, h)

        u_N = atmo_integrator.integrate_large_window(interface=total_interface,
                initial_prognostic=u_n, initial_diagnostic=phi_n,
                forcing=forcing, boundary=np.zeros_like(total_interface),
                DEBUG_LAST_VAL=True)
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_N)/np.sqrt(M-1)]
    assert abs(2. - error[0]/error[1]) < 0.4 # divided dt by 2, dividing error by ~4

def sinus_test_large_window():
    size_domain = 2
    error = []
    for M in (8, 17):
        T = 1
        nu = 1e-1
        r = 0.#1e-1
        LAMBDA=0.
        dt = .0001
        N = int(T/dt)
        atmo_integrator = AtmosphereBEFD(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt, K_c=0)
        h = atmo_integrator.h

        x_u = np.linspace(0,size_domain, M)
        x_phi = np.linspace(h/2,size_domain - h/2, M - 1)

        # we choose u to be cos (x+t).
        def u_real(x,t):
            return np.sin(size_domain - x)*np.cos(t)
        def phi_real(x,t): # du / dx
            return -np.cos(size_domain - x)*np.cos(t)
        def forcing_real(x,t): # d u / dt + r u - nu d^2u / dx^2
            return np.sin(size_domain - x) * (- np.sin(t) + r* np.cos(t) + nu*np.cos(t))

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)
        total_interface = np.array([LAMBDA*u_real(0, n*dt) + nu*phi_real(x=h/2, t=n*dt)
                for n in range(N+1)])
        def forcing(t):
            return forcing_real(x_u, t)

        u_N = atmo_integrator.integrate_large_window(interface=total_interface,
                initial_prognostic=u_n, initial_diagnostic=phi_n,
                forcing=forcing, boundary=np.zeros_like(total_interface),
                DEBUG_LAST_VAL=True)

        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_N)/np.sqrt(M-1)]
    assert abs(error[0]/error[1]) > 4.

def linear_test():
    size_domain = 200

    # we choose u to be (size_domain-x)*cos (x+t). it is 0 at size_domain !
    def u_real(x,t):
        return (size_domain - x)*np.cos(t)
    def phi_real(x,t): # du / dx
        return - np.cos(t) + np.zeros_like(x)
    def forcing_real(x,t, h): # d u / dt + r u - nu d^2u / dx^2
        return - (size_domain - x)*np.sin(t) + r* u_real(x,t)

    error = []
    for M in (6, 11):
        T = 1
        nu = 1e-1
        r = 1e-1j
        dt = .001
        LAMBDA=1e8
        N = int(T/dt)
        atmo_integrator = AtmosphereBEFD(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt, K_c=0)
        h = atmo_integrator.h

        x_u = np.linspace(0,size_domain, M)
        x_phi = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            n_star = n -1 - 1/np.sqrt(2)
            interface_robin = (LAMBDA*u_real(0,(n-1)*dt) + nu*phi_real(x=h/2, t=(n-1)*dt),
                    LAMBDA*u_real(0,n_star*dt) + nu*phi_real(x=h/2, t=n_star*dt),
                    LAMBDA*u_real(0, n*dt) + nu*phi_real(x=h/2, t=n*dt))
            forcing = (forcing_real(x=x_u, t=(n-1)*dt, h=h), 
                        forcing_real(x=x_u, t=n_star*dt, h=h), 
                        forcing_real(x=x_u, t=n*dt, h=h))
            u_n, phi_n = atmo_integrator.integrate_in_time(diagnosed=phi_n, prognosed=u_n,
                    interface_robin=interface_robin, forcing=forcing, boundary=(0,0,0))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    for err in error:
        assert abs(err - 0.04) < 1e-2

    error = []
    for dt in (0.1, 0.05):
        M = 5
        T = 1
        nu = 1e-1
        r = 1j*1e-1
        N = int(T/dt)
        atmo_integrator = AtmosphereBEFD(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt, K_c=0)
        h = atmo_integrator.h

        x_u = np.linspace(0,size_domain, M)
        x_phi = np.linspace(h/2,size_domain - h/2, M - 1)

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            n_star = n -1 - 1/np.sqrt(2)
            interface_robin = (LAMBDA*u_real(0,(n-1)*dt) + nu*phi_real(x=h/2, t=(n-1)*dt),
                    LAMBDA*u_real(0,n_star*dt) + nu*phi_real(x=h/2, t=n_star*dt),
                    LAMBDA*u_real(0, n*dt) + nu*phi_real(x=h/2, t=n*dt))
            forcing = (forcing_real(x=x_u, t=(n-1)*dt, h=h), 
                        forcing_real(x=x_u, t=n_star*dt, h=h), 
                        forcing_real(x=x_u, t=n*dt, h=h))
            u_n, phi_n = atmo_integrator.integrate_in_time(diagnosed=phi_n, prognosed=u_n,
                    interface_robin=interface_robin, forcing=forcing, boundary=(0,0,0))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    assert abs(2. - error[0]/error[1]) < 0.4 # divided dt by 2, dividing error by ~4

def sinus_test():
    size_domain = 2
    error = []
    for M in (8, 17):
        T = 1
        nu = 1e-1
        r = 0.#1e-1
        LAMBDA=0.
        dt = .0001
        N = int(T/dt)
        atmo_integrator = AtmosphereBEFD(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt, K_c=0)
        h = atmo_integrator.h

        x_u = np.linspace(0,size_domain, M)
        x_phi = np.linspace(h/2,size_domain - h/2, M - 1)

        # we choose u to be cos (x+t).
        def u_real(x,t):
            return np.sin(size_domain - x)*np.cos(t)
        def phi_real(x,t): # du / dx
            return -np.cos(size_domain - x)*np.cos(t)
        def forcing_real(x,t): # d u / dt + r u - nu d^2u / dx^2
            return np.sin(size_domain - x) * (- np.sin(t) + r* np.cos(t) + nu*np.cos(t))

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)

        for n in range(1, N+1):
            n_star = n -1 - 1/np.sqrt(2)
            interface_robin = (LAMBDA*u_real(0,(n-1)*dt) + nu*phi_real(x=h/2, t=(n-1)*dt),
                    LAMBDA*u_real(0,n_star*dt) + nu*phi_real(x=h/2, t=n_star*dt),
                    LAMBDA*u_real(0, n*dt) + nu*phi_real(x=h/2, t=n*dt))
            forcing = (forcing_real(x=x_u, t=(n-1)*dt), 
                        forcing_real(x=x_u, t=n_star*dt), 
                        forcing_real(x=x_u, t=n*dt))
            u_n, phi_n = atmo_integrator.integrate_in_time(diagnosed=phi_n, prognosed=u_n,
                    interface_robin=interface_robin, forcing=forcing, boundary=(0,0,0))
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_n)/np.sqrt(M-1)]
    assert abs(error[0]/error[1]) > 4.

def sinus_test_large_window_kc1():
    size_domain = 2
    error = []
    for M in (8, 17):
        T = 1
        nu = 1e-1
        r = 0.#1e-1
        LAMBDA=0.
        dt = .0001
        N = int(T/dt)
        atmo_integrator = AtmosphereBEFD(r=r, nu=nu, M=M, SIZE_DOMAIN=size_domain, LAMBDA=LAMBDA, DT=dt, K_c=1)
        h = atmo_integrator.h

        x_u = np.linspace(0,size_domain, M)
        x_phi = np.linspace(h/2,size_domain - h/2, M - 1)

        # we choose u to be sin(H-x) cos (t).
        def u_real(x,t):
            return np.sin(size_domain - x)*np.cos(t)
        def phi_real(x,t): # du / dx
            return -np.cos(size_domain - x)*np.cos(t)
        def forcing_real(x,t): # d u / dt + r u - nu d^2u / dx^2
            return np.sin(size_domain - x) * (- np.sin(t) + r* np.cos(t) + nu*np.cos(t))

        phi_n = phi_real(x_phi, 0)
        u_n = u_real(x_u, 0)
        total_interface = np.array([LAMBDA*u_real(0, n*dt) + nu*phi_real(x=0, t=n*dt)
                for n in range(N+1)])
        def forcing(t):
            return forcing_real(x_u, t)

        u_N = atmo_integrator.integrate_large_window(interface=total_interface,
                initial_prognostic=u_n, initial_diagnostic=phi_n,
                forcing=forcing, boundary=np.zeros_like(total_interface),
                DEBUG_LAST_VAL=True)
        error += [np.linalg.norm(u_real(x=x_u, t=N*dt) - u_N)/np.sqrt(M-1)]
    assert abs(error[0]/error[1]) > 4.
