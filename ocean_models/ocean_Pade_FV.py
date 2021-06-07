import numpy as np
from scipy.linalg import solve_banded

class OceanPadeFV():
    def __init__(self, r, # reaction coefficient
                 nu, # Diffusivity
                 M, # Number of collocation points
                 SIZE_DOMAIN, # Size of \\Omega_1
                 LAMBDA,
                 DT, # Time step
                 GAMMA_START=0, # convolution starts at n-gamma_start
                 GAMMA_COEFFS=[1+1/np.sqrt(2), -1/np.sqrt(2)]): # kernel gamma_coeffs: coeffs[i]*u[start+i]
        """
            The data needed is passed through this constructor.
            The space step is SIZE_DOMAIN / (M-1)
            there is M-1 points for u, M points for phi
        """
        self.r, self.nu, self.M, self.size_domain, self.Lambda, self.dt = \
            r, nu, M, SIZE_DOMAIN, LAMBDA, DT
        self.h = SIZE_DOMAIN / (M - 1)
        from cv_factor_pade import rho_Pade_FV
        self.discrete_rate = rho_Pade_FV
        self.gamma_start = GAMMA_START
        self.gamma_coefs = np.array(GAMMA_COEFFS)

    def size_u(self):
        return self.M - 1

    def interface_values(self, prognosed, diagnosed, overlap):
        u_interface = diagnosed[overlap] + self.h / 3 * prognosed[overlap] + self.h / 6 * prognosed[overlap+1]
        phi_interface = prognosed[overlap]
        return u_interface, phi_interface

    def integrate_large_window(self, interface,
            initial_prognostic=None, initial_diagnostic=None,
            forcing=None, boundary=None, DEBUG_LAST_VAL=False): 
        """
            Given the information, returns the interface information after integration in time.

            Parameters:
            interface: Robin condition given to the model, array of size N+1
                (N being the number of time steps, the first value can be generally set to 0)
            The following parameters are set to 0 if not prescribed:
                initial_prognostic (prognosed variable, space derivative of the solution), (size M)
                initial_diagnostic (diagnosed average solution), (size M-1)
                forcing: function [0,T] -> C^M used as forcing in each volume
                boundary: array of size N+1

            Returns:
            Tuple solution, derivative where solution and derivative are (N+1) 1D arrays

            scheme is:
                -Noting Y = 1+delta_x^2/6, b=1+1/sqrt(2), a=1+sqrt(2)
                -Noting R = r*dt, Gamma = nu*dt/h^2

                    (Y + b (R Y - Gamma delta_x^2))phi_star = (Y + a (R Y - Gamma delta_x^2)) phi_n
                    (Y + b (R Y - Gamma delta_x^2))phi_np1 = Y phi_star
                See pdf for the implementation.
        """
        initial_prognostic = np.zeros(self.M) if initial_prognostic is None else initial_prognostic
        initial_diagnostic = np.zeros(self.M-1) if initial_diagnostic is None else initial_diagnostic
        forcing = (lambda _: np.zeros(self.M)) if forcing is None else forcing
        boundary = np.zeros_like(interface) if boundary is None else boundary
        Gamma = self.nu * self.dt / self.h**2
        R = self.r * self.dt
        a = 1+np.sqrt(2)
        b = 1+1/np.sqrt(2)
        h, dt, nu = self.h, self.dt, self.nu
        N = interface.shape[0] - 1
        tilde_p = self.Lambda
        gcoefs = self.gamma_coefs
        gorder = gcoefs.shape[0]
        gstart = self.gamma_start
        Y_FV = 1/6*np.vstack((np.concatenate(([1e100, tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([2*tilde_p + 6*nu/h], 4*np.ones(self.M-2), [2])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        D_FV = np.vstack((np.concatenate(([1e100,tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([-tilde_p], -2*np.ones(self.M-2), [-1])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        def prod_banded(tridiag_mat, x):
            ret = tridiag_mat[1]*x
            ret[:-1] += tridiag_mat[0,1:]*x[1:]
            ret[1:] += tridiag_mat[2,:-1]*x[:-1]
            return ret
        matrix_to_inverse = (1+b*R)*Y_FV - b*Gamma*D_FV

        # initialisation of the main loop of integration
        u_current = initial_diagnostic
        phi_current = initial_prognostic
        solution = [initial_diagnostic[0] \
                + h / 3 * initial_prognostic[0] + h / 6 * initial_prognostic[0]]
        derivative = [initial_prognostic[0]]

        def get_star(function_of_n, n):
            return np.sum([(np.zeros_like(function_of_n(0)) if n+gstart+i>N or n+gstart+i < 0
                else function_of_n(n+gstart+i)*gcoefs[i]) for i in range(gorder)], axis=0)

        # actual integration:
        for n in range(N):
            f_n, f_np1 = forcing(n*dt), forcing((n+1)*dt)
            robin_n, robin_np1 = interface[n], interface[n+1]
            bd_n, bd_np1 = boundary[n], boundary[n+1]
            f_star = get_star((lambda i:forcing(i*dt)), n)
            robin_star = get_star((lambda i:interface[i]), n)
            bd_star = get_star((lambda i:boundary[i]), n)

            c_n = np.concatenate(([1/h*((robin_star - (1-(1-b)*R)*robin_n)/((1-b)*dt)-self.Lambda*f_n[0])],
                    np.diff(-f_n)/h, [1/h*(-(bd_star- (1-(1-b)*R)*bd_n)/((1-b)*dt) + f_n[-1])]))
            c_star = np.concatenate(([1/h*(((1+(1-b)*R)*robin_star - robin_n)/((1-b)*dt)-self.Lambda*f_star[0])],
                    np.diff(-f_star)/h, [1/h*(-((1+(1-b)*R)*bd_star- bd_n)/((1-b)*dt) + f_star[-1])]))
            c_np1 = np.concatenate(([1/h*(((1+b*R)*robin_np1 - robin_star)/(b*dt)-self.Lambda*f_np1[0])],
                    np.diff(-f_np1)/h, [1/h*(-((1+b*R)*bd_np1- bd_star)/(b*dt) + f_np1[-1])]))
            rhs_c_step1 = b*dt*c_star + (1-2*b)*dt*c_n
            rhs_c_step2 = b*dt*c_np1

            rhs_step1 = prod_banded(Y_FV + (1-2*b)*(Gamma*D_FV - R*Y_FV), phi_current) + rhs_c_step1

            phi_star = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step1)

            rhs_step2 = prod_banded(Y_FV, phi_star) + rhs_c_step2

            # we use diff(-phi) because we are reversed
            u_star = (u_current + b*(h*Gamma*np.diff(-phi_star) + dt*f_star) \
                    + a*(R*u_current - h*Gamma*np.diff(-phi_current) - dt*f_n)) / (1 + b*R)

            phi_next = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step2)
            u_next = (u_star + b*(h*Gamma*np.diff(-phi_next) + dt*f_np1)) / (1 + b*R)

            solution += u_next[0] + h / 3 * phi_next[0] + h / 6 * phi_next[0]
            derivative += [phi_next[0]]

            u_current = u_next
            phi_current = phi_next

        if DEBUG_LAST_VAL:
            return u_next
        return solution, derivative

    def integrate_in_time(self, prognosed, diagnosed, interface_robin, forcing, boundary):
        """
            Given the information, returns the tuple (phi_(n+1), u_(n+1)).

            Parameters:
            phi_n (prognostic variable, space derivative of solution), (size M)
            u_n (diagnosed average of solution on each volume), (size M-1)
            interface_robin, forcing, boundary: tuples for time (tn, t*, t{n+1})

            forcing: the forcing in the diffusion-reaction equation, averaged on each volume
            boundary conditions :
                -Dirichlet(t)=boundary(t) at the bottom of ocean
                -Robin(t)=interface_robin(t) at interface

            scheme is:
                -Noting Y = 1+delta_x^2/6, b=1+1/sqrt(2), a=1+sqrt(2)
                -Noting R = r*dt, Gamma = nu*dt/h^2

                    (Y + b (R Y - Gamma delta_x^2))phi_star = (Y + a (R Y - Gamma delta_x^2)) phi_n
                    (Y + b (R Y - Gamma delta_x^2))phi_np1 = Y phi_star
                See pdf for the implementation.
        """
        
        phi_n = prognosed
        u_n = diagnosed
        Gamma = self.nu * self.dt / self.h**2
        R = self.r * self.dt
        a = 1+np.sqrt(2)
        b = 1+1/np.sqrt(2)
        h, dt, nu = self.h, self.dt, self.nu
        tilde_p = self.Lambda
        f_n, f_star, f_np1 = forcing
        robin_n, robin_star, robin_np1 = interface_robin
        bd_n, bd_star, bd_np1 = boundary

        def prod_banded(tridiag_mat, x):
            ret = tridiag_mat[1]*x
            ret[:-1] += tridiag_mat[0,1:]*x[1:]
            ret[1:] += tridiag_mat[2,:-1]*x[:-1]
            return ret

        c_n = np.concatenate(([1/h*((robin_star - (1-(1-b)*R)*robin_n)/((1-b)*dt)-self.Lambda*f_n[0])],
                np.diff(-f_n)/h, [1/h*(-(bd_star- (1-(1-b)*R)*bd_n)/((1-b)*dt) + f_n[-1])]))
        c_star = np.concatenate(([1/h*(((1+(1-b)*R)*robin_star - robin_n)/((1-b)*dt)-self.Lambda*f_star[0])],
                np.diff(-f_star)/h, [1/h*(-((1+(1-b)*R)*bd_star- bd_n)/((1-b)*dt) + f_star[-1])]))
        c_np1 = np.concatenate(([1/h*(((1+b*R)*robin_np1 - robin_star)/(b*dt)-self.Lambda*f_np1[0])],
                np.diff(-f_np1)/h, [1/h*(-((1+b*R)*bd_np1- bd_star)/(b*dt) + f_np1[-1])]))
        rhs_c_step1 = b*dt*c_star + (1-2*b)*dt*c_n
        rhs_c_step2 = b*dt*c_np1

        Y_FV = 1/6*np.vstack((np.concatenate(([1e100, tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([2*tilde_p + 6*nu/h], 4*np.ones(self.M-2), [2])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        D_FV = np.vstack((np.concatenate(([1e100,tilde_p], np.ones(self.M-2))),#up_diag
            np.concatenate(([-tilde_p], -2*np.ones(self.M-2), [-1])), # diag
            np.concatenate((np.ones(self.M-1), [1e100]))))# low_diag
        matrix_to_inverse = (1+b*R)*Y_FV - b*Gamma*D_FV

        rhs_step1 = prod_banded(Y_FV + (1-2*b)*(Gamma*D_FV - R*Y_FV), phi_n) + rhs_c_step1

        phi_star = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step1)

        rhs_step2 = prod_banded(Y_FV, phi_star) + rhs_c_step2

        phi_np1 = solve_banded(l_and_u=(1, 1), ab=matrix_to_inverse, b=rhs_step2)

        # we use diff(-phi) because we are reversed
        u_star = (u_n + b*(h*Gamma*np.diff(-phi_star) + dt*f_star) \
                + a*(R*u_n - h*Gamma*np.diff(-phi_n) - dt*f_n)) / (1 + b*R)
        u_np1 = (u_star + b*(h*Gamma*np.diff(-phi_np1) + dt*f_np1)) / (1 + b*R)

        return phi_np1, u_np1

    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def repr(self):
        return "OcePadeFV"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.repr() == other.repr()

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())) + self.repr())

    def __repr__(self):
        return repr(sorted(self.__dict__.items())) + self.repr()
