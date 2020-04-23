#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import numpy as np
from numpy import pi
from memoisation import memoised, FunMem
import matplotlib.pyplot as plt
import functools
import discretizations
from simulator import frequency_simulation


"""
    The dictionnary all_figures contains all the functions
    of this module that begins with "fig_".
    When you want to add a figure,
    follow the following rule:
        if the figure is going to be labelled as "fig:foo"
        then the function that generates it should
                                        be named (fig_foo())
    The dictionnary is filling itself: don't try to
    manually add a function.
"""
all_figures = {}

class Builder():
    def __init__(self): # changing defaults will result in needing to recompute all cache
        self.COURANT_NUMBER = 100.
        self.M1 = 2000
        self.M2 = 2000
        self.SIZE_DOMAIN_1 = 200
        self.SIZE_DOMAIN_2 = 200
        self.D1 = .54
        self.D2 = .6
        self.DT = self.COURANT_NUMBER * (self.SIZE_DOMAIN_1 / self.M1)**2 / self.D1
        self.A = 0.
        self.C = 0.
        self.LAMBDA_1 = 1e9
        self.LAMBDA_2 = 0.

    def new(self, Discretisation):
        return Discretisation(A=self.A, C=self.C,
                              D1=self.D1, D2=self.D2,
                              M1=self.M1, M2=self.M2,
                              SIZE_DOMAIN_1=self.SIZE_DOMAIN_1,
                              SIZE_DOMAIN_2=self.SIZE_DOMAIN_2,
                              LAMBDA_1=self.LAMBDA_1,
                              LAMBDA_2=self.LAMBDA_2,
                              DT=self.DT)
    def build(self, time_discretization, space_discretization):
        """
            Given two abstract classes of a time and space discretization,
            build a scheme.
        """
        class AnonymousScheme(time_discretization, space_discretization):
            def __init__(self, *args, **kwargs):
                space_discretization.__init__(self, *args, **kwargs)
                time_discretization.__init__(self, *args, **kwargs)
        return self.new(AnonymousScheme)

    def frequency_cv_factor(self, time_discretization, space_discretization, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        return frequency_simulation(discretization, **kwargs)

    def robin_robin_theorical_cv_factor(self, time_discretization, space_discretization, *args, **kwargs):
        discretization = self.build(time_discretization, space_discretization)
        return discretization.analytic_robin_robin_modified(*args, **kwargs)

    """
        __eq__ and __hash__ are implemented, so that a discretization
        can be stored as key in a dict
        (it is useful for memoisation)
    """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())))

    def __repr__(self):
        return repr(sorted(self.__dict__.items()))

DEFAULT = Builder()

def fig_validatePadeAnalysisRR():
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.space.FD_corr import FiniteDifferencesCorr
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.Manfredi import Manfredi
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.LAMBDA_1 = .5
    builder.LAMBDA_2 = -.4
    builder.D1 = 1.
    builder.D2 = 2.
    builder.C = 0.5
    assert builder.C * builder.DT < 1
        
    discretizations = {}
    time_scheme = Manfredi

    discretizations["FV2"] = (time_scheme, QuadSplinesFV)
    discretizations["FV4"] = (time_scheme, FourthOrderFV)
    discretizations["FD, corr=0"] = (time_scheme, FiniteDifferencesNaive)
    discretizations["FD, extra"] = (time_scheme, FiniteDifferencesExtra)
    #discretizations["FD, corr=1"] = (time_scheme, FiniteDifferencesCorr)

    convergence_factors = {}
    theorical_convergence_factors = {}

    N = 300
    dt = DEFAULT.DT
    ###########
    # Computation of the frequency axis
    ###########
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[int(N//2)] = .5
    # w = 2 pi k T / (N)
    axis_freq = 2 * pi*all_k / N / dt

    kwargs_label_simu = {'label':"Validation by simulation"}
    kwargs_label_cont = {'label':"Continuous"}
    fig, axes = plt.subplots(1, 1, figsize=[6.4, 4.8])
    ###########
    # for each discretization, a simulation
    ###########
    for name in discretizations:
        time_dis, space_dis = discretizations[name]
        alpha_w = memoised(Builder.frequency_cv_factor, builder,
                time_dis, space_dis, N=N, number_samples=50)
        k = 1
        convergence_factors[name] = alpha_w[k+1] / alpha_w[k]

        dis = builder.build(time_dis, space_dis)
        try:
            theorical_convergence_factors[name] = \
                    dis.analytic_robin_robin_modified(w=axis_freq,
                            order_time=float('inf'), order_operators=float('inf'),
                            order_equations=float('inf'))
            axes.semilogx(axis_freq * dt, theorical_convergence_factors[name], "--", **kwargs_label_simu)
        except:
            pass
        continuous = dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=0, order_operators=0,
                        order_equations=0)
        axes.semilogx(axis_freq * dt, convergence_factors[name], label=name)
        if kwargs_label_simu: # We only want the legend to be present once
            axes.semilogx(axis_freq * dt, continuous, "--", **kwargs_label_cont)
            kwargs_label_simu = {}
            kwargs_label_cont = {}

        #axes[0].semilogx(axis_freq * dt, theorical_convergence_factors[name], label=name+ " theorical")
    w, rho_theoric = wAndRhoPadeRR(builder)
    axes.semilogx(w*DEFAULT.DT, rho_theoric, "k--", label="theoric")

    axes.set_xlabel("Frequency variable $\\omega \\delta t$")
    axes.set_ylabel("Convergence factor $\\rho$")
    axes.set_title("Various space discretizations with " + time_scheme.__name__)
    axes.legend()
    show_or_save("fig_validatePadeAnalysisRR")

def fig_compareSettingsDirichletNeumann():
    from discretizations.space.FD_naive import FiniteDifferencesNaive
    from discretizations.space.FD_corr import FiniteDifferencesCorr
    from discretizations.space.FD_extra import FiniteDifferencesExtra
    from discretizations.space.quad_splines_fv import QuadSplinesFV
    from discretizations.space.fourth_order_fv import FourthOrderFV
    from discretizations.time.backward_euler import BackwardEuler
    from discretizations.time.theta_method import ThetaMethod
    from discretizations.time.RK2 import RK2
    from discretizations.time.RK4 import RK4
    from discretizations.time.Manfredi import Manfredi
    # parameters of the schemes are given to the builder:
    builder = Builder()
    builder.LAMBDA_1 = 1e9  # extremely high lambda is a Dirichlet condition
    builder.LAMBDA_2 = 0. # lambda=0 is a Neumann condition
    builder.D1 = 1.
    builder.D2 = 2.
    builder.C = 0.4
    assert builder.C * builder.DT < 1
        


    discretizations = {}
    time_scheme = Manfredi

    discretizations["FV2"] = (time_scheme, QuadSplinesFV)
    discretizations["FV4"] = (time_scheme, FourthOrderFV)
    discretizations["FD, extra"] = (time_scheme, FiniteDifferencesExtra)
    discretizations["FD, corr=0"] = (time_scheme, FiniteDifferencesNaive)
    #discretizations["FD, corr=1"] = (time_scheme, FiniteDifferencesCorr)

    convergence_factors = {}
    theorical_convergence_factors = {}

    N = 300
    dt = DEFAULT.DT
    ###########
    # Computation of the frequency axis
    ###########
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[int(N//2)] = .5
    # w = 2 pi k T / (N)
    axis_freq = 2 * pi*all_k / N / dt

    kwargs_label_simu = {'label':"Validation by simulation"}
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    ###########
    # for each discretization, a simulation
    ###########
    for name in discretizations:
        time_dis, space_dis = discretizations[name]
        alpha_w = memoised(Builder.frequency_cv_factor, builder, time_dis, space_dis, N, number_samples=50)
        k = 1
        convergence_factors[name] = alpha_w[k+1] / alpha_w[k]

        dis = builder.build(time_dis, space_dis)
        theorical_convergence_factors[name] = \
                dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=0, order_operators=float('inf'),
                        order_equations=float('inf'))
        # continuous = dis.analytic_robin_robin_modified(w=axis_freq,
        #                 order_time=0, order_operators=float('inf'),
        #                 order_equations=float('inf'))
        # plt.plot(axis_freq * dt, continuous, "--", label="Continuous Theorical " + name)
        #axes[0].semilogx(axis_freq * dt, convergence_factors[name], "k--", **kwargs_label_simu)
        axes[0].plot(axis_freq * dt, convergence_factors[name], label=name)
        if kwargs_label_simu: # We only want the legend to be present once
            kwargs_label_simu = {}
        #axes[0].semilogx(axis_freq * dt, theorical_convergence_factors[name], label=name+ " theorical")
    w, rho_theoric = wAndRhoPadeRR(builder)
    axes[0].plot(w*DEFAULT.DT, rho_theoric, "k--", label="theoric")

    axes[0].set_xlabel("Frequency variable $\\omega \\delta t$")
    axes[0].set_ylabel("Convergence factor $\\rho$")
    axes[0].set_title("Various space discretizations with " + time_scheme.__name__)

    axes[1].set_xlabel("Frequency variable $\\omega \\delta t$")
    axes[1].set_ylabel("Convergence factor $\\rho$")
    axes[1].set_title("Various time discretizations with Finite Differences, Corr=0")

    space_scheme = FiniteDifferencesNaive
    discretizations = {}

    discretizations["BackwardEuler"] = (BackwardEuler, space_scheme)
    discretizations["ThetaMethod"] = (ThetaMethod, space_scheme)
    # discretizations["RK2"] = (RK2, space_scheme)
    # discretizations["RK4"] = (RK4, space_scheme)
    discretizations["Manfredi"] = (Manfredi, space_scheme)

    kwargs_label_simu = {'label':"Validation by simulation"}

    for name in discretizations:
        time_dis, space_dis = discretizations[name]
        alpha_w = memoised(Builder.frequency_cv_factor, builder, time_dis, space_dis, N, number_samples=50)
        k = 1
        convergence_factors[name] = alpha_w[k+1] / alpha_w[k]

        dis = builder.build(time_dis, space_dis)
        theorical_convergence_factors[name] = \
                dis.analytic_robin_robin_modified(w=axis_freq,
                        order_time=0, order_operators=float('inf'),
                        order_equations=float('inf'))
        # continuous = dis.analytic_robin_robin_modified(w=axis_freq,
        #                 order_time=0, order_operators=float('inf'),
        #                 order_equations=float('inf'))
        # plt.plot(axis_freq * dt, continuous, "--", label="Continuous Theorical " + name)
        axes[1].semilogx(axis_freq * dt, convergence_factors[name], label=name)

    axes[0].legend()
    axes[1].legend()
    show_or_save("fig_compareSettingsDirichletNeumann")

def fig_rootsManfrediFD():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=[9.6, 2.])
    plt.subplots_adjust(left=.07, bottom=.28, right=.97, top=.85)
    builder = DEFAULT

    r = .1
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    nu_1 = builder.D1
    nu_2 = builder.D2
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)

    def get_z(w):
        return np.exp(-1j*w*dt)

    def Gamma(ab, nu):
        return ab*dt*nu/h**2

    def get_f_g(w, nu):
        z = get_z(w)
        Gamma_a, Gamma_b = Gamma(a, nu), Gamma(b, nu)
        return (z - 1 + r*dt) / (2*Gamma_b**2), (Gamma_a - 2*z*Gamma_b) / (2*Gamma_b**2)

    def square_root_interior(f, g):
        return 1j*np.sqrt(-1*(-(4*(g-4)*(f-2*g+6) - (g-4)**3 - 8*(g-4))/(2*np.sqrt(g**2 - 4*f)) \
                - f + (g-4)**2/2 + 2*g - 8))/2

    def lambda_pp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + np.sqrt(g**2 - 4*f)/4 + square_root_interior(f, g)

    def lambda_pm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + np.sqrt(g**2 - 4*f)/4 - square_root_interior(f, g)

    def lambda_mp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (np.sqrt(g**2 - 4*f)/4 + square_root_interior(f, g))

    def lambda_mm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (np.sqrt(g**2 - 4*f)/4 - square_root_interior(f, g))

    w = np.exp(np.linspace(-6, np.log(pi), 1000))[:-1]

    sigma_1 = np.log(lambda_pm(w, nu_1)) / h
    sigma_2 = np.log(lambda_mp(w, nu_1)) / h
    sigma_3 = np.log(lambda_pp(w, nu_1)) / h
    sigma_4 = np.log(lambda_mm(w, nu_1)) / h

    axes[0].semilogx(w, np.real(sigma_1), label="$\\sigma_1$")

    axes[0].semilogx(w, np.real(sigma_2), label="$\\sigma_2$")
    axes[0].semilogx(w, np.real(sigma_3), label="$\\sigma_3$")
    axes[0].semilogx(w, np.real(sigma_4), label="$\\sigma_4$")

    axes[0].semilogx(w, np.abs(np.real(np.sqrt((r+1j*w)/nu_1))), "k--", label="$\\sigma_j$ continuous")
    axes[0].semilogx(w, np.abs(np.real(-np.sqrt((r+1j*w)/nu_1))), "k--")
    axes[0].set_xlabel("$\\Delta t\\omega$")
    axes[0].set_ylabel("$\\mathfrak{R}(\\sigma)$")
    axes[0].set_title("Real part $\\mathfrak{R}(\\sigma)$")
    axes[0].grid()

    axes[1].semilogx(w, np.imag(sigma_1), label="$\\sigma_1$")

    axes[1].semilogx(w, np.imag(sigma_2), label="$\\sigma_2$")
    axes[1].semilogx(w, np.imag(sigma_3), label="$\\sigma_3$")
    axes[1].semilogx(w, np.imag(sigma_4), label="$\\sigma_4$")

    axes[1].semilogx(w, np.imag(np.sqrt((r+1j*w)/nu_1)), "k--", label="$\\sigma_j$ continuous")
    axes[1].semilogx(w, np.imag(-np.sqrt((r+1j*w)/nu_1)), "k--")
    axes[1].set_xlabel("$\\Delta t\\omega$")
    axes[1].set_ylabel("$Im(\\sigma)$")
    axes[1].set_title("Imaginary part $Im(\\sigma)$")
    axes[1].grid()

    plt.legend()
    show_or_save("fig_rootsManfrediFD")

def wAndRhoPadeFDRR(builder=DEFAULT):
    ###########################
    # equation: (\Gamma_{a,j} = a*dt*nu/h^2, \Gamma_{b,j} = b*dt*nu/h^2)
    #        (z-1+r\Delta t)\lambda_i^2 + 
    #        \left(\Gamma_a - 2z\Gamma_b\right) \lambda \left(\lambda-1\right)^2 + 
    #        2\Gamma_b^2 \left(\lambda-1\right)^4 = 0
    # rewrite it for wolframAlpha: f* x^2 + g*x(x-1)^2 + (x-1)^4 = 0
    # where x = \lambda
    # where f = (z-1+r\Delta t) / (2\Gamma_b^2)
    # and g = (\Gamma_a - 2z\Gamma_b) / (2\Gamma_b^2)
    ##########################"
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.C
    nu_1 = builder.D1
    nu_2 = builder.D2
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2
    h = builder.SIZE_DOMAIN_1 / (builder.M1-1)

    def get_z(w):
        return np.exp(-1j*w*dt)

    def Gamma(ab, nu):
        return ab*dt*nu/h**2

    def get_f_g(w, nu):
        z = get_z(w)
        Gamma_a, Gamma_b = Gamma(a, nu), Gamma(b, nu)
        return (z - 1 + r*dt) / (2*Gamma_b**2), (Gamma_a - 2*z*Gamma_b) / (2*Gamma_b**2)

    def square_root_interior(f, g):
        return np.sqrt(-(4*(g-4)*(f-2*g+6) - (g-4)**3 - 8*(g-4))/(2*sqrt(g**2 - 4*f)) \
                - f + (g-4)**2/2 + 2*g - 8)/2

    def lambda_pp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + np.sqrt(g**2 - 4*f)/4 + square_root_interior(f, g)

    def lambda_pm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 + np.sqrt(g**2 - 4*f)/4 - square_root_interior(f, g)

    def lambda_mp(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (np.sqrt(g**2 - 4*f)/4 + square_root_interior(f, g))

    def lambda_mm(w, nu):
        f, g = get_f_g(w, nu)
        return (4 -g)/4 - (np.sqrt(g**2 - 4*f)/4 - square_root_interior(f, g))
    sigma_1 = np.log(lambda_pm) / h
    sigma_2 = np.log(lambda_mp) / h
    sigma_3 = np.log(lambda_pp) / h
    sigma_4 = np.log(lambda_mm) / h


def wAndRhoPadeRR(builder=DEFAULT):
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    dt= builder.DT
    r = builder.C
    nu_1 = builder.D1
    nu_2 = builder.D2
    L1 = builder.LAMBDA_1
    L2 = builder.LAMBDA_2

    def get_z_s(w):
        z = np.exp(-1j*w*dt)
        return z, (z - 1)/(z*dt)

    def gamma(w):
        z, _ = get_z_s(w)
        return b + z*(b-a)

    def square_root_interior(w):
        z, s = get_z_s(w)
        return 1j*np.sqrt(-1*(1+(a*dt*s)**2 - 2*dt*a**2*r/z - (a**2+1)*dt*s))

    def sigma_plus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s + square_root_interior(w))/(a*np.sqrt(dt*nu))

    def sigma_minus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s - square_root_interior(w))/(a*np.sqrt(dt*nu))

    N = 300
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[int(N//2)] = .5
    # w = 2 pi k T / (N)
    w = 2 * pi*all_k / N / dt

    sigma_1 = sigma_minus(w, nu_1)
    sigma_2 = - sigma_minus(w, nu_2)
    sigma_3 = sigma_plus(w, nu_1)
    sigma_4 = -sigma_plus(w, nu_2)
    assert (np.real(sigma_1) > 0).all()
    assert (np.real(sigma_2) < 0).all()
    assert (np.real(sigma_3) > 0).all()
    assert (np.real(sigma_4) < 0).all()

    z, s = get_z_s(w)
    mu_1 = z*(1 - b*dt*nu_1*sigma_1**2)
    mu_2 = z*(1 - b*dt*nu_2*sigma_2**2)
    mu_3 = z*(1 - b*dt*nu_1*sigma_3**2)
    mu_4 = z*(1 - b*dt*nu_2*sigma_4**2)
    assert (np.linalg.norm(mu_1 - mu_2) < 1e-10) # mu_1 == mu_2
    assert (np.linalg.norm(mu_3 - mu_4) < 1e-10) # mu_3 == mu_4
    gamma_t = (mu_1 - gamma(w))/(mu_1 - mu_3)

    varrho = ((L1 + nu_2*sigma_2)/(L2 + nu_2*sigma_2) * (1 - gamma_t) + \
             (L1 + nu_2*sigma_4)/(L2 + nu_2*sigma_4) * gamma_t) * \
             ((L2 + nu_1*sigma_1)/(L1 + nu_1*sigma_1) * (1 - gamma_t) + \
             (L2 + nu_1*sigma_3)/(L1 + nu_1*sigma_3) * gamma_t)

    return w, np.abs(varrho)

def fig_rhoDNPade():
    import matplotlib.pyplot as plt
    w, varrho = wAndRhoPadeRR()
    plt.semilogx(w*DEFAULT.DT, np.abs(varrho), label="$\\rho_{DN}^{Pade, c}$")
    plt.title("Convergence rate of Pade scheme") 
    plt.grid()
    plt.legend()
    show_or_save("fig_gammaTilde")

def fig_gammaTilde():
    import matplotlib.pyplot as plt
    dt=1.
    a = 1+np.sqrt(2)
    b = 1+1/np.sqrt(2)
    r=.0
    def mu_plus(w):
        z = np.exp(-1j*w*dt)
        s = (z - 1)/z
        return z*(1/np.sqrt(2) * (1-s) + \
                1j*(-b/a**2)*np.sqrt((1+a**2*s**2 - 2*dt*a**2*r/z - (a**2+1)*s)/(-1)))
    def mu_minus(w):
        z = np.exp(-1j*w*dt)
        s = (z - 1)/z
        return z*(1/np.sqrt(2) * (1-s) - \
                1j*(-b/a**2)*np.sqrt((1+a**2*s**2 - 2*dt*a**2*r/z - (a**2+1)*s)/(-1)))

    def gamma(w):
        z = np.exp(-1j*w*dt)
        return b + z*(b-a)
    w = np.linspace(0,pi, 1000)
    plt.plot(w, np.real((mu_minus(w) - gamma(w))/(mu_plus(w) - mu_minus(w))), label="Real part of $\\tilde{\\gamma}$")
    plt.plot(w, np.imag((mu_minus(w) - gamma(w))/(mu_plus(w) - mu_minus(w))), label="Imaginary part of $\\tilde{\\gamma}$")
    # plt.plot(w, np.abs(1-(mu_minus(w) - gamma(w))/(mu_plus(w) - mu_minus(w))), label="modulus of gamma")
    plt.title("Value of $\\tilde{\\gamma}$")
    plt.grid()
    plt.legend()
    show_or_save("fig_gammaTilde")

def fig_rootsManfredi():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=[9.6, 2.])
    plt.subplots_adjust(left=.07, bottom=.28, right=.97, top=.85)


    dt=1.
    a = 1+np.sqrt(2)
    r=.5
    nu_1 = 1.
    nu_2 = 2.
    def get_z_s(w):
        z = np.exp(-1j*w*dt)
        return z, (z - 1)/(z*dt)

    def square_root_interior(w):
        z, s = get_z_s(w)
        return 1j*np.sqrt(-1*(1+(a*dt*s)**2 - 2*dt*a**2*r/z - (a**2+1)*dt*s))

    def sigma_plus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s + square_root_interior(w))/(a*np.sqrt(dt*nu))

    def sigma_minus(w, nu):
        z, s = get_z_s(w)
        return np.sqrt(1+a*dt*s - square_root_interior(w))/(a*np.sqrt(dt*nu))

    w = np.exp(np.linspace(-3, np.log(pi), 1000))[:-1]

    axes[0].semilogx(w, (np.real(sigma_minus(w, nu_1))), label="$\\sigma_1$")

    #axes[0].semilogx(w, np.abs(np.real(-sigma_minus(w, nu_2))), label="$\\sigma_2$")
    axes[0].semilogx(w, (np.real(sigma_plus(w, nu_1))), label="$\\sigma_3$")
    #axes[0].semilogx(w, np.abs(np.real(-sigma_plus(w, nu_2))), label="$\\sigma_4$")

    axes[0].semilogx(w, np.abs(np.real(np.sqrt((r+1j*w)/nu_1))), "k--", label="$\\sigma_j$ continuous")
    #axes[0].semilogx(w, np.abs(np.real(-np.sqrt((r+1j*w)/nu_2))), "k--")
    axes[0].set_xlabel("$\\Delta t\\omega$")
    axes[0].set_ylabel("$\\mathfrak{R}(\\sigma)$")
    axes[0].set_title("Real part $\\mathfrak{R}(\\sigma)$")
    axes[0].grid()

    axes[1].loglog(w, np.abs(np.imag(sigma_minus(w, nu_1))), label="$\\sigma_1$")

    #axes[1].loglog(w, np.abs(np.imag(-sigma_minus(w, nu_2))), label="$\\sigma_2$")
    axes[1].loglog(w, np.abs(np.imag(sigma_plus(w, nu_1))), label="$\\sigma_3$")
    #axes[1].loglog(w, np.abs(np.imag(-sigma_plus(w, nu_2))), label="$\\sigma_4$")

    axes[1].loglog(w, np.abs(np.imag(np.sqrt((r+1j*w)/nu_1))), "k--", label="$\\sigma_j$ continuous")
    #axes[1].loglog(w, np.abs(np.imag(-np.sqrt((r+1j*w)/nu_2))), "k--")
    axes[1].set_xlabel("$\\Delta t\\omega$")
    axes[1].set_ylabel("$Im(\\sigma)$")
    axes[1].set_title("Imaginary part $Im(\\sigma)$")
    axes[1].grid()

    plt.legend()
    show_or_save("fig_rootsManfredi")


#############################################
# Utilities for saving, visualizing, calling functions
#############################################


def set_save_to_png():
    global SAVE_TO_PNG
    SAVE_TO_PNG = True

def set_save_to_pgf():
    global SAVE_TO_PGF
    SAVE_TO_PGF = True

SAVE_TO_PNG = False
SAVE_TO_PGF = False
def show_or_save(name_func):
    """
    By using this function instead plt.show(),
    the user has the possibiliy to use ./figsave name_func
    name_func must be the name of your function
    as a string, e.g. "fig_comparisonData"
    """
    name_fig = name_func[4:]
    directory = "figures_out/"
    if SAVE_TO_PNG:
        print("exporting to directory " + directory)
        import os
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.png')
    elif SAVE_TO_PGF:
        print("exporting to directory " + directory)
        import os
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.pgf')
    else:
        try:
            import matplotlib as mpl
            import os
            os.makedirs(directory, exist_ok=True)
            mpl.rcParams['savefig.directory'] = directory
            fig = plt.get_current_fig_manager()
            fig.canvas.set_window_title(name_fig) 
        except:
            print("cannot set default directory or name")
        plt.show()

##################################################################################
# Filling the dictionnary all_figures with the functions beginning with "fig_":  #
##################################################################################
# First take all globals defined in this module:
for key, glob in globals().copy().items():
    # Then select the names beginning with fig.
    # Note that we don't check if it is a function,
    # So that a user can give a callable (for example, with functools.partial)
    if key[:3] == "fig":
        all_figures[key] = glob
