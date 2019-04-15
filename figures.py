#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import numpy as np
from numpy import pi
from discretizations.finite_difference import FiniteDifferences
from discretizations.finite_volumes import FiniteVolumes
import functools
from cv_rate import continuous_analytic_rate_robin_neumann
from cv_rate import continuous_analytic_rate_robin_robin
from cv_rate import continuous_best_lam_robin_neumann
from cv_rate import rate_by_z_transform
from cv_rate import analytic_robin_robin
from cv_rate import rate_fast
from cv_rate import raw_simulation
from cv_rate import frequency_simulation
from memoisation import memoised, FunMem
import matplotlib.pyplot as plt

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

"""

all_figures["3"] = functools.partial(beauty_graph_finite,
                                     finite_volumes, 1e-9, 10, 100)
all_figures["4"] = functools.partial(beauty_graph_finite,
                                     finite_difference, 1e-9, 10,
                                     100)
all_figures["5"] = functools.partial(
    error_by_taking_continuous_rate,
    finite_difference,
    100,
    steps=250)
all_figures["6"] = functools.partial(
    error_by_taking_continuous_rate,
    finite_volumes,
    100,
    steps=250)
"""

def values_str(H1, H2, dt, T, D1, D2, a, c, number_dt_h2):
      return '$H_1$=' + \
          str(H1) + \
          ', $H_2$=' + \
          str(H2) + \
          ', T = ' + \
          str(T) + \
          ', dt = ' + str(dt) +', \n$D_1$=' + \
          str(D1) + \
          ', $D_2$=' + \
          str(D2) + \
          ', a=' + str(a)+ \
          ', c=' + str(c) + \
          ', \n$\\frac{D_1 dt}{h^2}$='+ str(number_dt_h2)


def fig_error_interface_time_domain_profiles():
    NUMBER_DDT_H2 = .1
    M1_DEFAULT = 200
    SIZE_DOMAIN_1 = 200
    D1_DEFAULT = .6
    DT_DEFAULT = NUMBER_DDT_H2 * (M1_DEFAULT / SIZE_DOMAIN_1)**2 / D1_DEFAULT
    finite_difference = FiniteDifferences(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.54,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)

    finite_volumes = FiniteVolumes(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.54,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)


    raw_plot((finite_difference, finite_volumes), 100)
    plt.title(values_str(200, -200, DT_DEFAULT, 100*DT_DEFAULT,
        D1_DEFAULT, .54, 0, 0, NUMBER_DDT_H2))
    plt.show()


def fig_validation_code_frequency_error():
    NUMBER_DDT_H2 = .1
    M1_DEFAULT = 200
    SIZE_DOMAIN_1 = 200
    D1_DEFAULT = .6
    DT_DEFAULT = NUMBER_DDT_H2 * (M1_DEFAULT / SIZE_DOMAIN_1)**2 / D1_DEFAULT
    finite_difference = FiniteDifferences(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.54,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)
    finite_volumes = FiniteVolumes(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.54,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)

    analysis_frequency_error((finite_volumes, ), 100)
    plt.title(values_str(200, -200, DT_DEFAULT, 100*DT_DEFAULT,
        D1_DEFAULT, .54, 0, 0, NUMBER_DDT_H2))
    plt.show()


def fig_validation_code_frequency_rate_dirichlet_neumann():
    NUMBER_DDT_H2 = .1
    M1_DEFAULT = 200
    SIZE_DOMAIN_1 = 200
    D1_DEFAULT = .54
    DT_DEFAULT = NUMBER_DDT_H2 * (M1_DEFAULT / SIZE_DOMAIN_1)**2 / D1_DEFAULT
    finite_difference = FiniteDifferences(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.6,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)
    finite_volumes = FiniteVolumes(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.6,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)

    analysis_frequency_rate((finite_difference, finite_volumes), 100, lambda_1=-1e13)
    plt.title(values_str(200, -200, DT_DEFAULT, 100*DT_DEFAULT,
        D1_DEFAULT, .54, 0, 0, NUMBER_DDT_H2))
    plt.show()

def fig_validation_code_frequency_rate():
    NUMBER_DDT_H2 = .1
    M1_DEFAULT = 200
    SIZE_DOMAIN_1 = 200
    D1_DEFAULT = .6
    DT_DEFAULT = NUMBER_DDT_H2 * (M1_DEFAULT / SIZE_DOMAIN_1)**2 / D1_DEFAULT
    finite_difference = FiniteDifferences(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.54,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)
    finite_volumes = FiniteVolumes(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.54,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=200,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)

    analysis_frequency_rate((finite_difference, finite_volumes), 100)
    plt.title(values_str(200, -200, DT_DEFAULT, 100*DT_DEFAULT,
        D1_DEFAULT, .54, 0, 0, NUMBER_DDT_H2))
    plt.show()

def fig_figure_2():  # TODO change name
    #TODO gérer ce problème de "transposition"
    # (lambda_plus qui est en fait lambda_moins dans cv_rate)

    NUMBER_DDT_H2 = .1
    M1_DEFAULT = 400
    SIZE_DOMAIN_1 = 200
    D1_DEFAULT = .6
    DT_DEFAULT = NUMBER_DDT_H2 * (M1_DEFAULT / SIZE_DOMAIN_1)**2 / D1_DEFAULT
    finite_difference = FiniteDifferences(A_DEFAULT=0., C_DEFAULT=1e-10,
                                          D1_DEFAULT=D1_DEFAULT, D2_DEFAULT=.54,
                                          M1_DEFAULT=M1_DEFAULT, M2_DEFAULT=400,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN_1,
                                          SIZE_DOMAIN_2=200,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)
    TIME_WINDOW_LEN = NUMBER_DDT_H2
    plot_3D_profile(finite_difference, TIME_WINDOW_LEN)


def analysis_frequency_error(discretization, N):
    def continuous_analytic_error_neumann(discretization, w):
        D1 = discretization.D1_DEFAULT
        D2 = discretization.D2_DEFAULT
        # sig1 is \sigma^1_{+}
        sig1 = np.sqrt(np.abs(w) / (2 * D1)) * (1 + np.abs(w) / w * 1j)
        # sig2 is \sigma^2_{-}
        sig2 = -np.sqrt(np.abs(w) / (2 * D2)) * (1 + np.abs(w) / w * 1j)
        return D1 * sig1 / (D2 * sig2)

    colors = ['r', 'g', 'y', 'm']
    for dis, col, col2 in zip(discretization, colors, colors[::-1]):
        # first: find a correct lambda : we take the optimal yielded by
        # continuous analysis

        # continuous_best_lam_robin_neumann(dis, N)
        lambda_1 = 0.6139250052109033
        #print("rate", dis.name(), ":", rate(dis, N, Lambda_1=lambda_1))
        dt = dis.DT_DEFAULT
        axis_freq = np.linspace(-pi / dt, pi / dt, N)

        frequencies = memoised(frequency_simulation,
                               dis,
                               N,
                               Lambda_1=lambda_1,
                               number_samples=135)
        # plt.plot(axis_freq, frequencies[0], col2+"--", label=" initial frequency ")
        # plt.plot(axis_freq, frequencies[1], col, label=dis.name()+" after 1 iteration")
        #plt.plot(axis_freq, frequencies[1], col+"--", label=dis.name()+" frequential error after the first iteration")
        plt.semilogy(axis_freq,
                 frequencies[1],
                 col,
                 label=dis.name() +
                 " observed error after 1 iteration (with fft on errors)")
        plt.semilogy(axis_freq,
                 frequencies[2],
                 col2,
                 label=dis.name() +
                 " observed error after 2 iteration (with fft on errors)")

        real_freq_discrete = np.fft.fftshift(np.array([
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=False,
                                 N=N) for w in axis_freq
        ]))

        real_freq_semidiscrete = np.array([
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=True,
                                 N=N) for w in axis_freq
        ])

        real_freq_continuous = np.array([
            continuous_analytic_rate_robin_neumann(dis, w=w, Lambda_1=lambda_1)
            for w in axis_freq
        ])

        plt.semilogy(axis_freq,
                 real_freq_continuous * frequencies[1],
                 'b',
                 label="theoric error after 2 iteration(continuous)")
        plt.semilogy(axis_freq,
                 real_freq_semidiscrete * frequencies[1],
                 col,
                 linestyle='dotted',
                 label=dis.name() + " theoric error after 2 iteration (semi-discrete)")
        plt.semilogy(axis_freq,
                 real_freq_discrete * frequencies[1],
                 'k',
                 linestyle='dashed',
                 label=dis.name() + " theoric error after 2 iteration(discrete)")

    plt.xlabel("$\\omega$")
    plt.ylabel("error $\\hat{e}$")
    plt.legend()

def optim_by_criblage_plot(discretization, T, number_dt_h2, steps=50):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    from scipy.optimize import minimize_scalar

    all_h = np.linspace(-2.2, 0, steps)
    all_h = np.exp(all_h[::-1]) / 2.1

    h_plot = all_h[0]
    lambdas = np.linspace(-2, 100, 300)
    color = ['r', 'g']
    for dis, col in zip(discretization, color):
        plt.plot(lambdas, [to_minimize_analytic_robin_robin2(l,
            h_plot, dis, number_dt_h2, T) for l in lambdas], col,
            label=dis.name() + " semi-discrete $\max_\\omega{\\rho}$")
    plt.plot(lambdas, [to_minimize_continuous_analytic_rate_robin_neumann2(l,
        h_plot, discretization[0], number_dt_h2, T) for l in lambdas], 'k',
        label="Continuous $\max_\\omega{\\rho}$")
    plt.xlabel("$\\Lambda$")
    plt.ylabel("$\\rho$")
    plt.legend()
    plt.show()



def analysis_frequency_rate(discretization, N, lambda_1=0.6139250052109033):
    def continuous_analytic_error_neumann(discretization, w):
        D1 = discretization.D1_DEFAULT
        D2 = discretization.D2_DEFAULT
        # sig1 is \sigma^1_{+}
        sig1 = np.sqrt(np.abs(w) / (2 * D1)) * (1 + np.abs(w) / w * 1j)
        # sig2 is \sigma^2_{-}
        sig2 = -np.sqrt(np.abs(w) / (2 * D2)) * (1 + np.abs(w) / w * 1j)
        return D1 * sig1 / (D2 * sig2)

    colors = ['r', 'g', 'y', 'm']
    for dis, col, col2 in zip(discretization, colors, colors[::-1]):
        # first: find a correct lambda : we take the optimal yielded by
        # continuous analysis

        # continuous_best_lam_robin_neumann(dis, N)
        #print("rate", dis.name(), ":", rate(dis, N, Lambda_1=lambda_1))
        dt = dis.DT_DEFAULT
        axis_freq = np.linspace(-pi / dt, pi / dt, N)

        frequencies = memoised(frequency_simulation,
                               dis,
                               N,
                               Lambda_1=lambda_1,
                               number_samples=135)
        # plt.plot(axis_freq, frequencies[0], col2+"--", label=" initial frequency ")
        # plt.plot(axis_freq, frequencies[1], col, label=dis.name()+" after 1 iteration")
        #plt.plot(axis_freq, frequencies[1], col+"--", label=dis.name()+" frequential error after the first iteration")
        plt.plot(axis_freq,
                 frequencies[2] / frequencies[1],
                 col,
                 label=dis.name() +
                 " observed rate (with fft on errors)")
        real_freq_discrete = np.fft.fftshift(np.array([
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=False,
                                 N=N) for w in axis_freq
        ]))

        real_freq_semidiscrete = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=True,
                                 N=N) for w in axis_freq
        ]

        real_freq_continuous = [
            continuous_analytic_rate_robin_neumann(dis, w=w, Lambda_1=lambda_1)
            for w in axis_freq
        ]

        plt.plot(axis_freq,
                 real_freq_continuous,
                 'b',
                 label="theoric rate (continuous)")
        plt.plot(axis_freq,
                 real_freq_semidiscrete,
                 col,
                 linestyle='dotted',
                 label=dis.name() + " theoric rate (semi-discrete)")
        plt.plot(axis_freq,
                 real_freq_discrete,
                 'k',
                 linestyle='dashed',
                 label=dis.name() + " theoric rate (discrete)")

    plt.xlabel("$\\omega$")
    plt.ylabel("convergence rate $\\rho$")
    plt.legend()


def raw_plot(discretization, N, number_samples=1000):
    colors = ['r', 'g', 'k', 'b', 'y', 'm']
    colors3 = ['k', 'b']
    for dis, col, col2, col3 in zip(discretization, colors, colors[::-1], colors3):
        # first: find a correct lambda : we take the optimal yielded by
        # continuous analysis

        # continuous_best_lam_robin_neumann(dis, N)
        lambda_1 = 0.6139250052109033
        #print("rate", dis.name(), ":", rate(dis, N, Lambda_1=lambda_1))
        dt = dis.DT_DEFAULT
        axis_freq = np.linspace(-pi / dt, pi / dt, N)

        times = memoised(raw_simulation,
                         dis,
                         N,
                         Lambda_1=lambda_1,
                         number_samples=number_samples)
        plt.plot(times[0], col, label=dis.name() + " first iteration")
        plt.plot(times[1], col2, label=dis.name() + " second")
        plt.plot(times[2], col3, label=dis.name() + " third")
        #plt.plot(np.fft.fftshift(np.fft.fft(times[1])), col2, label=dis.name()+"second")
        #plt.plot(np.fft.fftshift(np.fft.fft(times[2])), 'b', label=dis.name()+"third")
        # plt.plot(axis_freq, frequencies[0], col2+"--", label=" initial frequency ")
        # plt.plot(axis_freq, frequencies[0], col2+"--", label=" initial frequency ")
        # plt.plot(axis_freq, frequencies[1], col, label=dis.name()+" after 1 iteration")
        #plt.plot(axis_freq, frequencies[2]/frequencies[1], col+"--", label=dis.name()+" frequential convergence rate")
        """
        real_freq_discrete = [analytic_robin_robin(dis, w=w,
            Lambda_1=lambda_1) for w in axis_freq]
        real_freq_continuous = [continuous_analytic_rate_robin_neumann(dis,
            w=w, Lambda_1=lambda_1) for w in axis_freq]
        """
        #plt.plot(axis_freq, real_freq_continuous, col2, label="theoric rate (continuous)")
        #plt.plot(axis_freq, real_freq_discrete, col, label="theoric rate (discrete)")

    plt.xlabel("$\\omega$")
    plt.ylabel("Error $\\hat{e}_0$")
    plt.legend()

def plot_3D_profile(dis, N):
    rate_fdiff = functools.partial(rate_fast, dis, N)
    dt = dis.DT_DEFAULT
    assert continuous_analytic_rate_robin_neumann(
        dis, 2.3, pi / dt) - continuous_analytic_rate_robin_robin(
            dis, 2.3, 0., pi / dt) < 1e-13
    cont = functools.partial(continuous_analytic_rate_robin_robin, dis)

    def fun(x):
        return max([cont(Lambda_1=x[0], Lambda_2=x[1], w=pi / (n * dt))
                    for n in (1, N)])

    def fun_me(x):
        return max([analytic_robin_robin(dis,
                                         Lambda_1=x[0], Lambda_2=x[1],
                                         w=pi / (n * dt), semi_discrete=True)
                    for n in (1, N)])

    fig, ax = plot_3D_square(fun, -40., 40., 1.5, subplot_param=211)
    ax.set_title("Continuous case: convergence rate ")
    fig, ax = plot_3D_square(fun_me,
                             -40.,
                             40.,
                             1.5,
                             fig=fig,
                             subplot_param=212)
    ax.set_title(dis.name() + " case: convergence rate")
    fig.show()
    input()


"""
    fun must take a tuple of two parameters.
"""


def plot_3D_square(fun, bmin, bmax, step, fig=None, subplot_param=111):
    from mpl_toolkits.mplot3d import Axes3D
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot_param, projection='3d')
    N = int(abs(bmin - bmax) / step)
    X = np.ones((N, 1)) @ np.reshape(np.linspace(bmin, bmax, N), (1, N))
    Y = np.copy(X.T)
    Z = np.array([[fun((x, y)) for x, y in zip(linex, liney)]
                  for linex, liney in zip(X, Y)])
    ax.plot_surface(X, Y, Z)
    return fig, ax


def get_dt_N(h, number_dt_h2, T, D1):
    dt = number_dt_h2 * h * h / D1
    N = int(T / dt)
    if N <= 1:
        print("ERROR: N is too small (<2): h=", h)
    return dt, N


def to_minimize_continuous_analytic_rate_robin_neumann(l,
        h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    cont = functools.partial(
        continuous_analytic_rate_robin_neumann,
        discretization, l)
    return np.max([cont(pi / t) for t in np.linspace(dt, T, N)])


def to_minimize_analytic_robin_robin(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(discretization.analytic_robin_robin,
                          Lambda_1=l,
                          M1=M1,
                          M2=M2,
                          dt=dt)
    return max([f(pi / t * 1j) for t in np.linspace(dt, T, N)])


# The function can be passed in parameters of memoised:
to_minimize_analytic_robin_robin2 = FunMem(to_minimize_analytic_robin_robin)
to_minimize_continuous_analytic_rate_robin_neumann2 = \
        FunMem(to_minimize_continuous_analytic_rate_robin_neumann)


def error_by_taking_continuous_rate_constant_number_dt_h2(
        discretization, T, number_dt_h2, steps=50):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    from scipy.optimize import minimize_scalar

    dt = discretization.DT_DEFAULT
    N = int(T / dt)
    if N <= 1:
        print("ERROR BEGINNING: N is too small (<2)")

    all_h = np.linspace(-2.2, 2, steps)
    all_h = np.exp(all_h[::-1]) / 2.1

    def func_to_map(x): return memoised(minimize_scalar,
        fun=to_minimize_analytic_robin_robin2,
        args=(x, discretization, number_dt_h2, T))
    print("Computing lambdas in discrete framework.")
    ret_discrete = list(map(func_to_map, all_h))
    # ret_discrete = [minimize_scalar(fun=to_minimize_discrete, args=(h)) \
    #    for h in all_h]
    optimal_discrete = [ret.x for ret in ret_discrete]
    theorical_rate_discrete = [ret.fun for ret in ret_discrete]

    def func_to_map_cont(x): return memoised(minimize_scalar,
        fun=to_minimize_continuous_analytic_rate_robin_neumann2,
        args=(x, discretization, number_dt_h2, T))
    print("Computing lambdas in discrete framework.")
    ret_continuous = list(map(func_to_map_cont, all_h))
    # ret_discrete = [minimize_scalar(fun=to_minimize_discrete, args=(h)) \
    #    for h in all_h]
    optimal_continuous = [ret.x for ret in ret_continuous]
    theorical_cont_rate = [ret.fun for ret in ret_continuous]
    """
    fun_to_map = lambda h, l: rate(discretization, N,
            M1=int(discretization.SIZE_DOMAIN_1/h),
            M2=int(discretization.SIZE_DOMAIN_2/h),
            Lambda_1=l)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rate_with_continuous_lambda = list(executor.map(fun_to_map, all_h,
                                            np.ones_like(all_h)*optimal_continuous))
        rate_with_discrete_lambda = list(executor.map(fun_to_map, all_h,
                                            optimal_discrete))
    """
    rate_with_continuous_lambda = []
    rate_with_discrete_lambda = []

    try:
        for i in range(all_h.shape[0]):
            dt, N = get_dt_N(all_h[i], number_dt_h2, T,
                             discretization.D1_DEFAULT)
            print("h number:", i)
            M1 = int(discretization.SIZE_DOMAIN_1 / all_h[i])
            M2 = int(discretization.SIZE_DOMAIN_2 / all_h[i])
            rate_with_continuous_lambda += [
                    memoised(rate_fast, discretization,
                             N,
                             M1=M1,
                             M2=M2,
                             Lambda_1=optimal_continuous[i],
                             dt=dt)
                ]
            rate_with_discrete_lambda += [
                memoised(rate_fast, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_discrete[i],
                         dt=dt)
            ]
    except:
        pass

    import matplotlib.pyplot as plt
    plt.semilogx(all_h[:len(rate_with_discrete_lambda)],
                 rate_with_discrete_lambda,
                 "g",
                 label="Observed rate with discrete optimal $\\Lambda$")
    plt.semilogx(all_h,
                 theorical_rate_discrete,
                 "g--",
                 label="Theorical rate with discrete optimal $\\Lambda$")
    plt.semilogx(all_h[:len(rate_with_continuous_lambda)],
                 rate_with_continuous_lambda,
                 "r",
                 label="Observed rate with continuous optimal $\\Lambda$")
    plt.semilogx(all_h,
                 theorical_cont_rate,
                 "r--",
                 label="Theorical rate with continuous optimal $\\Lambda$")
    plt.xlabel("h")
    plt.ylabel("$\\rho$")
    plt.legend()
    plt.title('Error when using continuous Lambda with ' +
              discretization.name() +
              ' (Robin-Neumann)' +
              '\n$H_1$=' +
              str(discretization.SIZE_DOMAIN_1) +
              ', $H_2$=' +
              str(discretization.SIZE_DOMAIN_2) +
              ', T = ' +
              str(N) +
              'dt, $D_1$=' +
              str(discretization.D1_DEFAULT) +
              ', $D_2$=' +
              str(discretization.D2_DEFAULT) +
              ', a=c=0')
    plt.show()


######################################################################
# Filling the dictionnary with the functions beginning with "fig_":  #
######################################################################
# First take all globals defined in this module:
for key, glob in globals().copy().items():
    # Then select the names beginning with fig_.
    # Note that we don't check if it is a function,
    # So that a user can give a callable (for example, with functools.partial)
    if key[:4] == "fig_":
        all_figures[key] = glob
