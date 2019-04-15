#!/usr/bin/python3
import numpy as np
from numpy import pi
from discretizations.finite_difference import FiniteDifferences
from discretizations.finite_volumes import FiniteVolumes
import functools
import figures
from cv_rate import continuous_analytic_rate_robin_neumann
from cv_rate import continuous_analytic_rate_robin_robin
from cv_rate import continuous_best_lam_robin_neumann
from cv_rate import rate_by_z_transform
from cv_rate import analytic_robin_robin
from cv_rate import rate_fast
from cv_rate import raw_simulation
from cv_rate import frequency_simulation
from memoisation import memoised, FunMem

LAMBDA_1_DEFAULT = 0.0
LAMBDA_2_DEFAULT = 0.0

A_DEFAULT = 0.0
C_DEFAULT = 1e-10
D1_DEFAULT = .54
D2_DEFAULT = .6

M1_DEFAULT = 200
M2_DEFAULT = 200

SIZE_DOMAIN_1 = 200
SIZE_DOMAIN_2 = 200

NUMBER_DDT_H2 = .1
T = 10.

DT_DEFAULT = NUMBER_DDT_H2 * (M1_DEFAULT / SIZE_DOMAIN_1)**2 / D1_DEFAULT
# should not be too different from the value with M2, Size_domain2, and D2
TIME_WINDOW_LEN_DEFAULT = int(T / DT_DEFAULT)


def main():
    import sys

    if len(sys.argv) == 1:
        print("to launch tests, use \"python3 cv_rate.py test\"")
        print("Usage: cv_rate {test, graph, optimize, debug, analytic}")
    else:
        # defining discretization:

        finite_difference = FiniteDifferences(A_DEFAULT, C_DEFAULT, D1_DEFAULT,
                                              D2_DEFAULT, M1_DEFAULT,
                                              M2_DEFAULT, SIZE_DOMAIN_1,
                                              SIZE_DOMAIN_2, LAMBDA_1_DEFAULT,
                                              LAMBDA_2_DEFAULT, DT_DEFAULT)

        finite_volumes = FiniteVolumes(A_DEFAULT, C_DEFAULT, D1_DEFAULT,
                                       D2_DEFAULT, M1_DEFAULT, M2_DEFAULT,
                                       SIZE_DOMAIN_1, SIZE_DOMAIN_2,
                                       LAMBDA_1_DEFAULT, LAMBDA_2_DEFAULT,
                                       DT_DEFAULT)

        if sys.argv[1] == "test":
            import tests.test_linear_sys
            import tests.test_schwarz
            import tests.test_finite_volumes
            import tests.test_finite_differences
            import tests.test_optimal_neumann_robin
            test_dict = {
                'linear_sys': tests.test_linear_sys.launch_all_tests,
                'schwarz': tests.test_schwarz.launch_all_tests,
                'fvolumes': tests.test_finite_volumes.launch_all_tests,
                'rate': tests.test_optimal_neumann_robin.launch_all_tests,
                'fdifferences': tests.test_finite_differences.launch_all_tests
            }
            if len(sys.argv) > 2:
                test_dict[sys.argv[2]]()
            else:
                for test_func in test_dict.values():
                    test_func()

        elif sys.argv[1] == "figure":
            # TODO rather than that, use labels of the latex code,
            # and then just write (almost automatically)
            # labels corresponding to numbers (showlabels can be used)

            from label_to_figure import ALL_LABELS
            if len(sys.argv) == 2:
                print("Please enter the id of the figure in the paper.")
                print("The following ids are allowed:")
                print(list(ALL_LABELS.keys()))
            else:
                if sys.argv[2] in ALL_LABELS:
                    print("Function found. Plotting figure...")
                    figures.all_figures[ALL_LABELS[sys.argv[2]]]()
                else:
                    print("id does not exist. Please use one of:")
                    print(list(ALL_LABELS.keys()))

                # TODO enter all figures inside the dictionary

        elif sys.argv[1] == "graph":
            if len(sys.argv) == 2:
                pass
            elif sys.argv[2] == "volumes":
                figures.error_by_taking_continuous_rate_constant_number_dt_h2(
                    finite_volumes, T=T, number_dt_h2=.1, steps=200)
            elif sys.argv[2] == "differences":
                figures.error_by_taking_continuous_rate_constant_number_dt_h2(
                    finite_difference, T=T, number_dt_h2=.1, steps=20)
        elif sys.argv[1] == "optim_by_criblage_plot":
            figures.optim_by_criblage_plot((finite_difference, finite_volumes),
                                           T=T/7, number_dt_h2=.1, steps=200)

        elif sys.argv[1] == "2D_graph":
            error_2D_by_taking_continuous_rate(finite_difference, 50)
        elif sys.argv[1] == "optimize":
            from scipy.optimize import minimize_scalar
            print(
                "rate finite volumes:",
                minimize_scalar(
                    functools.partial(rate_fast, finite_volumes,
                                      TIME_WINDOW_LEN_DEFAULT)))
            print(
                "rate finite differences:",
                minimize_scalar(
                    functools.partial(rate_fast, finite_difference,
                                      TIME_WINDOW_LEN_DEFAULT)))

            D1 = D1_DEFAULT
            D2 = D2_DEFAULT

            def theory_star(wmin, wmax):
                a = (np.sqrt(D1) - np.sqrt(D2)) * (np.sqrt(wmin) +
                                                   np.sqrt(wmax))
                return 1 / (2 * np.sqrt(2)) * (a + np.sqrt(
                    a * a + 8 * np.sqrt(D1 * D2) * np.sqrt(wmin * wmax)))

            print(
                "theory:",
                theory_star(pi / DT_DEFAULT,
                            pi / (DT_DEFAULT * TIME_WINDOW_LEN_DEFAULT)))
        elif sys.argv[1] == "ztransform":
            # print(analytic_robin_robin_finite_differences(w=None, Lambda_1=-5, verbose=True))
            # rate_finite_differences2(-5)

            lambda_min = 1e-9
            lambda_max = 10
            steps = 40
            lambda_1 = np.linspace(lambda_min, lambda_max, steps)
            import matplotlib.pyplot as plt
            plt.plot(
                lambda_1, [
                    analytic_robin_robin(
                        finite_difference, w=pi / DT_DEFAULT,
                        N=TIME_WINDOW_LEN_DEFAULT, Lambda_1=i)
                    for i in lambda_1], "y")
            plt.plot(lambda_1, [
                rate_by_z_transform(finite_difference, i,
                                    TIME_WINDOW_LEN_DEFAULT) for i in lambda_1
            ], "r")
            plt.show()
        elif sys.argv[1] == "analytic":
            print(
                "TODO: lire rapport de Sophie et trouver quels tests faire !")
            import matplotlib.pyplot as plt
            lambda_min = 1e-9
            lambda_max = 10
            steps = 10
            beauty_graph_finite(finite_difference, lambda_min, lambda_max,
                                steps)
        elif sys.argv[1] == "optimal":
            optimal_function_of_h(finite_difference, 10)
        elif sys.argv[1] == "frequency":
            figures.fig_validation_code_frequency_rate()
        elif sys.argv[1] == "frequency_dirichlet_neumann":
            figures.fig_validation_code_frequency_rate_dirichlet_neumann()
        elif sys.argv[1] == "frequency_err":
            figures.fig_validation_code_frequency_error()
        elif sys.argv[1] == "raw_simu":
            figures.fig_error_interface_time_domain_profiles()
        elif sys.argv[1] == "debug":
            import rust_mod
            print(
                "finite differences:",
                rate_fast(finite_difference,
                          60,
                          Lambda_1=3.,
                          Lambda_2=0.,
                          a=0.,
                          c=1e-10,
                          dt=0.1,
                          M1=40,
                          M2=40,
                          function_to_use=np.linalg.norm,
                          seeds=range(150)))
        elif sys.argv[1] == "clean":
            import memoisation
            memoisation.clean()
            print("Memoisation folder cleaned.")
        elif sys.argv[1] == "global":
            print(list(figures.all_figures.keys()))

def error_2D_by_taking_continuous_rate(discretization, N):
    from scipy.optimize import minimize_scalar, minimize
    dt = discretization.DT_DEFAULT
    T = dt * N

    rate_eff = functools.partial(rate_fast, discretization, N)
    def to_minimize_all(x): return rate_fdiff(Lambda_1=x[0], Lambda_2=x[1])
    def to_minimize_rob_neu(x): return rate_fdiff(Lambda_1=x, Lambda_2=0.)
    def to_minimize_one_sided(x): return rate_fdiff(Lambda_1=x, Lambda_2=-x)

    def to_minimize_continuous(l):
        cont = functools.partial(continuous_analytic_rate_robin_robin,
                                 discretization, l[0], l[1])
        ret = np.max([cont(pi / t) for t in np.linspace(dt, T, N)])
        return ret

    x0_retcont = np.array(
        (continuous_best_lam_robin_neumann(discretization, N), 0.))
    print(to_minimize_continuous(x0_retcont))

    ret_cont = minimize(fun=to_minimize_continuous, x0=np.array((1, -2.)))
    optimal_continuous = ret_cont.x
    theoric_cont_rate = ret_cont.fun

    def to_minimize_discrete(l, h):
        M1 = int(discretization.SIZE_DOMAIN_1 / h)
        M2 = int(discretization.SIZE_DOMAIN_2 / h)
        f = functools.partial(discretization.analytic_robin_robin,
                              Lambda_1=l[0],
                              Lambda_2=l[1],
                              M1=M1,
                              M2=M2)
        return max([f(pi / t * 1j) for t in np.linspace(dt, T, N)])
        # return rate(discretization, M1=M1, M2=M2, Lambda_1=l)

    all_h = np.linspace(.05, 10, 50)
    ret_discrete = [
        minimize(
            fun=to_minimize_discrete,
            x0=optimal_continuous,
            args=(h)) for h in all_h]
    optimal_discrete = [ret.x for ret in ret_discrete]
    theorical_rate_discrete = [ret.fun for ret in ret_discrete]

    rate_with_continuous_lambda = []
    rate_with_discrete_lambda = []
    for i in range(all_h.shape[0]):
        print(i)
        M1 = int(discretization.SIZE_DOMAIN_1 / all_h[i])
        M2 = int(discretization.SIZE_DOMAIN_2 / all_h[i])
        rate_with_continuous_lambda += [
            rate_eff(M1=M1,
                     M2=M2,
                     Lambda_1=optimal_continuous[0],
                     Lambda_2=optimal_continuous[1])
        ]
        rate_with_discrete_lambda += [
            rate_eff(M1=M1,
                     M2=M2,
                     Lambda_1=optimal_discrete[i][0],
                     Lambda_2=optimal_discrete[i][1])
        ]
    import matplotlib.pyplot as plt
    plt.semilogx(all_h,
                 rate_with_discrete_lambda,
                 "g",
                 label="Observed rate with discrete optimal $\\Lambda$")
    plt.semilogx(all_h,
                 theorical_rate_discrete,
                 "g--",
                 label="Theorical rate with discrete optimal $\\Lambda$")
    plt.semilogx(all_h,
                 rate_with_continuous_lambda,
                 "r",
                 label="Observed rate with continuous optimal $\\Lambda$")
    plt.hlines(theoric_cont_rate,
               all_h[0],
               all_h[-1],
               "r",
               'dashed',
               label="Theorical rate with continuous optimal $\\Lambda$")
    plt.xlabel("h")
    plt.ylabel("$\\rho$")
    plt.legend()
    plt.title('Error when using continuous Lambda with ' +
              discretization.name())
    plt.show()

def error_by_taking_continuous_rate(discretization, N, steps=50):
    from scipy.optimize import minimize_scalar
    dt = discretization.DT_DEFAULT
    T = dt * N

    def to_minimize_continuous(l):
        cont = functools.partial(
            continuous_analytic_rate_robin_neumann,
            discretization,
            l,
        )
        return np.max([cont(pi / t) for t in np.linspace(dt, T, N)])

    ret_cont = minimize_scalar(fun=to_minimize_continuous)
    optimal_continuous, theoric_cont_rate = ret_cont.x, ret_cont.fun

    def to_minimize_discrete(l, h):
        M1 = int(discretization.SIZE_DOMAIN_1 / h)
        M2 = int(discretization.SIZE_DOMAIN_2 / h)
        f = functools.partial(discretization.analytic_robin_robin,
                              Lambda_1=l,
                              M1=M1,
                              M2=M2)
        return max([f(pi / t * 1j) for t in np.linspace(dt, T, N)])
        # return rate(discretization, M1=M1, M2=M2, Lambda_1=l)

    all_h = np.linspace(-4, 3, steps)
    all_h = np.exp(all_h) / 2.1

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        def func_to_map(x): return minimize_scalar(fun=to_minimize_discrete,
                                                   args=(x))
        ret_discrete = list(executor.map(func_to_map, all_h))
        # ret_discrete = [minimize_scalar(fun=to_minimize_discrete, args=(h)) \
        #    for h in all_h]
    optimal_discrete = [ret.x for ret in ret_discrete]
    theorical_rate_discrete = [ret.fun for ret in ret_discrete]
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

    for i in range(all_h.shape[0]):
        print(i)
        M1 = int(discretization.SIZE_DOMAIN_1 / all_h[i])
        M2 = int(discretization.SIZE_DOMAIN_2 / all_h[i])
        rate_with_continuous_lambda += [
            rate_fast(discretization,
                      N,
                      M1=M1,
                      M2=M2,
                      Lambda_1=optimal_continuous)
        ]
        rate_with_discrete_lambda += [
            rate_fast(discretization,
                      N,
                      M1=M1,
                      M2=M2,
                      Lambda_1=optimal_discrete[i])
        ]

    import matplotlib.pyplot as plt
    plt.semilogx(all_h,
                 rate_with_discrete_lambda,
                 "g",
                 label="Observed rate with discrete optimal $\\Lambda$")
    plt.semilogx(all_h,
                 theorical_rate_discrete,
                 "g--",
                 label="Theorical rate with discrete optimal $\\Lambda$")
    plt.semilogx(all_h,
                 rate_with_continuous_lambda,
                 "r",
                 label="Observed rate with continuous optimal $\\Lambda$")
    plt.hlines(theoric_cont_rate,
               all_h[0],
               all_h[-1],
               "r",
               'dashed',
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


def optimal_function_of_h(discretization, N):
    from scipy.optimize import minimize_scalar
    dt = discretization.DT_DEFAULT
    T = dt * N

    def to_minimize_continuous(l):
        cont = functools.partial(continuous_analytic_rate_robin_neumann,
                                 discretization, l)
        return np.max([
            cont(pi / t) for t in np.linspace(dt, T, TIME_WINDOW_LEN_DEFAULT)
        ])

    optimal_continuous = minimize_scalar(fun=to_minimize_continuous).x

    def to_minimize_discrete(l, h):
        M1 = discretization.SIZE_DOMAIN_1 / h
        M2 = discretization.SIZE_DOMAIN_2 / h
        f = functools.partial(discretization.analytic_robin_robin,
                              Lambda_1=l,
                              M1=M1,
                              M2=M2)
        return max([
            f(pi / t * 1j) for t in np.linspace(dt, T, TIME_WINDOW_LEN_DEFAULT)
        ])

    all_h = np.exp(-np.linspace(-1, 15, 30))
    all_h = np.linspace(0.01, 1, 30)
    ret_discrete = [minimize_scalar(fun=to_minimize_discrete, args=(h)).x
                    for h in all_h]
    import matplotlib.pyplot as plt
    plt.hlines(optimal_continuous,
               all_h[0],
               all_h[-1],
               "k",
               'dashed',
               label='best $\\Lambda$ in continuous')
    plt.plot(all_h, ret_discrete, label='discrete best $\\Lambda$')
    plt.legend()
    plt.show()


PARALLEL = True


def beauty_graph_finite(discretization,
                        lambda_min,
                        lambda_max,
                        steps=100,
                        **kwargs):
    rate_func = functools.partial(rate_fast, discretization,
                                  TIME_WINDOW_LEN_DEFAULT, **kwargs)
    rate_func_normL2 = functools.partial(rate_fast,
                                         discretization,
                                         TIME_WINDOW_LEN_DEFAULT,
                                         function_to_use=np.linalg.norm,
                                         **kwargs)

    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar, minimize

    lambda_1 = np.linspace(lambda_min, lambda_max, steps)
    dt = DT_DEFAULT
    T = dt * TIME_WINDOW_LEN_DEFAULT
    print("> Starting frequency analysis.")
    rho = []
    for t in np.linspace(dt, T, TIME_WINDOW_LEN_DEFAULT):
        rho += [[analytic_robin_robin(discretization,
                                      w=pi / t,
                                      Lambda_1=i, semi_discrete=True,
                                      **kwargs) for i in lambda_1]]
        rho += [[analytic_robin_robin(discretization,
                                      w=-pi / t,
                                      Lambda_1=i, semi_discrete=True,
                                      **kwargs) for i in lambda_1]]

    continuous_best_lam = continuous_best_lam_robin_neumann(
        discretization, TIME_WINDOW_LEN_DEFAULT)

    min_rho, max_rho = np.min(np.array(rho), axis=0), np.max(np.array(rho),
                                                             axis=0)
    best_analytic = lambda_1[np.argmin(max_rho)]

    plt.fill_between(lambda_1,
                     min_rho,
                     max_rho,
                     facecolor="green",
                     label="pi/T < |w| < pi/dt")
    plt.vlines(best_analytic,
               0,
               1,
               "g",
               'dashed',
               label='best $\\Lambda$ with frequency analysis')
    plt.plot(
        lambda_1, [
            analytic_robin_robin(
                discretization, w=0, Lambda_1=i,
                semi_discrete=True, **kwargs) for i in lambda_1], "y")
    plt.vlines(continuous_best_lam,
               0,
               1,
               "k",
               'dashed',
               label='best $\\Lambda$ in continuous case')
    #plt.plot(lambda_1, [rate_by_z_transform(discretization, i, TIME_WINDOW_LEN_DEFAULT) for i in lambda_1], "m")
    #plt.plot(lambda_1, [rate_by_z_transform(discretization, i, TIME_WINDOW_LEN_DEFAULT) for i in lambda_1], "m", label="Z TRANSFORMED ANALYTIC RATE")
    rho = []
    for logt in np.arange(0, 25):
        t = dt * 2.**logt
        rho += [[analytic_robin_robin(discretization,
                                      w=pi / t,
                                      Lambda_1=i, semi_discrete=True,
                                      **kwargs) for i in lambda_1]]
        rho += [[analytic_robin_robin(discretization,
                                      w=-pi / t,
                                      Lambda_1=i, semi_discrete=True,
                                      **kwargs) for i in lambda_1]]
    plt.fill_between(lambda_1,
                     np.min(np.array(rho), axis=0),
                     np.max(np.array(rho), axis=0),
                     facecolor="grey",
                     label="|w| < pi/dt",
                     alpha=0.4)

    print("> Starting simulations (this might take a while)")

    x = np.linspace(lambda_min, lambda_max, steps)
    if PARALLEL:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            rate_f = executor.map(rate_func, x)
            rate_f_L2 = executor.map(rate_func_normL2, x)
    else:
        rate_f = map(rate_func, x)
        rate_f_L2 = map(rate_func_normL2, x)

    print("> Starting minimization in infinite norm.")

    best_linf_norm = minimize_scalar(fun=rate_func).x
    print("> Starting minimization in L2 norm.")
    best_L2_norm = minimize_scalar(fun=rate_func_normL2).x
    plt.plot(x,
             list(rate_f_L2),
             "b",
             label=discretization.name() + ", $L^2$ norm")
    plt.vlines(best_L2_norm,
               0,
               1,
               "b",
               'dashed',
               label='best $\\Lambda$ for $L^2$')
    plt.plot(x,
             list(rate_f),
             "r",
             label=discretization.name() + ", $L^\\infty$ norm")
    plt.vlines(best_linf_norm,
               0,
               1,
               "r",
               'dashed',
               label='best $\\Lambda$ for $L^\\infty$')

    plt.xlabel("$\\Lambda^1$")
    plt.ylabel("$\\rho$")

    #plt.plot(lambda_1, [rate_finite_differences_by_z_transform(i) for i in lambda_1], "r--")
    # plt.plot(lambda_1,
    #       [analytic_robin_robin_finite_volumes(Lambda_1=i) \
    #               for i in lambda_1], "k--", label="Theorical rate")
    plt.title(
        "Global in time: T=" +
        str(TIME_WINDOW_LEN_DEFAULT) +
        "dt\n $D_1$=" +
        str(D1_DEFAULT) +
        ", $D_2$=" +
        str(D2_DEFAULT) +
        ", " +
        "$0=a\\approx c$, $H_1$=-" +
        str(SIZE_DOMAIN_1) +
        ", $H_2$=" +
        str(SIZE_DOMAIN_2) +
        "\n" +
        "$\\Lambda^2$=" +
        str(LAMBDA_2_DEFAULT) +
        ",  dt=" +
        str(DT_DEFAULT) +
        ", h=0.5")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
