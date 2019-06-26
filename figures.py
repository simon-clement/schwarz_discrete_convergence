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
from discretizations.finite_difference_no_corrective_term \
        import FiniteDifferencesNoCorrectiveTerm
from discretizations.finite_difference_naive_neumann \
        import FiniteDifferencesNaiveNeumann
import functools
import cv_rate
from cv_rate import continuous_analytic_rate_robin_neumann
from cv_rate import continuous_analytic_rate_robin_robin
from cv_rate import analytic_robin_robin
from cv_rate import rate_fast
from cv_rate import raw_simulation
from cv_rate import frequency_simulation, frequency_simulation_slow
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


class Default():
    """
        AVOID AT ALL COST CHANGING DEFAULT : it will change all figures and invalidate all cache.
        Remember to keep the synchronisation between this class and the PDF.
    """
    def __init__(self):
        self.COURANT_NUMBER = .1
        self.T = 100.
        self.M1 = 200
        self.M2 = 200
        self.SIZE_DOMAIN_1 = 200
        self.SIZE_DOMAIN_2 = 200
        self.D1 = .54
        self.D2 = .6
        self.DT = self.COURANT_NUMBER * (self.SIZE_DOMAIN_1 / self.M1)**2 / self.D1
        self.A = 0.
        self.C = 1e-10
        self.LAMBDA_1 = 0.
        self.LAMBDA_2 = 0.
        self.N = int(self.T/self.DT)

    def new(self, Discretisation):
        return Discretisation(A_DEFAULT=self.A, C_DEFAULT=self.C,
                              D1_DEFAULT=self.D1, D2_DEFAULT=self.D2,
                              M1_DEFAULT=self.M1, M2_DEFAULT=self.M2,
                              SIZE_DOMAIN_1=self.SIZE_DOMAIN_1,
                              SIZE_DOMAIN_2=self.SIZE_DOMAIN_2,
                              LAMBDA_1_DEFAULT=self.LAMBDA_1,
                              LAMBDA_2_DEFAULT=self.LAMBDA_2,
                              DT_DEFAULT=self.DT)

DEFAULT = Default()


def fig_rho_robin_neumann():
    """
        This is the shape of \\rho when using a Robin-Neumann interface.
        We can see with one of the curve that the maximum is not always 0 or \\infty,
        making it way harder to analyse.
    """
    def f(r, w, Lambdaprime):
        return np.abs(Lambdaprime*w*1j + 1 - np.sqrt(1+r*w*1j))

    w = np.linspace(-30,30, 1000)
    r = 0.9
    all_Lambdaprime = np.linspace(-1.1, 1, 5)
    for Lambdaprime in all_Lambdaprime:
        plt.plot(w, f(r,w, Lambdaprime)/f(1,w, Lambdaprime), label="$\\Lambda'="+
                str(round(Lambdaprime, 3))+"$", )
    plt.xlabel("$\\omega$")
    plt.ylabel("$\\hat{\\rho}$")
    plt.legend()
    show_or_save("fig_rho_robin_neumann")


def fig_want_to_show_decreasing(c=0.4):
    """
        We would like to show that the functions plot on this figure decrease with r.
        it seems true.
        The plot are done for one particular reaction coefficient,
        and we chose some frequencies to plot the ratio for theses frequencies.
        The figure @fig_want_to_show_decreasing_irregularities
        show that for very high frequencies it may not be decreasing anymore,
        but it seems to me that the problems are only from the floating precision.
    """
    def f(r,w):
        return 1 + np.sqrt(r*w + 1) - np.sqrt(2*np.sqrt(r*w + 1) + c*(np.sqrt(4*r + c*c) - c)*w+2)
    def partial(r, w):
        numerator = r / np.sqrt(r*w + 1) + c*(np.sqrt(4*r+c*c)-c)
        denominator = 2*np.sqrt(2*np.sqrt(r*w+1) + c*(np.sqrt(4*r+c*c)-c)*w+2)
        return r / (2*np.sqrt(r*w+1)) - numerator / denominator

    allllll_w = np.linspace(0, 20, 20)
    r = np.linspace(0,1, 200)
    for w in allllll_w:
        plt.semilogy(r, partial(r,w)/f(r,w))
    plt.xlabel("$\\bar{r}$")
    plt.ylabel("$\\frac{\\frac{\\partial f}{\\partial \\bar{\\omega}}}{f}$")
    show_or_save("fig_want_to_show_decreasing")

def fig_want_to_show_decreasing_irregularities(c=0.4):
    def f(r,w):
        return 1 + np.sqrt(r*w + 1) - np.sqrt(2*np.sqrt(r*w + 1) + c*(np.sqrt(4*r + c*c) - c)*w+2)
    def partial(r, w):
        numerator = r / np.sqrt(r*w + 1) + c*(np.sqrt(4*r+c*c)-c)
        denominator = 2*np.sqrt(2*np.sqrt(r*w+1) + c*(np.sqrt(4*r+c*c)-c)*w+2)
        return r / (2*np.sqrt(r*w+1)) - numerator / denominator

    r = np.linspace(0,1, 200)
    w = 1e-6
    plt.plot(r, (f(r,w)))
    #plt.plot(r, np.log(np.abs(partial(r,w))))
    plt.xlabel("$\\bar{r}$")
    plt.ylabel("$\\frac{\\frac{\\partial f}{\\partial \\bar{\\omega}}}{f}$")
    show_or_save("fig_want_to_show_decreasing_irregularities")

def fig_w5_rob_neumann_volumes():
    """
        The green zone is the zone where for each value, there exist a frequency
        between T/dt and 1/dt such that the convergence rate is equal to this value.
        the function we minimize is therefore the top border of this zone.
        Results of figure @fig_error_by_taking_continuous_rate_constant_number_dt_h2_vol
        can be seen here: when choosing the optimal lambda of the continuous analysis,
        (take the intersection between red line and black dashed line) we have a
        value of \\rho that is not really the lowest value \\rho can take.

        We can see on this figure that the min-max frequency analysis is not always
        exactly the same analysis we would do in the time domain.
    """
    import rust_mod
    finite_volumes = DEFAULT.new(FiniteVolumes)
    w5_robin_neumann(finite_volumes)
    show_or_save("fig_w5_rob_neumann_volumes")

def fig_w5_rob_neumann_diff_extrapolation():
    """
        The green zone is the zone where for each value, there exist a frequency
        between T/dt and 1/dt such that the convergence rate is equal to this value.
        the function we minimize is therefore the top border of this zone.
        Results of figure @fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff
        can be seen here: when choosing the optimal lambda of the continuous analysis,
        (take the intersection between red line and black dashed line) we have a
        value of \\rho that is not really the lowest value \\rho can take.

        We can see on this figure that the min-max frequency analysis is not always
        exactly the same analysis we would do in the time domain.
    """
    import rust_mod
    finite_difference = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    w5_robin_neumann(finite_difference)
    show_or_save("fig_w5_rob_neumann_diff_extrapolation")

def fig_w5_rob_neumann_diff_naive():
    """
        The green zone is the zone where for each value, there exist a frequency
        between T/dt and 1/dt such that the convergence rate is equal to this value.
        the function we minimize is therefore the top border of this zone.
        Results of figure @fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff_naive
        can be seen here: when choosing the optimal lambda of the continuous analysis,
        (take the intersection between red line and black dashed line) we have a
        value of \\rho that is not really the lowest value \\rho can take.

        We can see on this figure that the min-max frequency analysis is not always
        exactly the same analysis we would do in the time domain.
    """
    import rust_mod
    finite_difference = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    w5_robin_neumann(finite_difference)
    show_or_save("fig_w5_rob_neumann_diff_naive")

def fig_w5_rob_neumann_diff():
    """
        The green zone is the zone where for each value, there exist a frequency
        between T/dt and 1/dt such that the convergence rate is equal to this value.
        the function we minimize is therefore the top border of this zone.
        Results of figure @fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff
        can be seen here: when choosing the optimal lambda of the continuous analysis,
        (take the intersection between red line and black dashed line) we have a
        value of \\rho that is not really the lowest value \\rho can take.

        We can see on this figure that the min-max frequency analysis is not always
        exactly the same analysis we would do in the time domain.
    """
    import rust_mod
    finite_difference = DEFAULT.new(FiniteDifferences)
    w5_robin_neumann(finite_difference)
    show_or_save("fig_w5_rob_neumann_diff")


def w5_robin_neumann(discretization):
    lambda_min = 1e-9
    lambda_max = 5
    steps = 100
    courant_numbers = [0.1, 1.]
    import matplotlib.pyplot as plt
    # By default figsize is 6.4, 4.8
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8])

    to_map = functools.partial(beauty_graph_finite, discretization,
                               lambda_min, lambda_max, steps)
        
    to_map(courant_numbers[0], fig, axes[0], legend=False)
    to_map(courant_numbers[1], fig, axes[1], legend=True)
    """
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(to_map, courant_numbers, figures, axes))
    """


def fig_schwarz_method_converging_to_full_domain_solution_global():
    """
        Evolution of errors across schwarz iterations.
        The corrective term allows us to converge to the precision machine.
        The other discretizations of the neumann condition don't
        converge to the full domain solution.
        All the methods have the same convergence rate,
        because we use Dirichlet-Neumann algorithm and
        the bottleneck are the low frequencies
        (where the convergence rate are all the same)
    """
    discretizations = (DEFAULT.new(FiniteDifferencesNaiveNeumann),
                       DEFAULT.new(FiniteDifferencesNoCorrectiveTerm),
                       DEFAULT.new(FiniteDifferences))
    colors = ['k:', 'y--', 'r']
    from tests.test_schwarz import schwarz_convergence_global
    fig, ax = plt.subplots()

    for dis, col in zip(discretizations, colors):
        errors = memoised(schwarz_convergence_global,dis)
        ax.semilogy(errors, col, label=dis.name())
    ax.set_title("Convergence de la méthode de Schwarz")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$\\max_t(e)$")
    ax.legend()
    show_or_save("fig_schwarz_method_converging_to_full_domain_solution_global")

def fig_schwarz_method_converging_to_full_domain_solution_local():
    discretizations = (FiniteDifferences(),
               FiniteDifferencesNaiveNeumann(),
               FiniteDifferencesNoCorrectiveTerm())
    colors = ['r', 'k', 'y']
    from tests.test_schwarz import schwarz_convergence
    for dis, col in zip(discretizations, colors):
        errors = schwarz_convergence(dis)
        plt.semilogy(errors, col, label=dis.name())
    plt.legend()
    plt.title("Local in time Dirichlet-Neumann convergence of the Schwarz method")
    show_or_save("fig_schwarz_method_converging_to_full_domain_solution_local")

def fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff():
    """
        We see on this figure the utility of making the discrete analysis.
        For a given h, we compute the optimal free parameter $\\Lambda^1$
        of the robin interface condition.
        If we compute it with the continuous framework, we get always the same 
        $\\Lambda^1$.
        We can then compare the observed convergence rate of the parameter
        obtained in the continuous framework and the parameter obtained in the discrete framework.
        The theorical convergence rate plotted on the figure is obtained with the discrete formula :
        this is why it changes for the continuous framework when we change h.

        In the case of the finite difference with a corrective term, it is better to use the continous framework.
        This is explained in details in the PDF : the corrective term blocks the convergence of high frequencies.
    """
    finite_difference = DEFAULT.new(FiniteDifferences)
    T = DEFAULT.T
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)

    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[0], finite_difference,
                                                          T=T, number_dt_h2=.1,
                                                          steps=50,
                                                          number_samples=70,
                                                          bounds_h=(-2.5,1.), legend=False)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_difference,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=500,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.))
    show_or_save("fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff")

def fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff_naive():
    """
        We see on this figure the utility of making the discrete analysis.
        For a given h, we compute the optimal free parameter $\\Lambda^1$
        of the robin interface condition.
        If we compute it with the continuous framework, we get always the same 
        $\\Lambda^1$.
        We can then compare the observed convergence rate of the parameter
        obtained in the continuous framework and the parameter obtained in the discrete framework.
        The theorical convergence rate plotted on the figure is obtained with the discrete formula :
        this is why it changes for the continuous framework when we change h.

        As expected, performing the optimization in the discrete framework gives better results,
        since it is closer to reality.
    """
    finite_difference = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    T = DEFAULT.T
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[0], finite_difference,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=70,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.), legend=False)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_difference,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=500,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.))
    show_or_save("fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff_naive")

def fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff_no_corr():
    """
        We see on this figure the utility of making the discrete analysis.
        For a given h, we compute the optimal free parameter $\\Lambda^1$
        of the robin interface condition.
        If we compute it with the continuous framework, we get always the same 
        $\\Lambda^1$.
        We can then compare the observed convergence rate of the parameter
        obtained in the continuous framework and the parameter obtained in the discrete framework.
        The theorical convergence rate plotted on the figure is obtained with the discrete formula :
        this is why it changes for the continuous framework when we change h.

        As expected, performing the optimization in the discrete framework gives better results,
        since it is closer to reality.
    """
    T = DEFAULT.T
    finite_difference = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[0], finite_difference,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=70,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.), legend=False)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_difference,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=500,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.))
    show_or_save("fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff_no_corr")

def fig_error_by_taking_continuous_rate_constant_number_dt_h2_vol():
    """
        We see on this figure the utility of making the discrete analysis.
        For a given h, we compute the optimal free parameter $\\Lambda^1$
        of the robin interface condition.
        If we compute it with the continuous framework, we get always the same 
        $\\Lambda^1$.
        We can then compare the observed convergence rate of the parameter
        obtained in the continuous framework and the parameter obtained in the discrete framework.
        The theorical convergence rate plotted on the figure is obtained with the discrete formula :
        this is why it changes for the continuous framework when we change h.

        As expected, performing the optimization in the discrete framework gives better results,
        since it is closer to reality.
    """
    T = DEFAULT.T
    finite_volumes = DEFAULT.new(FiniteVolumes)
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[0], finite_volumes,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=70,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.), legend=False)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_volumes,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=500,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.))
    show_or_save("fig_error_by_taking_continuous_rate_constant_number_dt_h2_vol")


def fig_compare_continuous_discrete_rate_robin_robin_vol():
    """
        see @fig_error_by_taking_continuous_rate_constant_number_dt_h2_vol
        except it is in the Robin-Robin case instead of Robin-Neumann
    """

    T = 6.
    finite_volumes = DEFAULT.new(FiniteVolumes)
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)
    compare_continuous_discrete_rate_robin_robin(fig, axes[0], finite_volumes,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=120,
                                                          steps=40,
                                                          legend=False,
                                                          bounds_h=(-1.5,0.))
    compare_continuous_discrete_rate_robin_robin(fig, axes[1], finite_volumes,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=800,
                                                          steps=40,
                                                          bounds_h=(-1.5,0.),
                                                          plot_perfect_performances=True)
    show_or_save("fig_compare_continuous_discrete_rate_robin_robin_vol")

def fig_compare_continuous_discrete_rate_robin_robin_diff_naive():
    """
        see @fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff
        except it is in the Robin-Robin case instead of Robin-Neumann.
        The figure with naive discretization has not been done in Robin-Neumann, why ?
    """
    T = 6.
    finite_diff = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)
    compare_continuous_discrete_rate_robin_robin(fig, axes[0], finite_diff,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=70,
                                                          steps=50,
                                                          legend=False,
                                                          bounds_h=(-2.5,0.))
    compare_continuous_discrete_rate_robin_robin(fig, axes[1], finite_diff,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=500,
                                                          steps=50,
                                                          bounds_h=(-2.5,0.))
    show_or_save("fig_compare_continuous_discrete_rate_robin_robin_diff_naive")

def fig_compare_continuous_discrete_rate_robin_robin_diff_extra():
    """
        see @fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff_no_corr
        except it is in the Robin-Robin case instead of Robin-Neumann
    """
    T = 6.
    finite_diff_extra = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)
    compare_continuous_discrete_rate_robin_robin(fig, axes[0], finite_diff_extra,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=70,
                                                          steps=50,
                                                          legend=False,
                                                          bounds_h=(-2.5,0.))
    compare_continuous_discrete_rate_robin_robin(fig, axes[1], finite_diff_extra,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=500,
                                                          steps=50,
                                                          bounds_h=(-2.5,0.))
    show_or_save("fig_compare_continuous_discrete_rate_robin_robin_diff_extra")

def fig_compare_continuous_discrete_rate_robin_robin_diff():
    """
        see @fig_error_by_taking_continuous_rate_constant_number_dt_h2_diff
        except it is in the Robin-Robin case instead of Robin-Neumann
    """
    T = 6.
    finite_diff = DEFAULT.new(FiniteDifferences)
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    axes[1].yaxis.set_tick_params(labelbottom=True)
    compare_continuous_discrete_rate_robin_robin(fig, axes[0], finite_diff,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=70,
                                                          steps=50,
                                                          legend=False,
                                                          bounds_h=(-2.5,0.))
    compare_continuous_discrete_rate_robin_robin(fig, axes[1], finite_diff,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=500,
                                                          steps=50,
                                                          bounds_h=(-2.5,0.))
    show_or_save("fig_compare_continuous_discrete_rate_robin_robin_diff")

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

def fig_what_am_i_optimizing_criblage():
    """
        Simple plot of the function we minimize when looking for the
        optimal Robin parameter.
        The convergence rate is smaller in the non-degenerated discrete case
        than in the continuous framework:
        it means we can find a better parameter than the parameter yielded by the continuous analysis.
        Once again, the corrective term blocks the convergence of the high frequencies:
        The best parameter we can find still has a bad value.
    """
    T = 10. / 7
    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_difference_wout_corr = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    finite_volumes = DEFAULT.new(FiniteVolumes)

    optim_by_criblage_plot((finite_difference, finite_volumes,
                            finite_difference_wout_corr),
                           T=T, number_dt_h2=DEFAULT.COURANT_NUMBER, steps=200)
    show_or_save("fig_what_am_i_optimizing_criblage")


def fig_error_interface_time_domain_profiles():
    finite_difference = DEFAULT.new(FiniteDifferences)

    finite_volumes = DEFAULT.new(FiniteVolumes)

    raw_plot((finite_difference, finite_volumes), 100)
    plt.title(values_str(200, -200, DT_DEFAULT, 100*DT_DEFAULT,
        D1_DEFAULT, .54, 0, 0, NUMBER_DDT_H2))
    show_or_save("fig_error_interface_time_domain_profiles")


def fig_validation_code_frequency_error_diffboth():
    """
        Initial error after ITERATION iteration.
        It is a way of validation of the code : the theoric error
        is close to the error observed in simulation.
        for the first iteration (ITERATION==0), we see
        that we do'nt match at all. This is explained by
        the fact that the first guess is not a solution
        of the diffusion equation. Therefore, we need to change
        the theorical rate. It is explained in details in the PDF.
        
        to obtained the predictive errors, we multiply the first
        guess by the theorical rate.
    """
    NUMBER_DDT_H2 = .1
    D1 = .1
    DT = NUMBER_DDT_H2 * (DEFAULT.SIZE_DOMAIN_1 / DEFAULT.M1)**2 / D1

    finite_difference = DEFAULT.new(FiniteDifferences)

    finite_difference.D1_DEFAULT = D1
    finite_difference.DT_DEFAULT = DT
    fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8])

    analysis_frequency_error((finite_difference, ), 100, iteration=0, lambda_1=1e13, fig=fig, ax=axes[0], legend=False)
    axes[0].set_title("Première itération")
    analysis_frequency_error((finite_difference, ), 100, iteration=1, lambda_1=1e13, fig=fig, ax=axes[1])
    axes[1].set_title("Deuxième itération")
    fig.suptitle("Profils de l'erreur : Différences finies avec terme correctif")
    show_or_save("fig_validation_code_frequency_error_diffboth")


def fig_validation_code_frequency_error_diff1(ITERATION=0):
    """
        Initial error after ITERATION iteration.
        It is a way of validation of the code : the theoric error
        is close to the error observed in simulation.
        for the first iteration (ITERATION==0), we see
        that we do'nt match at all. This is explained by
        the fact that the first guess is not a solution
        of the diffusion equation. Therefore, we need to change
        the theorical rate. It is explained in details in the PDF.
        
        to obtained the predictive errors, we multiply the first
        guess by the theorical rate.
    """
    NUMBER_DDT_H2 = .1
    D1 = .1
    DT = NUMBER_DDT_H2 * (DEFAULT.SIZE_DOMAIN_1 / DEFAULT.M1)**2 / D1

    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_volumes = DEFAULT.new(FiniteVolumes)

    finite_difference.D1_DEFAULT = D1
    finite_volumes.D1_DEFAULT = D1
    finite_difference.DT_DEFAULT = DT
    finite_volumes.DT_DEFAULT = DT

    analysis_frequency_error((finite_difference, ), 100, iteration=ITERATION, lambda_1=1e13)
    iteration_str = "first iteration" if ITERATION==0 else "second iteration"
    suffixe_name = str(ITERATION+1)
    plt.title("Error profile: " + iteration_str + " (Finite differences)")
    show_or_save("fig_validation_code_frequency_error_diff" + suffixe_name)

fig_validation_code_frequency_error_diff2 = \
        functools.partial(fig_validation_code_frequency_error_diff1,
                          ITERATION=1)

def fig_validation_code_frequency_rate_dirichlet_neumann():
    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_volumes = DEFAULT.new(FiniteVolumes)

    analysis_frequency_rate((finite_difference, finite_volumes),
                            100, lambda_1=-1e13)
    plt.title(values_str(200, -200, DEFAULT.DT, 100*DEFAULT.DT,
        DEFAULT.D1, .54, 0, 0, DEFAULT.COURANT_NUMBER))
    show_or_save("fig_validation_code_frequency_rate_dirichlet_neumann")


def verification_analysis_naive_neumann():
    NUMBER_DDT_H2 = 1.
    M = 200
    SIZE_DOMAIN = 200
    D1 = .54
    D2 = .6
    DT_DEFAULT = NUMBER_DDT_H2 * (M / SIZE_DOMAIN)**2 / D1
    a = .0
    c = 0.0

    finite_difference_naive = \
        FiniteDifferencesNaiveNeumann(A_DEFAULT=a, C_DEFAULT=c,
                                          D1_DEFAULT=D1, D2_DEFAULT=D2,
                                          M1_DEFAULT=M, M2_DEFAULT=M,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN,
                                          SIZE_DOMAIN_2=SIZE_DOMAIN,
                                          LAMBDA_1_DEFAULT=0.,
                                          LAMBDA_2_DEFAULT=0.,
                                          DT_DEFAULT=DT_DEFAULT)

    analysis_frequency_rate((finite_difference_naive, ),
                            100, lambda_1=-1e53)


    h = SIZE_DOMAIN/(M-1)
    all_R1 = []
    all_R2 = []
    ret = []
    derivative = []
    all_w = np.linspace(-1.5, 1.5, 2000)
    for w in all_w:
        s=w*1j
        Y_0 = -D1/(h*h)
        Y_1 = 2*D1/(h*h)+c
        Y_2 = -D1/(h*h)
        r1 = 4*D1/h / h
        r2 = 4*D2/h / h
        lambda_1m1 = (-s+np.sqrt((Y_1+s)**2 - 4*Y_0*Y_2))/(2*Y_2)
        R1m = np.sqrt((Y_1+s)**2-4*Y_0*Y_2) - w * 1j
        R1p = np.sqrt((Y_1+s)**2-4*Y_0*Y_2) + w * 1j

        Y_0 = -D2/(h*h)
        Y_1 = 2*D2/(h*h)+c
        Y_2 = -D2/(h*h)
        lambda_2m1 = (-s+np.sqrt((Y_1+s)**2 - 4*Y_0*Y_2))/(2*Y_2)
        R2m = np.sqrt((Y_1+s)**2-4*Y_0*Y_2) - w * 1j
        R2p = np.sqrt((Y_1+s)**2-4*Y_0*Y_2) + w * 1j

        S1 = (r1 * 1j - 2*w) / (2*np.sqrt(-w*(w-r1*1j)))
        S2 = (r2 * 1j - 2*w) / (2*np.sqrt(-w*(w-r2*1j)))
        R1R2 = np.sqrt(R1m*R1p*R2m*R2p)
        derivative += [((S1-1j)*R1m
                - (S2-1j)*R1m*R1p*R2p / (R2m*R2p))/R1R2]
        derivative_num = (S1-1j)*R1m/np.sqrt(R1m*R1p)
        derivative_den = (S2-1j)*R2m/np.sqrt(R2m*R2p)

        rho = np.abs(D1/D2 * (lambda_1m1) / (lambda_2m1))
        rho_num = np.abs(np.sqrt(1 + r1/s) -1)
        rho_den = np.abs(np.sqrt(1 + r2/s) -1)

        derivative[-1] = derivative_num*rho_den - derivative_den*rho_num
        derivative[-1] /= rho_den*rho_den

        #all_R1 +=[R1]
        #all_R2 +=[R2]

        ret += [rho_num/rho_den]

    plt.plot(all_w, ret, "y--")
    #plt.plot(all_w, all_R1, "y--")
    #plt.plot(all_w, all_R2, "y", linestyle="dotted")
    plt.plot(all_w, derivative, "m--")
    plt.plot(all_w[:-1], np.diff(np.array(ret))/np.diff(all_w), "k")
    plt.title(values_str(200, -200, DT_DEFAULT, 1000*DT_DEFAULT,
        D1, .54, a, c, NUMBER_DDT_H2))
    show_or_save("verification_analysis_naive_neumann")

def fig_frequency_rate_dirichlet_neumann_comparison_c_nonzero():
    """
        see @fig_frequency_rate_dirichlet_neumann_comparison_c_zero,
        except we have a reaction term.
        We see that the reaction term changes the global maximum
        of \\rho and changes it differently for each discretization.
        Except for degenerated case (with corrective term), it diminuish it.
    """
    c = 0.4
    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_volumes = DEFAULT.new(FiniteVolumes)
    finite_difference_wout_corr = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    finite_difference_naive = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    for dis in (finite_difference, finite_volumes,
                finite_difference_wout_corr, finite_difference_naive):
        dis.C_DEFAULT = c
        dis.DT_DEFAULT *= 10

    analysis_frequency_rate((finite_difference, finite_volumes,
                             finite_difference_wout_corr, finite_difference_naive),
                            1000, lambda_1=-1e13)
    plt.title("Taux de convergence avec $c \\neq 0$ : interface \"Dirichlet Neumann\"")
    show_or_save("fig_frequency_rate_dirichlet_neumann_comparison_c_nonzero")


def fig_frequency_rate_dirichlet_neumann_comparison_c_zero():
    """
        Convergence rate for each frequency with
        Dirichlet-Neumann interface conditions.
        It is a way of validation of the code : the theoric rate
        is very close to the observed rate in simulations.

        We see that the finite difference scheme with a corrective term
        have a very bad (close to 1) convergence rate for high frequencies.
        The other discretizatiosn have a better (smaller) convergence rate
        than the continuous analysis.
        Note that the convergence rate is independant from the frequency
        in the continuous analysis.
    """
    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_volumes = DEFAULT.new(FiniteVolumes)
    finite_difference_wout_corr = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    finite_difference_naive = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    for dis in (finite_difference, finite_volumes, finite_difference_wout_corr, finite_difference_naive):
        dis.DT_DEFAULT *= 10

    analysis_frequency_rate((finite_difference, finite_volumes,
                             finite_difference_naive),
                            1000, lambda_1=-1e13)
    plt.title("Taux de convergence : interface \"Dirichlet Neumann\"")
    show_or_save("fig_frequency_rate_dirichlet_neumann_comparison_c_zero")


def fig_validation_code_frequency_rate_robin_neumann():
    T = 1000.
    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_volumes = DEFAULT.new(FiniteVolumes)

    analysis_frequency_rate((finite_difference, finite_volumes),
                            N=int(T/DEFAULT.DT), number_samples=1350,
                            fftshift=False)
    plt.title(values_str(200, -200, DEFAULT.DT, T,
        DEFAULT.D1, .54, 0, 0, DEFAULT.COURANT_NUMBER))
    show_or_save("fig_validation_code_frequency_rate_robin_neumann")

def fig_plot3D_function_to_minimize():
    """
        Same function as @fig_what_am_i_optimizing_criblage
        except it is now in the Robin-Robin case, with two parameters.
        in 3D it is hard to visualize multiple data on the same plot,
        but we can see that both continuous and discrete analysis
        share the same global shape.
    """
    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_difference2 = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    finite_difference3 = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    finite_vol = DEFAULT.new(FiniteVolumes)
    facteur = 1.4
    finite_difference2.M1_DEFAULT *= facteur
    finite_difference2.M2_DEFAULT *= facteur
    fig = plot_3D_profile((finite_difference2, ), DEFAULT.N)
    show_or_save("fig_plot3D_function_to_minimize")



def analysis_frequency_error(discretization, N, iteration=1, lambda_1=0.6139250052109033, fig=None, ax=None, legend=True):
    if fig is None:
        fig, ax = plt.subplots()
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
        # continuous analysis : 0.6 (dirichlet neumann case : just put 1e13 in lambda_1)

        dt = dis.DT_DEFAULT
        axis_freq = np.linspace(-pi / dt, pi / dt, N)

        frequencies = memoised(frequency_simulation,
                               dis,
                               N,
                               Lambda_1=lambda_1,
                               number_samples=135)
        linebe4, = ax.semilogy(axis_freq,
                 frequencies[iteration],
                 col + ':')
        lineafter, = ax.semilogy(axis_freq,
                 frequencies[iteration+1],
                 col2 + '-')

        real_freq_discrete = np.fft.fftshift(np.array([
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=False,
                                 N=N) for w in axis_freq
        ]))

        real_freq_continuous = np.array([
            continuous_analytic_rate_robin_neumann(dis, w=w, Lambda_1=lambda_1)
            for w in axis_freq
        ])

        linethebe4, = ax.semilogy(axis_freq,
                 real_freq_continuous * frequencies[iteration],
                 'b-.')
        linetheafter, = ax.semilogy(axis_freq,
                 real_freq_discrete * frequencies[iteration],
                 'k',
                 linestyle='dashed')

    if legend:
        linebe4.set_label("Observé avant l'itération")
        lineafter.set_label("Observé après l'itération")
        linethebe4.set_label("Théorique après l'itération (continu)")
        linetheafter.set_label("Théorique après l'itération (discret)")
        fig.legend(loc="lower center")
    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("Erreur $\\hat{e}$")

def optim_by_criblage_plot(discretization, T, number_dt_h2, steps=50):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    from scipy.optimize import minimize_scalar

    all_h = np.linspace(-2.2, 0, steps)
    all_h = np.exp(all_h[::-1]) / 2.1

    h_plot = all_h[0]
    lambdas = np.linspace(0, 10, 300)
    color = ['r-', 'g:', 'y--']
    for dis, col in zip(discretization, color):
        plt.plot(lambdas, [to_minimize_analytic_robin_neumann(l,
            h_plot, dis, number_dt_h2, T) for l in lambdas], col,
            label=dis.name() + ", semi-discret")
    plt.plot(lambdas, [to_minimize_continuous_analytic_rate_robin_neumann(l,
        h_plot, discretization[0], number_dt_h2, T) for l in lambdas], 'b-.',
        label="Continu")
    plt.xlabel("$\\Lambda$")
    plt.ylabel("$\\max_s{\\hat{\\rho}}$")
    plt.title("Fonction à minimiser : interface \"Robin-Neumann\"")
    plt.legend()



def analysis_frequency_rate(discretization, N,
                            lambda_1=0.6139250052109033,
                            number_samples=135, fftshift=True):
    fig, ax = plt.subplots()
    def continuous_analytic_error_neumann(discretization, w):
        D1 = discretization.D1_DEFAULT
        D2 = discretization.D2_DEFAULT
        # sig1 is \sigma^1_{+}
        sig1 = np.sqrt(np.abs(w) / (2 * D1)) * (1 + np.abs(w) / w * 1j)
        # sig2 is \sigma^2_{-}
        sig2 = -np.sqrt(np.abs(w) / (2 * D2)) * (1 + np.abs(w) / w * 1j)
        return D1 * sig1 / (D2 * sig2)

    colors = ['r', 'g', 'm']
    for dis, col, col2 in zip(discretization, colors, colors[::-1]):
        # first: find a correct lambda : we take the optimal yielded by
        # continuous analysis

        # continuous_best_lam_robin_neumann(dis, N)
        #print("rate", dis.name(), ":", rate(dis, N, Lambda_1=lambda_1))
        dis.DT_DEFAULT /= 10
        dt = dis.DT_DEFAULT
        axis_freq = np.linspace(-pi / dt, pi / dt, N)

        frequencies = memoised(frequency_simulation,
                               dis,
                               N,
                               Lambda_1=lambda_1,
                               number_samples=number_samples)
        # plt.plot(axis_freq, frequencies[0], col2+"--", label=" initial frequency ")
        # plt.plot(axis_freq, frequencies[1], col, label=dis.name()+" after 1 iteration")
        #plt.plot(axis_freq, frequencies[1], col+"--", label=dis.name()+" frequential error after the first iteration")
        lsimu, = ax.plot(axis_freq,
                 frequencies[2] / frequencies[1],
                 col)
        ax.annotate(dis.name(), xy=(axis_freq[0], frequencies[2][0] / frequencies[1][0]),
                    xycoords='data', horizontalalignment='left', verticalalignment='top')


        real_freq_discrete = np.array([
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=False,
                                 N=N) for w in axis_freq
        ])
        real_freq_discrete[np.isnan(real_freq_discrete)] = 1.
        if fftshift:
            real_freq_discrete = np.fft.fftshift(real_freq_discrete)

        real_freq_semidiscrete = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=True,
                                 N=N) for w in axis_freq
        ]

        real_freq_semidiscrete_modified_time1 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=True,
                                 modified_time=1,
                                 N=N) for w in axis_freq
        ]
        real_freq_semidiscrete_modified_time2 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=True,
                                 modified_time=2,
                                 N=N) for w in axis_freq
        ]
        real_freq_semidiscrete_modified_time3 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=True,
                                 modified_time=3,
                                 N=N) for w in axis_freq
        ]
        real_freq_semidiscrete_modified_time4 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 semi_discrete=True,
                                 modified_time=4,
                                 N=N) for w in axis_freq
        ]

        real_freq_continuous = [
            continuous_analytic_rate_robin_neumann(dis, w=w, Lambda_1=lambda_1)
            for w in axis_freq
        ]

        lsemi, = ax.plot(axis_freq,
                 real_freq_semidiscrete,
                 col,
                 linestyle='dotted')
        lmodified, = ax.plot(axis_freq,
                 real_freq_semidiscrete_modified_time1,
                 'k',
                 linestyle='dashed')
        lmodified, = ax.plot(axis_freq,
                 real_freq_semidiscrete_modified_time2,
                 'k',
                 linestyle='dashed')
        lmodified, = ax.plot(axis_freq,
                 real_freq_semidiscrete_modified_time3,
                 'k',
                 linestyle='dashed')
        lmodified, = ax.plot(axis_freq,
                 real_freq_semidiscrete_modified_time4,
                 'k',
                 linestyle='dashed')
        """
        lfull, = ax.plot(axis_freq,
                 real_freq_discrete,
                 'k',
                 linestyle='dashed')
        """

    lcont, = ax.plot(axis_freq,
             real_freq_continuous,
             'b-.')

    from matplotlib.lines import Line2D
    lsimu = Line2D([0], [0], color="k")
    lsemi = Line2D([0], [0], color="k", linestyle=":")

    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("Taux de convergence $\\hat{\\rho}$")
    plt.legend((lsimu, lsemi, lmodified, lcont),
               ('Simulation', 'Semi-discret (théorique)',
                'Semi-Discret (théorique, modifié avec termes 1,2,3,4 en temps)', 'Continu (théorique)'), loc='upper right')


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

def plot_3D_profile(all_dis, N):
    dt = DEFAULT.DT

    cont = functools.partial(continuous_analytic_rate_robin_robin, all_dis[0])
    subplot_param = (2 + len(list(all_dis)))*100 + 11

    def fun(x):
        return max([cont(Lambda_1=x[0], Lambda_2=x[1], w=pi / (n * dt))
                    for n in (1, N)])

    fig, ax = plot_3D_square(fun, 0, 2., -2., -0,  200, 100, subplot_param=subplot_param)
    ax.set_title("Taux de convergence : analyse continue")
    ax.set_ylabel("$\\Lambda^2$")


    cont_modified = functools.partial(cv_rate.continuous_analytic_rate_robin_robin_modified_naive_ordre3, all_dis[0])
    subplot_param += 1

    def fun_modified(x):
        return max([cont_modified(Lambda_1=x[0], Lambda_2=x[1], w=pi / (n * dt))
                    for n in (1, N)])

    fig, ax = plot_3D_square(fun_modified, -0, 2., -2., 0,  500, 100, fig=fig, subplot_param=subplot_param)
    ax.set_title("Taux de convergence : analyse continue modifiée")
    ax.set_ylabel("$\\Lambda^2$")

    """
    cont_modified3 = functools.partial(cv_rate.continuous_analytic_rate_robin_robin_modified_naive_ordre3, all_dis[0])
    subplot_param += 1

    def fun_modified3(x):
        return max([cont_modified3(Lambda_1=x[0], Lambda_2=x[1], w=pi / (n * dt))
                    for n in (1, N)])

    fig, ax = plot_3D_square(fun_modified3, -0, 2., -2., 0,  500, 100, fig=fig, subplot_param=subplot_param)
    ax.set_title("Taux de convergence : analyse continue modifiée (interface ordre 3)")
    ax.set_ylabel("$\\Lambda^2$")

    """

    for dis in all_dis:
        subplot_param += 1

        rate_fdiff = functools.partial(rate_fast, dis, N)
        def fun_me(x):
            return max([analytic_robin_robin(dis,
                                             Lambda_1=x[0], Lambda_2=x[1],
                                             w=pi / (n * dt), semi_discrete=True, modified_time=3)
                        for n in (1, N)])

        fig, ax = plot_3D_square(fun_me, -0, 2., -2., 0, 500, 100, fig=fig,
                                 subplot_param=subplot_param)
        ax.set_ylabel("$\\Lambda^2$")
        ax.set_title(dis.name() + ", modifiée en temps")
    ax.set_xlabel("$\\Lambda^1$")

    return fig


def reverse_colourmap(cmap, name = 'my_cmap_r'):
    import matplotlib as mpl
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

"""
    fun must take a tuple of two parameters.
"""


def plot_3D_square(fun, xmin, xmax, ymin, ymax, Nx, Ny, fig=None, subplot_param=111):
    from mpl_toolkits.mplot3d import Axes3D
    plot_colorbar = fig is None
    if fig is None:
        fig = plt.figure(figsize=[6.4 , 4.8*2])
    ax = fig.add_subplot(subplot_param)
    X = np.ones((Ny, 1)) @ np.reshape(np.linspace(xmin, xmax, Nx), (1, Nx))
    Y = (np.ones((Nx, 1)) @ np.reshape(np.linspace(ymin, ymax, Ny), (1, Ny))).T
    Z = np.array([[fun((x, y)) for x, y in zip(linex, liney)]
                  for linex, liney in zip(X, Y)])
    from matplotlib import cm
    cmap = reverse_colourmap(cm.YlGnBu)
    surf = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=.15, vmax=1)
    #min=0.2, max=0.5
    if plot_colorbar:
        fig.subplots_adjust(right=0.8, hspace=0.5)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_ax.set_title("$\\max_s\\hat{\\rho}$")
        fig.colorbar(surf, shrink=0.5, aspect=5, cax=cbar_ax)


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


def to_minimize_analytic_robin_neumann(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(discretization.analytic_robin_robin,
                          Lambda_1=l,
                          M1=M1,
                          M2=M2,
                          dt=dt)
    return max([f(pi / t * 1j) for t in np.linspace(dt, T, N)])

def to_minimize_continuous_analytic_rate_robin_robin(l,
        h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    cont = functools.partial(
        continuous_analytic_rate_robin_robin,
        discretization, l[0], l[1])
    return np.max([cont(pi / t) for t in np.linspace(dt, T, N)])


def to_minimize_analytic_robin_robin(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(discretization.analytic_robin_robin,
                          Lambda_1=l[0],
                          Lambda_2=l[1],
                          M1=M1,
                          M2=M2,
                          dt=dt)
    return max([f(pi / t * 1j) for t in np.linspace(dt, T, N)])

def to_minimize_analytic_robin_robin_modified_eq(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(cv_rate.analytic_robin_robin, discretization,
                          Lambda_1=l[0],
                          Lambda_2=l[1],
                          M1=M1,
                          M2=M2,
                          semi_discrete=True,
                          modified_time=3,
                          N=N,
                          dt=dt)
    return max([f(pi / t) for t in np.linspace(dt, T, N)])

def to_minimize_analytic_robin_robin_fulldiscrete(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(cv_rate.analytic_robin_robin, discretization,
                          Lambda_1=l[0],
                          Lambda_2=l[1],
                          M1=M1,
                          M2=M2,
                          semi_discrete=False,
                          dt=dt,
                          N=N)
    return max([f(w) for w in np.concatenate(((0,), pi/np.linspace(dt, T, N)))])

def to_minimize_robin_robin_perfect(l, h, discretization, number_dt_h2, T, number_samples):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1_DEFAULT)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    lambda_1 = l[0]
    lambda_2 = l[1]
    w = frequency_simulation(discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=lambda_1,
                         Lambda_2=lambda_2,
                         number_samples=number_samples,
                         dt=dt)
    return max(w[2] / w[1])


# The function can be passed in parameters of memoised:
to_minimize_analytic_robin_robin2 = FunMem(to_minimize_analytic_robin_robin)
to_minimize_analytic_robin_robin2_fulldiscrete = FunMem(to_minimize_analytic_robin_robin_fulldiscrete)
to_minimize_analytic_robin_robin2_modified = FunMem(to_minimize_analytic_robin_robin_modified_eq)
to_minimize_continuous_analytic_rate_robin_robin2 = \
        FunMem(to_minimize_continuous_analytic_rate_robin_robin)

to_minimize_robin_robin2_perfect = FunMem(to_minimize_robin_robin_perfect)

# The function can be passed in parameters of memoised:
to_minimize_analytic_robin_neumann2 = FunMem(to_minimize_analytic_robin_neumann)
to_minimize_continuous_analytic_rate_robin_neumann2 = \
        FunMem(to_minimize_continuous_analytic_rate_robin_neumann)


def compare_continuous_discrete_rate_robin_robin(fig, ax,
        discretization, T, number_dt_h2, steps=50, bounds_h=(0,2), legend=True,
        number_samples=500, plot_perfect_performances=False):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    from scipy.optimize import minimize

    dt = discretization.DT_DEFAULT
    N = int(T / dt)
    if N <= 1:
        print("ERROR BEGINNING: N is too small (<2)")

    all_h = np.linspace(bounds_h[0], bounds_h[1], steps)
    all_h = np.exp(all_h[::-1])

    def func_to_map(x): return memoised(minimize,
        fun=to_minimize_analytic_robin_robin2,
        x0=(0.6, 0.),
        args=(x, discretization, number_dt_h2, T))

    def func_to_map_fulldiscrete(x): return memoised(minimize,
        fun=to_minimize_analytic_robin_robin2_fulldiscrete,
        x0=(0.6, 0.),
        args=(x, discretization, number_dt_h2, T))

    def func_to_map_discrete_modif(x): return memoised(minimize,
        fun=to_minimize_analytic_robin_robin2_modified,
        x0=(0.6, 0.),
        args=(x, discretization, number_dt_h2, T))

    def func_to_map_perfect_perf(x): 
        ret = memoised(minimize, method="Nelder-Mead",
                fun=to_minimize_robin_robin2_perfect,
                x0=(0.5, -0.5),
                args=(x, discretization, number_dt_h2, T, number_samples*4))
        print(ret)
        return ret


    print("Computing lambdas in discrete framework.")
    ret_discrete = list(map(func_to_map, all_h))
    ret_fulldiscrete = list(map(func_to_map_fulldiscrete, all_h))
    ret_discrete_modif = list(map(func_to_map_discrete_modif, all_h))

    if plot_perfect_performances:
        perfect_performances = list(map(func_to_map_perfect_perf, all_h))
        theorical_rate_perfect = [ret.fun for ret in perfect_performances]

    optimal_discrete = [ret.x for ret in ret_discrete]
    optimal_fulldiscrete = [ret.x for ret in ret_fulldiscrete]
    optimal_discrete_modif = [ret.x for ret in ret_discrete_modif]
    theorical_rate_discrete = [ret.fun for ret in ret_discrete]
    theorical_rate_discrete_modif = [ret.fun for ret in ret_discrete_modif]

    def func_to_map_cont(x): return memoised(minimize,
        fun=to_minimize_continuous_analytic_rate_robin_robin2,
        x0=(0.6,0),
        args=(x, discretization, number_dt_h2, T))
    print("Computing lambdas in continuous framework.")
    ret_continuous = list(map(func_to_map_cont, all_h))
    # ret_discrete = [minimize_scalar(fun=to_minimize_discrete, args=(h)) \
    #    for h in all_h]
    optimal_continuous = [ret.x for ret in ret_continuous]
    theorical_cont_rate = [ret.fun for ret in ret_continuous]

    rate_with_continuous_lambda = []
    rate_with_discrete_lambda = []
    rate_with_fulldiscrete_lambda = []
    rate_with_discrete_modif_lambda = []
    print("optimal-continuous[0]:", optimal_continuous[0])
    print("optimal-discrete[0]:", optimal_discrete[0])

    try:
        for i in range(all_h.shape[0]):
            dt, N = get_dt_N(all_h[i], number_dt_h2, T,
                             discretization.D1_DEFAULT)
            print("h number:", i)
            M1 = int(discretization.SIZE_DOMAIN_1 / all_h[i])
            M2 = int(discretization.SIZE_DOMAIN_2 / all_h[i])
            rate_with_continuous_lambda += [
                    memoised(frequency_simulation, discretization,
                             N,
                             M1=M1,
                             M2=M2,
                             Lambda_1=optimal_continuous[i][0],
                             Lambda_2=optimal_continuous[i][1],
                             number_samples=number_samples,
                             dt=dt)
                ]
            rate_with_discrete_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_discrete[i][0],
                         Lambda_2=optimal_discrete[i][1],
                         number_samples=number_samples,
                         dt=dt)
            ]

            rate_with_fulldiscrete_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_fulldiscrete[i][0],
                         Lambda_2=optimal_fulldiscrete[i][1],
                         number_samples=number_samples,
                         dt=dt)
            ]
            rate_with_discrete_modif_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_discrete_modif[i][0],
                         Lambda_2=optimal_discrete_modif[i][1],
                         number_samples=number_samples,
                         dt=dt)
            ]
    except:
        pass

    rate_with_continuous_lambda = [max(w[2] / w[1])
            for w in rate_with_continuous_lambda]
    rate_with_discrete_lambda = [max(w[2] / w[1])
            for w in rate_with_discrete_lambda]
    rate_with_fulldiscrete_lambda = [max(w[2] / w[1])
            for w in rate_with_fulldiscrete_lambda]
    rate_with_discrete_modif_lambda = [max(w[2] / w[1])
            for w in rate_with_discrete_modif_lambda]

    linefdo, = ax.semilogx(all_h[:len(rate_with_fulldiscrete_lambda)],
                 rate_with_fulldiscrete_lambda,
                 "y")

    linemdo, = ax.semilogx(all_h[:len(rate_with_discrete_modif_lambda)],
                 rate_with_discrete_modif_lambda,
                 "b")
    linemdt, = ax.semilogx(all_h,
                 theorical_rate_discrete_modif,
                 "b--")

    linedo, = ax.semilogx(all_h[:len(rate_with_discrete_lambda)],
                 rate_with_discrete_lambda,
                 "g")
    linedt, = ax.semilogx(all_h,
                 theorical_rate_discrete,
                 "g--")
    lineco, = ax.semilogx(all_h[:len(rate_with_continuous_lambda)],
                 rate_with_continuous_lambda,
                 "r")
    linect, = ax.semilogx(all_h,
                 theorical_cont_rate,
                 "r--")

    if plot_perfect_performances:
        linept, = ax.semilogx(all_h,
                     theorical_rate_perfect,
                     "k--")
        if legend:
            linept.set_label("Taux théorique avec optimisation directement sur le taux observé")

    if legend:
        linefdo.set_label("Taux observé avec $\\Lambda$ optimal discret en temps et en espace")
        linemdo.set_label("Taux observé avec $\\Lambda$ optimal semi-discret, modifié en temps")
        linedo.set_label("Taux observé avec $\\Lambda$ optimal semi-discret")
        linedt.set_label("Taux théorique avec $\\Lambda$ optimal semi-discret")
        lineco.set_label("Taux observé avec $\\Lambda$ optimal continu")
        linect.set_label("Taux théorique avec $\\Lambda$ optimal continu")
        fig.legend(loc="lower left")

    ax.set_xlabel("h")
    ax.set_ylabel("$\\hat{\\rho}$")
    fig.suptitle('Comparaison des analyses semi-discrètes et continues' +
            ' (Robin-Robin), ' +
              discretization.name())
    ax.set_title('Nombre de Courant : $D_1\\frac{dt}{h^2}$ = ' + str(number_dt_h2))


def error_by_taking_continuous_rate_constant_number_dt_h2(fig, ax,
        discretization, T, number_dt_h2, steps=50, bounds_h=(0,2), legend=True, number_samples=500):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    from scipy.optimize import minimize_scalar

    dt = discretization.DT_DEFAULT
    N = int(T / dt)
    if N <= 1:
        print("ERROR BEGINNING: N is too small (<2)")

    all_h = np.linspace(bounds_h[0], bounds_h[1], steps)
    all_h = np.exp(all_h[::-1])

    def func_to_map(x): return memoised(minimize_scalar,
        fun=to_minimize_analytic_robin_neumann2,
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
    print("Computing lambdas in continuous framework.")
    ret_continuous = list(map(func_to_map_cont, all_h))
    # ret_discrete = [minimize_scalar(fun=to_minimize_discrete, args=(h)) \
    #    for h in all_h]
    optimal_continuous = [ret.x for ret in ret_continuous]
    theorical_cont_rate = [ret.fun for ret in ret_continuous]

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
                    memoised(frequency_simulation, discretization,
                             N,
                             M1=M1,
                             M2=M2,
                             Lambda_1=optimal_continuous[i],
                             number_samples=number_samples,
                             dt=dt)
                ]
            rate_with_discrete_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         number_samples=number_samples,
                         Lambda_1=optimal_discrete[i],
                         dt=dt)
            ]
    except:
        pass

    rate_with_continuous_lambda = [max(w[2] / w[1])
            for w in rate_with_continuous_lambda]
    rate_with_discrete_lambda = [max(w[2] / w[1])
            for w in rate_with_discrete_lambda]

    linedo, = ax.semilogx(all_h[:len(rate_with_discrete_lambda)],
                 rate_with_discrete_lambda,
                 "g", linewidth=2.5)
    linedt, = ax.semilogx(all_h,
                 theorical_rate_discrete,
                 "g--", linewidth=2.5)
    lineco, = ax.semilogx(all_h[:len(rate_with_continuous_lambda)],
                 rate_with_continuous_lambda,
                 "r")
    linect, = ax.semilogx(all_h,
                 theorical_cont_rate,
                 "r--")
    if legend:
        linedo.set_label("Taux observé avec $\\Lambda$ optimal semi-discret")
        linedt.set_label("Taux théorique avec $\\Lambda$ optimal semi-discret")
        lineco.set_label("Taux observé avec $\\Lambda$ optimal continu")
        linect.set_label("Taux théorique avec $\\Lambda$ optimal continu")
        fig.legend(loc="center left")

    ax.set_xlabel("h")
    ax.set_ylabel("$\\rho$")
    fig.suptitle('Comparaison des analyses semi-discrètes et continues' +
              ' (Robin-Neumann), ' + discretization.name()
              #+', $D_1$='
              #+str(discretization.D1_DEFAULT)
              #+', $D_2$='
              #+str(discretization.D2_DEFAULT)
              #+', a=c=0'
              )
    ax.set_title("Nombre de Courant : $D_1\\frac{dt}{h^2}=$" + str(round(number_dt_h2, 3)))


def fig_optimal_lambda_function_of_h():
    """
        Simple figure to show that when we work with a discrete framework,
        we don't get the same optimal parameter than the parameters we get
        with the analysis in the continuous framework.
        The difference is greater with the finite difference scheme because
        the corrective term used damp the convergence rate in high frequencies.
    """
    finite_difference = DEFAULT.new(FiniteDifferences)
    finite_volumes = DEFAULT.new(FiniteVolumes)
    N = DEFAULT.N
    # it was N=100

    optimal_function_of_h((finite_difference, finite_volumes), N)
    show_or_save("fig_optimal_lambda_function_of_h")


def optimal_function_of_h(discretizations, N):
    from scipy.optimize import minimize_scalar
    dt = discretizations[0].DT_DEFAULT
    T = dt * N
    all_h = np.linspace(0.01, 1, 300)
    colors=['r', 'g']

    for discretization, col in zip(discretizations, colors):
        def to_minimize_continuous(l):
            cont = functools.partial(continuous_analytic_rate_robin_neumann,
                                     discretization, l)
            return np.max([
                cont(pi / t) for t in np.linspace(dt, T, N)
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
                f(pi / t * 1j) for t in np.linspace(dt, T, N)
            ])

        ret_discrete = [minimize_scalar(fun=to_minimize_discrete, args=(h)).x
                        for h in all_h]
        plt.plot(all_h, ret_discrete, col, label=discretization.name() +
                ', $\\Lambda$ optimal')

    plt.hlines(optimal_continuous,
               all_h[0],
               all_h[-1],
               "k",
               'dashed',
               label='$\\Lambda$ optimal : analyse continue')
    plt.legend()
    plt.xlabel("h")
    plt.ylabel("$\\Lambda$ optimal")


def fig_contour_advection():
    N = 1000
    fig, ax = plt.subplots(1,1,figsize=[6.4 , 4.8])
    xmin, xmax = 0, 0.75
    ymin, ymax = 0, 2

    def function_to_plot(aw_h, Dw_h2):
        return np.clip(np.abs(1+((-1/2) + np.sqrt(Dw_h2*1j + 1/12 + aw_h**2))/ ((1/6) - Dw_h2*1j + aw_h*1j)), 0, 1.2)
    from mpl_toolkits.mplot3d import Axes3D
    plot_colorbar = fig is None
    X = np.ones((N, 1)) @ np.reshape(np.linspace(xmin, xmax, N), (1, N))
    Y = (np.ones((N, 1)) @ np.reshape(np.linspace(ymin, ymax, N), (1, N))).T
    Z = np.array([[function_to_plot(x, y) for x, y in zip(linex, liney)]
                  for linex, liney in zip(X, Y)])
    from matplotlib import cm

    cmap = reverse_colourmap(cm.YlGnBu)
    surf = ax.pcolormesh(X, Y, Z, cmap=cmap)
    ax.contour(X, Y, Z, levels=[1])
    #min=0.2, max=0.5
    fig.subplots_adjust(right=0.8, hspace=0.5)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax.set_title("$|\\lambda^{FV}_{-}|$")
    ax.set_xlabel("$\\frac{a\\omega}{h}$")
    ax.set_ylabel("$\\frac{D\\omega}{h^2}$")
    fig.colorbar(surf, shrink=0.5, aspect=5, cax=cbar_ax)
    show_or_save("fig_contour_advection")


def beauty_graph_finite(discretization,
                        lambda_min,
                        lambda_max,
                        steps,
                        courant_number,
                        fig, ax, legend,
                        **kwargs):
    PARALLEL = True
    LAMBDA_2_DEFAULT = discretization.LAMBDA_2_DEFAULT

    D1_DEFAULT = discretization.D1_DEFAULT
    D2_DEFAULT = discretization.D2_DEFAULT

    M1_DEFAULT = discretization.M1_DEFAULT
    M2_DEFAULT = discretization.M2_DEFAULT

    SIZE_DOMAIN_1 = discretization.SIZE_DOMAIN_1
    SIZE_DOMAIN_2 = discretization.SIZE_DOMAIN_2

    NUMBER_DDT_H2 = courant_number
    T = 100.

    DT_DEFAULT = NUMBER_DDT_H2 * (SIZE_DOMAIN_1 / M1_DEFAULT)**2 / D1_DEFAULT
    # should not be too different from the value with M2, Size_domain2, and D2
    TIME_WINDOW_LEN_DEFAULT = int(T / DT_DEFAULT)
    rate_func = functools.partial(cv_rate.rate_freq_slow, discretization,
                                  TIME_WINDOW_LEN_DEFAULT,
                                  seeds = range(100),
                                  **kwargs)
    rate_func_normL2 = functools.partial(cv_rate.rate_freq_slow,
                                         discretization,
                                         TIME_WINDOW_LEN_DEFAULT,
                                         function_to_use=np.linalg.norm,
                                         seeds = range(100),
                                         **kwargs)
    rate_func.__name__ = "bgf_rate_func_freq" + discretization.repr() + str(TIME_WINDOW_LEN_DEFAULT)
    rate_func_normL2.__name__ = "bgf_rate_func_normL2_freq" + discretization.repr() + str(TIME_WINDOW_LEN_DEFAULT)

    from scipy.optimize import minimize_scalar, minimize

    lambda_1 = np.linspace(lambda_min, lambda_max, steps)
    dt = DT_DEFAULT
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

    continuous_best_lam = cv_rate.continuous_best_lam_robin_neumann(
        discretization, TIME_WINDOW_LEN_DEFAULT)

    min_rho, max_rho = np.min(np.array(rho), axis=0), np.max(np.array(rho),
                                                             axis=0)
    best_analytic = lambda_1[np.argmin(max_rho)]

    filled = ax.fill_between(lambda_1,
                     min_rho,
                     max_rho,
                     facecolor="green", alpha=0.3)

    vline = ax.vlines(continuous_best_lam,
               0,
               1,
               "k",
               'dashed')
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

    print("> Starting simulations (this might take a while)")

    x = np.linspace(lambda_min, lambda_max, steps)
    rate_f = []
    rate_f_L2 = []
    for xi in x:
        rate_f += [memoised(rate_func, xi)]
        rate_f_L2 += [memoised(rate_func_normL2, xi)]

    print("> Starting minimization in infinite norm.")

    best_linf_norm = x[np.argmin(np.array(list(rate_f)))]
    print("> Starting minimization in L2 norm.")
    best_L2_norm = x[np.argmin(np.array(list(rate_f_L2)))]
    l2line, = ax.plot(x,
             list(rate_f_L2),
             "b")
    linfline, = ax.plot(x,
             list(rate_f),
             "r", linewidth=2.5)

    ax.set_xlabel("$\\Lambda^1$")
    ax.set_ylabel("$\\hat{\\rho}$")

    ax.set_title("Nombre de Courant : $D\\frac{dt}{h^2} = " + str(courant_number)+"$")

    if legend:
        filled.set_label("$\\hat{\\rho}$ : pi/T < |w| < pi/dt")
        l2line.set_label(discretization.name() + ", norme $L^2$")
        linfline.set_label(discretization.name() + ", norme $L^\\infty$")
        vline.set_label('$\\Lambda$ optimal (analyse continue)')

    fig.legend(loc="lower center")

def set_save_to_png():
    global SAVE_TO_PNG
    SAVE_TO_PNG = True

def set_save_to_pgf():
    global SAVE_TO_PGF
    SAVE_TO_PGF = True

SAVE_TO_PNG = False
SAVE_TO_PGF = False
def show_or_save(name_func):
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
