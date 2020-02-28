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
from discretizations.finite_volumes_spline2 import FiniteVolumesSpline2
from discretizations.rk4_finite_volumes import Rk4FiniteVolumes
from discretizations.rk2_finite_volumes import Rk2FiniteVolumes
from discretizations.rk2_finite_volumes_spline2 import Rk2FiniteVolumesSpline2
from discretizations.rk4_finite_differences import Rk4FiniteDifferences
from discretizations.rk2_finite_differences import Rk2FiniteDifferences
from discretizations.rk2_finite_difference_extra import Rk2FiniteDifferencesExtra
from discretizations.rk4_finite_difference_extra import Rk4FiniteDifferencesExtra
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
        return Discretisation(A=self.A, C=self.C,
                              D1=self.D1, D2=self.D2,
                              M1=self.M1, M2=self.M2,
                              SIZE_DOMAIN_1=self.SIZE_DOMAIN_1,
                              SIZE_DOMAIN_2=self.SIZE_DOMAIN_2,
                              LAMBDA_1=self.LAMBDA_1,
                              LAMBDA_2=self.LAMBDA_2,
                              DT=self.DT)

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
                                                          number_samples=20,
                                                          bounds_h=(-2.5,1.), legend=False)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_difference,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=100,
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
                                                          number_samples=20,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.), legend=False)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_difference,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=100,
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
                                                          number_samples=20,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.), legend=False)
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_difference,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=100,
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
    finite_volumes.COURANT_NUMBER = .1
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[0], finite_volumes,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=20,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.), legend=False)
    finite_volumes.COURANT_NUMBER = 1.
    error_by_taking_continuous_rate_constant_number_dt_h2(fig, axes[1], finite_volumes,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=100,
                                                          steps=50,
                                                          bounds_h=(-2.5,1.))
    show_or_save("fig_error_by_taking_continuous_rate_constant_number_dt_h2_vol")

def fig_compare_continuous_discrete_rate_robin_robin_volspl2rk2():
    """
        see @fig_error_by_taking_continuous_rate_constant_number_dt_h2_vol
        except it is in the Robin-Robin case instead of Robin-Neumann
    """

    T = 600.
    finite_volumes = DEFAULT.new(Rk2FiniteVolumesSpline2)
    #fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    fig, axes = plt.subplots(1, 1, figsize=[6.4, 4.8], sharey=True)
    axes.yaxis.set_tick_params(labelbottom=True)
    #fig, axes = None, None
    finite_volumes.COURANT_NUMBER = .1
    finite_volumes.C = 0.1
    compare_continuous_discrete_rate_robin_robin(fig, axes, finite_volumes,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=4,
                                                          steps=26,
                                                          bounds_h=(-4.,1.),
                                                          plot_perfect_performances=False)
    show_or_save("fig_compare_continuous_discrete_rate_robin_robin_volspl2rk2")


def fig_compare_continuous_discrete_rate_robin_robin_vol():
    """
        see @fig_error_by_taking_continuous_rate_constant_number_dt_h2_vol
        except it is in the Robin-Robin case instead of Robin-Neumann
    """

    T = 600.
    finite_volumes = DEFAULT.new(FiniteVolumes)
    #fig, axes = plt.subplots(1, 2, figsize=[6.4 * 1.7, 4.8], sharey=True)
    fig, axes = plt.subplots(1, 1, figsize=[6.4, 4.8], sharey=True)
    axes.yaxis.set_tick_params(labelbottom=True)
    #fig, axes = None, None
    finite_volumes.COURANT_NUMBER = .1
    """
    compare_continuous_discrete_rate_robin_robin(fig, axes[0], finite_volumes,
                                                          T=T, number_dt_h2=.1,
                                                          number_samples=20,
                                                          steps=40,
                                                          legend=False,
                                                          bounds_h=(-1.5,0.))
    """
    finite_volumes.COURANT_NUMBER = 10
    finite_volumes.C = .05
    compare_continuous_discrete_rate_robin_robin(fig, axes, finite_volumes,
                                                          T=T, number_dt_h2=10,
                                                          number_samples=2000,
                                                          steps=41,
                                                          bounds_h=(-2.,1.),
                                                          plot_perfect_performances=False)
    show_or_save("fig_compare_continuous_discrete_rate_robin_robin_vol")

def figModifEqRobinOneSidedVol():
    T = 80.
    finite_volumes = DEFAULT.new(FiniteVolumes)
    finite_volumes.D1 = finite_volumes.D2
    finite_volumes.D1 = finite_volumes.D2
    fig, axes = plt.subplots(1, 2)
    validation_theorical_modif_resolution_robin_onesided(fig, axes[0], finite_volumes,
                                                          T=T, number_dt_h2=1,
                                                          number_samples=100,
                                                          steps=40,
                                                          bounds_h=(-1.5,0.),
                                                          plot_perfect_performances=False)
    validation_theorical_modif_resolution_robin_onesided(fig, axes[1], finite_volumes,
                                                          T=T, number_dt_h2=10,
                                                          number_samples=100,
                                                          steps=40,
                                                          bounds_h=(-1.5,0.),
                                                          plot_perfect_performances=False, legend=False)
    show_or_save("figModifEqRobinOneSidedVol")



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
                                                          number_samples=20,
                                                          steps=50,
                                                          legend=False,
                                                          bounds_h=(-2.5,0.))
    compare_continuous_discrete_rate_robin_robin(fig, axes[1], finite_diff,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=100,
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
                                                          number_samples=20,
                                                          steps=50,
                                                          legend=False,
                                                          bounds_h=(-2.5,0.))
    compare_continuous_discrete_rate_robin_robin(fig, axes[1], finite_diff_extra,
                                                          T=T, number_dt_h2=1.,
                                                          number_samples=100,
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
    plt.title(values_str(200, -200, DT, 100*DT,
        D1, .54, 0, 0, NUMBER_DDT_H2))
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

    finite_difference.D1 = D1
    finite_difference.DT = DT
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

    finite_difference.D1 = D1
    finite_volumes.D1 = D1
    finite_difference.DT = DT
    finite_volumes.DT = DT

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
    DT = NUMBER_DDT_H2 * (M / SIZE_DOMAIN)**2 / D1
    a = .0
    c = 0.0

    finite_difference_naive = \
        FiniteDifferencesNaiveNeumann(A=a, C=c,
                                          D1=D1, D2=D2,
                                          M1=M, M2=M,
                                          SIZE_DOMAIN_1=SIZE_DOMAIN,
                                          SIZE_DOMAIN_2=SIZE_DOMAIN,
                                          LAMBDA_1=0.,
                                          LAMBDA_2=0.,
                                          DT=DT)

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
    plt.title(values_str(200, -200, DT, 1000*DT,
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
        dis.C = c
        dis.DT *= 10

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
        dis.SIZE_DOMAIN_1 = 100
        dis.SIZE_DOMAIN_2 = 100
        dis.M1 = 400
        dis.M2 = 400
        courant_number = .1
        dis.A = 0
        #dis.C = 10
        dis.DT = courant_number * (dis.SIZE_DOMAIN_1 / (dis.M1-1))**2 / DEFAULT.D1

    analysis_frequency_rate((finite_difference_naive,finite_volumes),
                            int(1e4), lambda_1=.5, number_samples=3)
    plt.title("Taux de convergence : interface \"Robin Neumann\"")
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
    facteur = 1.6
    finite_difference2.M1 *= facteur
    finite_difference2.M2 *= facteur
    fig = plot_3D_profile((finite_difference2, ), DEFAULT.N)
    show_or_save("fig_plot3D_function_to_minimize")

def fig_compare_modif_approaches_corr():
    dis = DEFAULT.new(FiniteDifferences)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Differences : modified convergence factor (corrected interface)")
    show_or_save("fig_compare_modif_approaches_corr")

def fig_compare_modif_approaches_naive():
    dis = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Differences : modified convergence factor (naive interface)")
    show_or_save("fig_compare_modif_approaches_naive")
    
def fig_compare_fullmodif_approaches_naive():
    dis = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    fig, ax = compare_modif_approaches(dis, full_discrete=True)
    ax.set_title("Finite Differences : modified convergence factor (naive interface)")
    show_or_save("fig_compare_modif_approaches_naive")
    
def fig_compare_modif_approaches_extra():
    dis = DEFAULT.new(FiniteDifferencesNoCorrectiveTerm)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Differences : modified convergence factor (extrapolated interface)")
    show_or_save("fig_compare_modif_approaches_extra")
    
def fig_compare_modif_approaches_naive_rk2():
    dis = DEFAULT.new(Rk2FiniteDifferences)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Differences : modified convergence factor (naive interface, rk2)")
    show_or_save("fig_compare_modif_approaches_naive_rk2")

def fig_compare_modif_approaches_naive_rk4():
    dis = DEFAULT.new(Rk4FiniteDifferences)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Differences : modified convergence factor (naive interface, rk4)")
    show_or_save("fig_compare_modif_approaches_naive_rk4")
    
def fig_compare_modif_approaches_extra_rk2():
    dis = DEFAULT.new(Rk2FiniteDifferencesExtra)
    setup_modified = [(0,0,0, "continuous"), (0,4,0,"op"),(4,4,0,"time"), (4,4,4,"space")]
    fig, ax = compare_modif_approaches(dis, setup_modified=setup_modified)
    ax.set_title("Finite Differences : modified convergence factor (extrapolated interface, rk2)")
    show_or_save("fig_compare_modif_approaches_extra_rk2")

def fig_compare_modif_approaches_extra_rk4():
    dis = DEFAULT.new(Rk4FiniteDifferencesExtra)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Differences : modified convergence factor (extrapolated interface, rk4)")
    show_or_save("fig_compare_modif_approaches_extra_rk4")

def fig_compare_all_modif_approaches_vol():
    dis = DEFAULT.new(FiniteVolumes)
    setup_modified = [(0,0,0, "continuous"), (0,4,0,"operators"), (4,4,0,"operators+time"), (4, 4, 4, "operators+time+space")]
    fig, ax = compare_modif_approaches(dis, full_discrete=True, setup_modified=setup_modified)
    ax.set_title("Finite Volumes : modified convergence factor")
    show_or_save("fig_compare_modif_approaches_vol")

def fig_compare_modif_approaches_vol():
    dis = DEFAULT.new(FiniteVolumes)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Volumes : modified convergence factor")
    show_or_save("fig_compare_modif_approaches_vol")

def fig_compare_modif_approaches_vol_spline2():
    dis = DEFAULT.new(FiniteVolumesSpline2)
    #setup_modified = [(0,0,0, "continuous"), (4,0,0,"time"), (4, 0, 4, "time+space"),(4, 1.5, 4, "time+space+op")]
    dis.C = 0.01
    setup_modified = [(0,0,0, "continuous"), (4,0,4,"space")]
    fig, ax = compare_modif_approaches(dis, setup_modified=setup_modified)
    ax.set_title("Finite Volumes : modified convergence factor")
    show_or_save("fig_compare_modif_approaches_vol")

def fig_compare_modif_approaches_volspl2rk2():
    dis = DEFAULT.new(Rk2FiniteVolumesSpline2)
    dis.C = 0.01
    setup_modified = [(0,0,0, "continuous"), (0,0,4,"space")]
    fig, ax = compare_modif_approaches(dis, setup_modified=setup_modified)
    ax.set_title("RK2 : modified convergence factor")
    show_or_save("fig_compare_modif_approaches_volspl2rk2")

def fig_compare_modif_approaches_volspl2rk2_semidiscrete():
    dis = DEFAULT.new(Rk2FiniteVolumesSpline2)
    dis.C = 0.01
    setup_modified = [(0,0,0, "continuous")]
    fig, ax = compare_modif_approaches(dis, setup_modified=setup_modified, semi_discrete=True)
    ax.set_title("RK2 : modified convergence factor")
    show_or_save("fig_compare_modif_approaches_volspl2rk2_semidiscrete")

def fig_compare_modif_approaches_vol_rk2():
    dis = DEFAULT.new(Rk2FiniteVolumes)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Volumes : modified convergence factor (rk2)")
    show_or_save("fig_compare_modif_approaches_vol_rk2")

def fig_compare_modif_approaches_vol_rk4():
    dis = DEFAULT.new(Rk4FiniteVolumes)
    fig, ax = compare_modif_approaches(dis)
    ax.set_title("Finite Volumes : modified convergence factor (rk4)")
    show_or_save("fig_compare_modif_approaches_vol_rk4")


def fig_plzplotwhatIwant():
    """
        Compare the approaches used with modified equations :
        plot cv rate :
        - simulated
        - with continuous approach
        - with semi-discrete in space, modif in time
        - with modified equations, modified operators
        - with interface operator
    """
    dis = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    # 0.5; -0.5 is generally a good choice with our parameters
    lambda_1 = .5
    lambda_2 = -.5
    N = DEFAULT.N * 10

    # we take a little more points
    facteur = 1
    dis.SIZE_DOMAIN_1 *= facteur
    dis.SIZE_DOMAIN_2 *= facteur
    dis.M1 = int(dis.M1*facteur)
    dis.M2 = int(dis.M2*facteur)

    dt = dis.DT

    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[N//2] = .5

    # w = 2 pi k T / (N)
    axis_freq = 2 * pi*all_k / N / dt
    print(axis_freq)

    fig, ax = plt.subplots()

    simulated_freq = memoised(frequency_simulation,
                           dis,
                           N,
                           Lambda_1=lambda_1,
                           Lambda_2=lambda_2,
                           number_samples=50)
    simulated_cv = simulated_freq[2] / simulated_freq[1]
    nomodif_approach = [continuous_analytic_rate_robin_robin(dis, w=w,
                                                               Lambda_1=lambda_1,
                                                               Lambda_2=lambda_2)
                                        for w in axis_freq]
    semi_discrete_modif_time = [analytic_robin_robin(dis, Lambda_1=lambda_1, Lambda_2=lambda_2,
                                             w=w, semi_discrete=False, modified_time=3, N=N)
                                        for w in axis_freq]


    ax.plot(axis_freq*dt, simulated_cv, label="simulation")
    ax.plot(axis_freq*dt, nomodif_approach, label="continuous, not modified")
    #ax.plot(axis_freq*dt, continuous_modified, label="continuous modified")


    ax.plot(axis_freq*dt, semi_discrete_modif_time, "k--", label="semi-discrete in space, modified in time")
    ax.set_xlabel("$\\omega*\\delta t$")
    ax.set_ylabel("$\\hat{\\rho}$")
    ax.set_xlim(left=-5*np.pi/dt/N, right=axis_freq[-1]*dt)
    ax.set_ylim(bottom=0, top=1)
    ax.grid()
    fig.legend(loc="center right")
    plt.show()

def compare_modif_approaches(dis, full_discrete=False, setup_modified=[(0,0,0, "continuous"), (4, 4, 4, "modified")], semi_discrete=False):
    """
        Compare the approaches used with modified equations :
        plot cv rate :
        - simulated
        - with discrete in space time if full_discrete == True
        - with continuous approach if continuous==True
        - with all the modified setups (modified in time,
                                        modified interface operators,
                                        modified space equations,
                                        label)
    """
    # 0.5; -0.5 is generally a good choice with our parameters
    lambda_1 = .2
    lambda_2 = -.2
    dis.LAMBDA_1 = lambda_1
    dis.LAMBDA_2 = lambda_2
    N = DEFAULT.N * 20

    # we take a little more points
    facteur = 1
    dis.SIZE_DOMAIN_1 *= facteur
    dis.SIZE_DOMAIN_2 *= facteur
    dis.M1 = int(dis.M1*facteur)
    dis.M2 = int(dis.M2*facteur)

    dt = dis.DT

    #axis_freq = np.linspace(-pi / dt, pi / dt, N)
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[N//2] = 1/2
    # w = 2 pi k / (N)
    axis_freq = 2 * pi*all_k / N / dt


    fig, ax = plt.subplots()

    simulated_freq = memoised(frequency_simulation,
                           dis,
                           N,
                           number_samples=100)
    simulated_cv = simulated_freq[2] / simulated_freq[1]
    #ax.plot(axis_freq*dt, simulated_cv, label="simulation")

    if semi_discrete:
        semidiscrete = [analytic_robin_robin(dis, w=w, modified_time=0,
                semi_discrete=True, N=N) for w in axis_freq]
        ax.semilogx(axis_freq*dt, np.abs(np.array(semidiscrete)-simulated_cv), label="semi-discrete")

        semidiscrete = [analytic_robin_robin(dis, w=w, modified_time=2,
                semi_discrete=True, N=N) for w in axis_freq]
        ax.plot(axis_freq*dt, np.abs(np.array(semidiscrete)-simulated_cv), label="semi-discrete modified")

    if full_discrete:
        discrete = [analytic_robin_robin(dis, w=w,
            semi_discrete=not full_discrete, N=N)
                                            for w in axis_freq]
        #ax.plot(axis_freq*dt, semi_discrete_modif_time, "k--", label="discrete in space and time")
        #ax.semilogy(axis_freq*dt, np.abs(simulated_cv - np.array(discrete))/simulated_cv, label="discrete")
        ax.plot(axis_freq*dt, np.array(discrete), label="discrete")

    for (modified_time, modified_op, modified_space, label) in setup_modified:
        continuous_modified = [dis.analytic_robin_robin_modified(w=w,
                                                                 order_time=modified_time,
                                                                 order_equations=modified_space,
                                                                 order_operators=modified_op)
                                            for w in axis_freq]

        #ax.semilogy(axis_freq*dt, np.abs(simulated_cv - np.array(continuous_modified))/simulated_cv, label=label)
        ax.semilogx(axis_freq*dt, np.abs(np.array(continuous_modified) - np.array(simulated_cv)), label=label)


    ax.set_xlabel("$\\omega*\\delta t$")
    #ax.set_ylabel("$\\frac{\\hat{\\rho}-(\\hat{\\rho}_{{simulation}})}{\\hat{\\rho}_{{simulation}}}$")
    ax.set_ylabel("Convergence factor error: $|\\hat{\\rho} - \\hat{\\rho}_{{sim}}|$")
    ax.set_xlim(left=-5*np.pi/dt/N, right=axis_freq[-1]*dt)
    ax.set_ylim(bottom=0, top=1)
    ax.grid()
    fig.legend(loc="center right")
    return fig, ax

def visualize_modif_simu(dis, N, T, number_samples):
    """
        Compare the approaches used with modified equations :
        plot cv rate :
        - simulated
        - with discrete in space time if full_discrete == True
        - with continuous approach if continuous==True
        - with all the modified setups (modified in time,
                                        modified interface operators,
                                        modified space equations,
                                        label)
    """
    # 0.5; -0.5 is generally a good choice with our parameters
    dt = dis.DT

    #axis_freq = np.linspace(-pi / dt, pi / dt, N)
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[N//2] = 1/2
    # w = 2 pi k / (N)
    axis_freq = 2 * pi*all_k / N / dt

    simulated_freq = memoised(frequency_simulation,
                           dis,
                           N,
                           number_samples=number_samples)
    simulated_cv = simulated_freq[2] / simulated_freq[1]
    fig, ax = plt.subplots()
    print(simulated_freq.shape)
    ax.plot(axis_freq*dt, simulated_cv, label="simulation")

    discrete = [analytic_robin_robin(dis, w=w, semi_discrete=False, N=N)
                                        for w in axis_freq]
    ax.plot(axis_freq*dt, discrete, label="discrete")

    h1, h2 = dis.get_h()
    h = h2[0]
    dt_other, N_other = get_dt_N(h, dis.COURANT_NUMBER, T, dis.D1)

    axis_freq = np.flipud(pi/np.linspace(3*dt, T, N))
    all_factors = [dis.analytic_robin_robin_modified(w) for w in axis_freq]
    """
    continuous_modified = [dis.analytic_robin_robin_modified(w=w, Lambda_1=lambda_1, Lambda_2=lambda_2,
                                                             order_time=float('inf'),
                                                             order_equations=float('inf'),
                                                             order_operators=float('inf'))
                                        for w in axis_freq]
    """

    ax.plot(axis_freq*dt, all_factors, label="modified")

    """
    continuous_modified_std = [dis.analytic_robin_robin_modified(w=w, Lambda_1=.5, Lambda_2=-.5,
                                                             order_time=float('inf'),
                                                             order_equations=float('inf'),
                                                             order_operators=float('inf'))
                                        for w in axis_freq]

    ax.plot(axis_freq*dt, continuous_modified_std, label="modified with l=+-.5")
    """

    ax.set_xlabel("$\\omega*\\delta t$")
    ax.set_ylabel("$\\frac{\\hat{\\rho}-(\\hat{\\rho}_{{simulation}})}{\\hat{\\rho}_{{simulation}}}$")
    ax.set_xlim(left=-5*np.pi/dt/N, right=axis_freq[-1]*dt)
    ax.set_ylim(bottom=0, top=1)
    ax.grid()
    fig.legend(loc="center right")
    plt.show()
    return


def fig_validate_analysis_modif_approach():
    """
        Compare the equations used with modified equations :
        plot cv rate :
        - simulated
        - with continuous approach
        - with semi-discrete in space, modif in time
        - with modified equations, modified operators
        - with interface operator
    """
    dis = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    dis.D1 = dis.D2
    dis.C = 0
    dis.DT *= 1
    # 0.5; -0.5 is generally a good choice with our parameters
    lambda_1 = 0.1
    lambda_2 = -0.5
    N = DEFAULT.N * 10

    # we take a little more points
    facteur = 1
    dis.M1 = int(dis.M1*facteur)
    dis.M2 = int(dis.M2*facteur)

    dt = dis.DT

    axis_freq = np.linspace(-pi / dt, pi / dt, N)

    fig, ax = plt.subplots()

    simulated_freq = memoised(frequency_simulation,
                           dis,
                           N,
                           Lambda_1=lambda_1,
                           Lambda_2=lambda_2,
                           number_samples=5)
    simulated_cv = simulated_freq[2] / simulated_freq[1]
    nomodif_approach = [continuous_analytic_rate_robin_robin(dis, w=w,
                                                               Lambda_1=lambda_1,
                                                               Lambda_2=lambda_2)
                                        for w in axis_freq]
    continuous_modified_basic = [cv_rate.continuous_analytic_rate_robin_robin_modified_only_eq(dis,
                                        Lambda_1=lambda_1, Lambda_2=lambda_2, w=w)
                                        for w in axis_freq]
    continuous_modified_simpler = [cv_rate.continuous_analytic_rate_robin_robin_modified_only_eq_simple_formula(dis,
                                        Lambda_1=lambda_1, Lambda_2=lambda_2, w=w)
                                        for w in axis_freq]


    ax.plot(axis_freq, simulated_cv, label="simulation")
    ax.plot(axis_freq, nomodif_approach, label="continuous, not modified")
    ax.plot(axis_freq, continuous_modified_basic, label="continuous modified equations, initial formula")
    ax.plot(axis_freq, continuous_modified_simpler, label="continuous modified equations, simpler (but false) formula")

    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("$\\hat{\\rho}$")

    fig.legend()
    show_or_save("fig_validate_analysis_modif_approach")

def fig_validate_analysis_modif_approach():
    """
        Compare the equations used with modified equations :
        plot cv rate :
        - simulated
        - with continuous approach
        - with semi-discrete in space, modif in time
        - with modified equations, modified operators
        - with interface operator
    """
    dis = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    dis.D1 = dis.D2
    dis.C = 0
    dis.DT *= 1
    # 0.5; -0.5 is generally a good choice with our parameters
    lambda_1 = 0.1
    lambda_2 = -0.5
    N = DEFAULT.N * 10

    # we take a little more points
    facteur = 1
    dis.M1 = int(dis.M1*facteur)
    dis.M2 = int(dis.M2*facteur)

    dt = dis.DT

    axis_freq = np.linspace(-pi / dt, pi / dt, N)

    fig, ax = plt.subplots()

    simulated_freq = memoised(frequency_simulation,
                           dis,
                           N,
                           Lambda_1=lambda_1,
                           Lambda_2=lambda_2,
                           number_samples=5)
    simulated_cv = simulated_freq[2] / simulated_freq[1]
    nomodif_approach = [continuous_analytic_rate_robin_robin(dis, w=w,
                                                               Lambda_1=lambda_1,
                                                               Lambda_2=lambda_2)
                                        for w in axis_freq]
    continuous_modified_basic = [cv_rate.continuous_analytic_rate_robin_robin_modified_only_eq(dis,
                                        Lambda_1=lambda_1, Lambda_2=lambda_2, w=w)
                                        for w in axis_freq]
    continuous_modified_simpler = [cv_rate.continuous_analytic_rate_robin_robin_modified_only_eq_simple_formula(dis,
                                        Lambda_1=lambda_1, Lambda_2=lambda_2, w=w)
                                        for w in axis_freq]


    ax.plot(axis_freq, simulated_cv, label="simulation")
    ax.plot(axis_freq, nomodif_approach, label="continuous, not modified")
    ax.plot(axis_freq, continuous_modified_basic, label="continuous modified equations, initial formula")
    ax.plot(axis_freq, continuous_modified_simpler, label="continuous modified equations, simpler (but false) formula")

    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("$\\hat{\\rho}$")

    fig.legend()
    show_or_save("fig_validate_analysis_modif_approach")



def analysis_frequency_error(discretization, N, iteration=1, lambda_1=0.6139250052109033, fig=None, ax=None, legend=True):
    if fig is None:
        fig, ax = plt.subplots()
    def continuous_analytic_error_neumann(discretization, w):
        D1 = discretization.D1
        D2 = discretization.D2
        # sig1 is \sigma^1_{+}
        sig1 = np.sqrt(np.abs(w) / (2 * D1)) * (1 + np.abs(w) / w * 1j)
        # sig2 is \sigma^2_{-}
        sig2 = -np.sqrt(np.abs(w) / (2 * D2)) * (1 + np.abs(w) / w * 1j)
        return D1 * sig1 / (D2 * sig2)

    colors = ['r', 'g', 'y', 'm']
    for dis, col, col2 in zip(discretization, colors, colors[::-1]):
        # first: find a correct lambda : we take the optimal yielded by
        # continuous analysis : 0.6 (dirichlet neumann case : just put 1e13 in lambda_1)

        dt = dis.DT
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
                            number_samples=13, fftshift=True):
    fig, ax = plt.subplots()
    def continuous_analytic_error_neumann(discretization, w):
        D1 = discretization.D1
        D2 = discretization.D2
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
        #dis.DT /= 10
        dt = dis.DT
        axis_freq = np.linspace(-pi / dt, pi / dt, N)

        frequencies = memoised(frequency_simulation,
                               dis,
                               N,
                               Lambda_1=lambda_1,
                               number_samples=number_samples, NUMBER_IT=15)
        # plt.plot(axis_freq, frequencies[0], col2+"--", label=" initial frequency ")
        # plt.plot(axis_freq, frequencies[1], col, label=dis.name()+" after 1 iteration")
        #plt.plot(axis_freq, frequencies[1], col+"--", label=dis.name()+" frequential error after the first iteration")
        for i in range(1,14):
            lsimu, = ax.semilogy(axis_freq,
                    frequencies[i+1] / frequencies[i],
                     col)
            ax.annotate(dis.name(), xy=(axis_freq[0], frequencies[2][0] / frequencies[1][0]), xycoords='data', horizontalalignment='left', verticalalignment='top')


        real_freq_discrete = np.array([
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 Lambda_2=0,
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
                                 Lambda_2=0,
                                 semi_discrete=True,
                                 N=N) for w in axis_freq
        ]

        real_freq_semidiscrete_modified_time1 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 Lambda_2=0,
                                 semi_discrete=True,
                                 modified_time=1,
                                 N=N) for w in axis_freq
        ]
        real_freq_semidiscrete_modified_time2 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 Lambda_2=0,
                                 semi_discrete=True,
                                 modified_time=2,
                                 N=N) for w in axis_freq
        ]
        real_freq_semidiscrete_modified_time3 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 Lambda_2=0,
                                 semi_discrete=True,
                                 modified_time=3,
                                 N=N) for w in axis_freq
        ]
        real_freq_semidiscrete_modified_time4 = [
            analytic_robin_robin(dis,
                                 w=w,
                                 Lambda_1=lambda_1,
                                 Lambda_2=0,
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
        dt = dis.DT
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


def fig_colormap_one_sided_problem():
    dis = DEFAULT.new(FiniteVolumes)
    dis.D1 = dis.D2
    dis.C = 0
    cont_modified = functools.partial(cv_rate.continuous_analytic_rate_robin_robin_modified_only_eq, dis)
    def fun(x):
        return np.clip(cont_modified(Lambda_1=x[0], Lambda_2=-x[0], w=x[1]), 0, 1)
    def fun2(x):
        l = x[0]
        xi = x[1]
        return np.abs((l-np.sqrt(1j*xi + xi**2))**2/((l+np.sqrt(1j*xi + xi**2))**2))

    fig, ax = plot_3D_square(fun2, 0, 1.5, 1e-3, 5,  1000, 1000)
    ax.set_title("")
    ax.set_xlabel("$\\Lambda$")
    ax.set_ylabel("$\\xi$")
    plt.show()


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
            return abs(np.linalg.norm(np.array([analytic_robin_robin(dis,
                                             Lambda_1=x[0], Lambda_2=x[1],
                                             w=pi / (n * dt), semi_discrete=True, modified_time=3)
                                             - cont_modified(Lambda_1=x[0], Lambda_2=x[1], w=pi / (n * dt))
                        for n in (1, N)])))

        fig, ax = plot_3D_square(fun_me, -0, 2., -2., 0, 500, 100, fig=fig,
                                 subplot_param=subplot_param)
        ax.set_ylabel("$\\Lambda^2$")
        ax.set_title(dis.name() + ", modifiée en temps")

        subplot_param += 1
        """

        def fun_me(x):
            return abs(max([analytic_robin_robin(dis,
                                             Lambda_1=x[0], Lambda_2=x[1],
                                             w=pi / (n * dt), semi_discrete=True, modified_time=3)

                        for n in (1, N)]))

        fig, ax = plot_3D_square(fun_me, -0, 2., -2., 0, 500, 100, fig=fig,
                                 subplot_param=subplot_param)
        ax.set_ylabel("$\\Lambda^2$")
        ax.set_title(dis.name() + ", modifiée en temps")
        """
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
    cmap = cm.YlGnBu # remove this TODO
    surf = ax.pcolormesh(X, Y, Z, cmap=cmap)#, vmin=.15, vmax=1)
    #min=0.2, max=0.5
    if plot_colorbar:
        fig.subplots_adjust(right=0.8, hspace=0.5)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_ax.set_title("$\\max_s\\hat{\\rho}$")
        cbar_ax.set_title("$\\hat{\\rho}$")
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
    dt, N = discretization.DT, T/discretization.DT
    cont = functools.partial(
        continuous_analytic_rate_robin_neumann,
        discretization, l)
    return np.max([cont(pi / t) for t in np.linspace(dt, T, N)])


def to_minimize_analytic_robin_neumann(l, discretization, number_dt_h2, T):
    dt, N = discretization.DT, T/discretization.DT
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(discretization.analytic_robin_robin,
                          Lambda_1=l,
                          M1=M1,
                          M2=M2,
                          dt=dt)
    return max([f(pi / t * 1j) for t in np.linspace(dt, T, N)])

def to_minimize_continuous_analytic_rate_robin_robin(l, discretization, T):
    dis = discretization.clone()
    dt, N = dis.DT, T/dis.DT
    dis.LAMBDA_1, dis.LAMBDA_2 = l
    cont = functools.partial(continuous_analytic_rate_robin_robin, dis)
    return np.max([cont(pi / t) for t in np.linspace(dt, T, N)])

def to_minimize_continuous_analytic_rate_robin_robin_fullmodif(l, discretization, T):
    #l = (l[0]**2, -l[1]**2)
    dis = discretization.clone()
    dt, N = dis.DT, T/dis.DT
    dis.LAMBDA_1, dis.LAMBDA_2 = l

    axis_freq = pi/np.linspace(3*dt, T, N)
    all_factors = [dis.analytic_robin_robin_modified(w) for w in axis_freq]
    return np.max(all_factors)


def to_minimize_analytic_robin_robin(l, discretization, T):
    dis = discretization.clone()
    dt, N = dis.DT, T/dis.DT
    dis.LAMBDA_1 = l[0]
    dis.LAMBDA_2 = l[1]
    return max([dis.analytic_robin_robin(pi / t * 1j) for t in np.linspace(dt, T, N)])

def to_minimize_analytic_robin_robin_modified_eq(l, discretization, T):
    dis = discretization.clone()
    dt, N = dis.DT, T/dis.DT
    dis.LAMBDA_1 = l[0]
    dis.LAMBDA_2 = l[1]
    f = functools.partial(cv_rate.analytic_robin_robin, dis,
                          semi_discrete=True,
                          modified_time=3,
                          N=N)
    return max([f(pi / t) for t in np.linspace(dt, T, N)])

def to_minimize_analytic_robin_robin_semidiscrete_modif(l, discretization, T):
    dis = discretization.clone()
    dt, N = dis.DT, min(T/dis.DT, 1000)
    dis.LAMBDA_1 = l[0]
    dis.LAMBDA_2 = l[1]
    f = functools.partial(cv_rate.analytic_robin_robin, dis,
                          modified_time=4,
                          semi_discrete=True,
                          N=N)
    return max([f(pi / t) for t in np.linspace(3*dt, T, N)])

def to_minimize_analytic_robin_robin_semidiscrete(l, discretization, T):
    dis = discretization.clone()
    dt, N = dis.DT, min(T/dis.DT, 1000)
    dis.LAMBDA_1 = l[0]
    dis.LAMBDA_2 = l[1]
    f = functools.partial(cv_rate.analytic_robin_robin, dis,
                          semi_discrete=True,
                          N=N)
    return max([f(pi / t) for t in np.linspace(dt, T, N)])

def to_minimize_analytic_robin_robin_fulldiscrete(l, discretization, T):
    dis = discretization.clone()
    dt, N = dis.DT, T/dis.DT
    dis.LAMBDA_1 = l[0]
    dis.LAMBDA_2 = l[1]
    f = functools.partial(cv_rate.analytic_robin_robin, dis,
                          semi_discrete=False,
                          N=N)
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[int(N//2)] = .5

    # w = 2 pi k T / (N)
    axis_freq = 2 * pi*all_k / N / dt
    return max([f(w) for w in axis_freq])

def to_minimize_robin_robin_perfect(l, discretization, T, number_samples):
    dis = discretization.clone()
    dis.LAMBDA_1 = l[0]
    dis.LAMBDA_2 = l[1]
    w = frequency_simulation(dis, N,
                         number_samples=number_samples)
    return max(w[2] / w[1])


# The function can be passed in parameters of memoised:
to_minimize_analytic_robin_robin2 = FunMem(to_minimize_analytic_robin_robin)
to_minimize_analytic_robin_robin2_fulldiscrete = FunMem(to_minimize_analytic_robin_robin_fulldiscrete)
to_minimize_analytic_robin_robin2_semidiscrete = FunMem(to_minimize_analytic_robin_robin_semidiscrete)
to_minimize_analytic_robin_robin2_semidiscrete_modif = FunMem(to_minimize_analytic_robin_robin_semidiscrete_modif)
to_minimize_analytic_robin_robin2_modified = FunMem(to_minimize_analytic_robin_robin_modified_eq)
to_minimize_analytic_robin_robin2_fullmodified = FunMem(to_minimize_continuous_analytic_rate_robin_robin_fullmodif)
to_minimize_continuous_analytic_rate_robin_robin2 = \
        FunMem(to_minimize_continuous_analytic_rate_robin_robin)

to_minimize_robin_robin2_perfect = FunMem(to_minimize_robin_robin_perfect)

def compare_continuous_discrete_rate_robin_robin(fig, ax,
        discretization, T, number_dt_h2, steps=50, bounds_h=(0,2), legend=True,
        number_samples=500, plot_perfect_performances=False):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    assert number_dt_h2 == discretization.COURANT_NUMBER
    from scipy.optimize import minimize

    dt = discretization.DT
    N = int(T / dt)
    if N <= 1:
        print("ERROR BEGINNING: N is too small (<2)")

    all_h = np.linspace(bounds_h[0], bounds_h[1], steps)
    all_h = np.exp(all_h[::-1])

    def func_to_map(h, fun_to_minimize, x0, method, *args):
        discretization.M1 = int(discretization.SIZE_DOMAIN_1 / h)
        discretization.M2 = int(discretization.SIZE_DOMAIN_2 / h)
        dt, N = get_dt_N(h, number_dt_h2, T,
                         discretization.D1)
        discretization.DT = dt
        return memoised(minimize,
            fun=fun_to_minimize,
            x0=x0,
            method=method,
            args=(discretization, T, *args))


    from itertools import repeat
    print("Computing lambdas in discrete framework.")
    """
    ret_fullmodif = list(map(func_to_map, all_h,
        repeat(to_minimize_analytic_robin_robin2_fullmodified), repeat((2., -.3)), repeat("Nelder-Mead")))

    ret_fulldiscrete = list(map(func_to_map, all_h,
        repeat(to_minimize_analytic_robin_robin2_fulldiscrete), repeat((2., -.3)), repeat("Nelder-Mead")))
    """

    ret_discrete = list(map(func_to_map, all_h,
        repeat(to_minimize_analytic_robin_robin2_semidiscrete), repeat((2., -.3)), repeat("Nelder-Mead")))
    ret_discrete_modif = list(map(func_to_map, all_h,
        repeat(to_minimize_analytic_robin_robin2_semidiscrete_modif), repeat((2., -.3)), repeat("Nelder-Mead")))


    if plot_perfect_performances:
        perfect_performances = list(map(func_to_map, all_h,
            repeat(to_minimize_robin_robin2_perfect), repeat((2., -.3)), repeat("Nelder-Mead"), repeat(number_samples*4)))
        theorical_rate_perfect = [ret.fun for ret in perfect_performances]

    """
    optimal_fullmodif = [ret.x for ret in ret_fullmodif]
    theorical_rate_full_modif = [ret.fun for ret in ret_fullmodif]
    optimal_fulldiscrete = [ret.x for ret in ret_fulldiscrete]
    """
    optimal_discrete = [ret.x for ret in ret_discrete]
    optimal_discrete_modif = [ret.x for ret in ret_discrete_modif]
    theorical_rate_discrete = [ret.fun for ret in ret_discrete]
    theorical_rate_discrete_modif = [ret.fun for ret in ret_discrete_modif]

    """
    print("Computing lambdas in continuous framework.")
    ret_continuous = list(map(func_to_map, all_h,
        repeat(to_minimize_continuous_analytic_rate_robin_robin2), repeat((2., -.3)), repeat("Nelder-Mead")))

    optimal_continuous = [ret.x for ret in ret_continuous]
    theorical_cont_rate = [ret.fun for ret in ret_continuous]
    """

    rate_with_continuous_lambda = []
    rate_with_discrete_lambda = []
    rate_with_fulldiscrete_lambda = []
    fun_modified_on_opti_cont = []
    rate_with_discrete_modif_lambda = []
    rate_with_fullmodif_lambda = []

    try:
        for i in range(all_h.shape[0]):
            discretization.M1 = int(discretization.SIZE_DOMAIN_1 / all_h[i])
            discretization.M2 = int(discretization.SIZE_DOMAIN_2 / all_h[i])
            dt, N = get_dt_N(all_h[i], number_dt_h2, T, discretization.D1)
            discretization.DT = dt
            print("h number:", i)
            print("N :", N)
            """
            discretization.LAMBDA_1, discretization.LAMBDA_2 = optimal_continuous[i]
            rate_with_continuous_lambda += [
                    memoised(frequency_simulation, discretization, N,
                             number_samples=number_samples)]


            discretization.LAMBDA_1, discretization.LAMBDA_2 = optimal_fulldiscrete[i]
            rate_with_fulldiscrete_lambda += [
                memoised(frequency_simulation, discretization, N,
                         number_samples=number_samples) ]

            """
            discretization.LAMBDA_1, discretization.LAMBDA_2 = optimal_discrete[i]
            rate_with_discrete_lambda += [
                memoised(frequency_simulation, discretization, N,
                         number_samples=number_samples)
            ]

            discretization.LAMBDA_1, discretization.LAMBDA_2 = optimal_discrete_modif[i]
            rate_with_discrete_modif_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         number_samples=number_samples) ]

            """
            discretization.LAMBDA_1, discretization.LAMBDA_2 = optimal_fullmodif[i]
            rate_with_fullmodif_lambda += [
                memoised(frequency_simulation, discretization, N,
                         number_samples=number_samples) ]
            """

            try:
                pass
                #visualize_modif_simu(discretization, N, T, 50)
                """
                fonction = functools.partial(to_minimize_analytic_robin_robin2_fullmodified,
                        h=all_h[i], discretization=discretization, number_dt_h2=number_dt_h2, T=T)
                    
                fig, ax = plot_3D_square(fonction, xmin=0, xmax=2, ymin=-2, ymax=0, Nx=50, Ny=50)
                ax.set_title("")
                ax.set_xlabel("$\\Lambda$")
                ax.set_ylabel("$\\xi$")
                plt.show()
                """
            except:
                raise

    except:
        pass

    rate_with_continuous_lambda = [max(w[2] / w[1])
            for w in rate_with_continuous_lambda]
    rate_with_fulldiscrete_lambda = [max(w[2] / w[1])
            for w in rate_with_fulldiscrete_lambda]
    rate_with_discrete_lambda = [max(w[2] / w[1])
            for w in rate_with_discrete_lambda]
    rate_with_discrete_modif_lambda = [max(w[2] / w[1])
            for w in rate_with_discrete_modif_lambda]
    rate_with_fullmodif_lambda = [max(w[2] / w[1])
        for w in rate_with_fullmodif_lambda]

    #linefdo, = ax.semilogx(all_h[:len(rate_with_fulldiscrete_lambda)],
    #             rate_with_fulldiscrete_lambda,
    #             "y")
    lineco, = ax.semilogx(all_h[:len(rate_with_continuous_lambda)],
                 rate_with_continuous_lambda,
                 "r")
    # linect, = ax.semilogx(all_h[:len(theorical_cont_rate)],
    #              theorical_cont_rate,
    #              "r--")

    linefmo, = ax.semilogx(all_h[:len(rate_with_fullmodif_lambda)],
                 rate_with_fullmodif_lambda,
                 "m")
    # linefmt, = ax.semilogx(all_h[:len(theorical_rate_full_modif)],
    #              theorical_rate_full_modif,
    #              "m--")
    linemdo, = ax.semilogx(all_h[:len(rate_with_discrete_modif_lambda)],
                 rate_with_discrete_modif_lambda,
                 "b")
    linemdt, = ax.semilogx(all_h[:len(theorical_rate_discrete_modif)],
                  theorical_rate_discrete_modif,
                  "b--")

    linedo, = ax.semilogx(all_h[:len(rate_with_discrete_lambda)],
                 rate_with_discrete_lambda,
                 "g")
    linedt, = ax.semilogx(all_h[:len(theorical_rate_discrete)],
                  theorical_rate_discrete,
                  "g--")

    if plot_perfect_performances:
        linept, = ax.semilogx(all_h,
                     theorical_rate_perfect,
                     "k--")
        if legend:
            linept.set_label("Taux théorique avec optimisation directement sur le taux observé")

    if legend:
        linefmo.set_label("Taux observé avec optimisation sur les équations modifiées")
        #linefmt.set_label("Taux théorique avec optimisation sur les équations modifiées")
        #linefdo.set_label("Taux observé avec $\\Lambda$ optimal discret")
        linemdo.set_label("Taux observé avec $\\Lambda$ optimal semi-discret, modifié en temps")
        linedo.set_label("Taux observé avec $\\Lambda$ optimal semi-discret")
        #linedt.set_label("Taux théorique avec $\\Lambda$ optimal semi-discret")
        lineco.set_label("Taux observé avec $\\Lambda$ optimal continu")
        #linect.set_label("Taux théorique avec $\\Lambda$ optimal continu")
        ax.legend(loc="upper right")
        ax.grid()

    ax.set_xlabel("h")
    ax.set_ylabel("$\\hat{\\rho}$")
    ax.set_ylim(ymin=0., ymax=1.)
    fig.suptitle('Comparaison des analyses continues, discrètes et modifiées' +
            ' ' )
    ax.set_title('Nombre de Courant : $D_1\\frac{dt}{h^2}$ = ' + str(number_dt_h2))


def to_minimize_continuous_analytic_rate_robin_onesided(l,
        h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1)
    cont = functools.partial(
        continuous_analytic_rate_robin_robin,
        discretization, l, -l)
    return np.max([cont(pi / t) for t in np.linspace(dt, T, N)])

def to_minimize_continuous_analytic_rate_robin_onesided_fullmodif(l,
        h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1)
    cont = functools.partial(
        discretization.modified_equations_fun(),
        discretization, l, -l)

    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    all_k[N//2] = .5

    # w = 2 pi k T / (N)
    axis_freq = 2 * pi*all_k / N / dt

    return np.max([cont(w) for w in axis_freq])


def to_minimize_analytic_robin_onesided(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(discretization.analytic_robin_robin,
                          Lambda_1=l,
                          Lambda_2=-l,
                          M1=M1,
                          M2=M2,
                          dt=dt)
    return max([f(pi / t * 1j) for t in np.linspace(dt, T, N)])

def to_minimize_analytic_robin_onesided_modified_eq(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(cv_rate.analytic_robin_robin, discretization,
                          Lambda_1=l,
                          Lambda_2=-l,
                          M1=M1,
                          M2=M2,
                          semi_discrete=True,
                          modified_time=3,
                          N=N,
                          dt=dt)
    return max([f(pi / t) for t in np.linspace(dt, T, N)])

def to_minimize_analytic_robin_onesided_fulldiscrete(l, h, discretization, number_dt_h2, T):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    f = functools.partial(cv_rate.analytic_robin_robin, discretization,
                          Lambda_1=l,
                          Lambda_2=-l,
                          M1=M1,
                          M2=M2,
                          semi_discrete=False,
                          dt=dt,
                          N=N)
    return max([f(w) for w in list(pi/np.linspace(dt, T, N))])

def to_minimize_robin_onesided_perfect(l, h, discretization, number_dt_h2, T, number_samples):
    dt, N = get_dt_N(h, number_dt_h2, T, discretization.D1)
    M1 = int(discretization.SIZE_DOMAIN_1 / h)
    M2 = int(discretization.SIZE_DOMAIN_2 / h)
    lambda_1 = l
    lambda_2 = -l
    w = frequency_simulation(discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=lambda_1,
                         Lambda_2=lambda_2,
                         number_samples=number_samples,
                         dt=dt)
    return max(w[2] / w[1])

to_minimize_analytic_robin_onesided2 = FunMem(to_minimize_analytic_robin_onesided)
to_minimize_analytic_robin_onesided2_fulldiscrete = FunMem(to_minimize_analytic_robin_onesided_fulldiscrete)
to_minimize_analytic_robin_onesided2_modified = FunMem(to_minimize_analytic_robin_onesided_modified_eq)
to_minimize_analytic_robin_onesided2_fullmodified = FunMem(to_minimize_continuous_analytic_rate_robin_onesided_fullmodif)
to_minimize_continuous_analytic_rate_robin_onesided2 = \
        FunMem(to_minimize_continuous_analytic_rate_robin_onesided)

to_minimize_robin_onesided2_perfect = FunMem(to_minimize_robin_onesided_perfect)

def validation_theorical_modif_resolution_robin_onesided(fig, ax,
        discretization, T, number_dt_h2, steps=50, bounds_h=(0,2), legend=True,
        number_samples=500, plot_perfect_performances=False):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    from scipy.optimize import minimize_scalar as minimize

    D = discretization.D1
    assert D == discretization.D2

    all_h = np.linspace(bounds_h[0], bounds_h[1], steps)
    all_h = np.exp(all_h[::-1])

    def func_to_map(x): return memoised(minimize,
        fun=to_minimize_analytic_robin_onesided2,
        args=(x, discretization, number_dt_h2, T))

    def func_to_map_fulldiscrete(x): return memoised(minimize,
        fun=to_minimize_analytic_robin_onesided2_fulldiscrete,
        args=(x, discretization, number_dt_h2, T))

    def func_to_map_discrete_modif(x): return memoised(minimize,
        fun=to_minimize_analytic_robin_onesided2_modified,
        args=(x, discretization, number_dt_h2, T))

    def func_to_map_full_modif(x): return memoised(minimize,
        fun=to_minimize_analytic_robin_onesided2_fullmodified,
        args=(x, discretization, number_dt_h2, T))

    def func_to_map_theoric_modif(h):
        dt = h**2*number_dt_h2/D
        return cv_rate.continuous_best_lam_robin_onesided_modif_vol(
                discretization, dt, number_dt_h2, pi/T, pi/dt/3)

    def func_to_map_perfect_perf(x): 
        ret = memoised(minimize,
                fun=to_minimize_robin_onesided2_perfect,
                args=(x, discretization, number_dt_h2, T, number_samples*4))
        return ret


    print("Computing lambdas in discrete framework.")
    ret_discrete = list(map(func_to_map, all_h))
    ret_fulldiscrete = list(map(func_to_map_fulldiscrete, all_h))
    ret_discrete_modif = list(map(func_to_map_discrete_modif, all_h))

    ret_discrete_fullmodif = list(map(func_to_map_full_modif, all_h))
    ret_theoric_modif = list(map(func_to_map_theoric_modif, all_h))

    if plot_perfect_performances:
        perfect_performances = list(map(func_to_map_perfect_perf, all_h))
        theorical_rate_perfect = [ret.fun for ret in perfect_performances]

    optimal_theomodif = [ret[0] for ret in ret_theoric_modif]
    theorical_rate_theomodif = [ret[1] for ret in ret_theoric_modif]
    optimal_fullmodif = [ret.x for ret in ret_discrete_fullmodif]
    theorical_rate_full_modif = [ret.fun for ret in ret_discrete_fullmodif]
    optimal_discrete = [ret.x for ret in ret_discrete]
    optimal_fulldiscrete = [ret.x for ret in ret_fulldiscrete]
    optimal_discrete_modif = [ret.x for ret in ret_discrete_modif]
    theorical_rate_discrete = [ret.fun for ret in ret_discrete]
    theorical_rate_discrete_modif = [ret.fun for ret in ret_discrete_modif]

    print(list(zip(optimal_fullmodif, optimal_fulldiscrete)))
    def func_to_map_cont(x): return memoised(minimize,
        fun=to_minimize_continuous_analytic_rate_robin_onesided2,
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
    rate_with_fulldiscrete_lambda = []
    rate_with_discrete_modif_lambda = []
    rate_with_fullmodif_lambda = []
    rate_with_theomodif_lambda = []
    print("optimal-continuous[0]:", optimal_continuous[0])
    print("optimal-discrete[0]:", optimal_discrete[0])

    try:
        for i in range(all_h.shape[0]):
            dt, N = get_dt_N(all_h[i], number_dt_h2, T,
                             discretization.D1)
            print("h number:", i)
            M1 = int(discretization.SIZE_DOMAIN_1 / all_h[i])
            M2 = int(discretization.SIZE_DOMAIN_2 / all_h[i])
            rate_with_continuous_lambda += [
                    memoised(frequency_simulation, discretization,
                             N,
                             M1=M1,
                             M2=M2,
                             Lambda_1=optimal_continuous[i],
                             Lambda_2=-optimal_continuous[i],
                             number_samples=number_samples,
                             dt=dt)
                ]
            rate_with_discrete_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_discrete[i],
                         Lambda_2=-optimal_discrete[i],
                         number_samples=number_samples,
                         dt=dt)
            ]

            rate_with_fulldiscrete_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_fulldiscrete[i],
                         Lambda_2=-optimal_fulldiscrete[i],
                         number_samples=number_samples,
                         dt=dt)
            ]
            rate_with_discrete_modif_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_discrete_modif[i],
                         Lambda_2=-optimal_discrete_modif[i],
                         number_samples=number_samples,
                         dt=dt)
            ]
            rate_with_fullmodif_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_fullmodif[i],
                         Lambda_2=-optimal_fullmodif[i],
                         number_samples=number_samples,
                         dt=dt)
            ]
            rate_with_theomodif_lambda += [
                memoised(frequency_simulation, discretization,
                         N,
                         M1=M1,
                         M2=M2,
                         Lambda_1=optimal_theomodif[i],
                         Lambda_2=-optimal_theomodif[i],
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
    rate_with_fullmodif_lambda = [max(w[2] / w[1])
        for w in rate_with_fullmodif_lambda]
    rate_with_theomodif_lambda = [max(w[2] / w[1])
        for w in rate_with_theomodif_lambda]

    """
    linefdo, = ax.semilogx(all_h[:len(rate_with_fulldiscrete_lambda)],
                 rate_with_fulldiscrete_lambda,
                 "y")
    """

    linemdo, = ax.semilogx(all_h[:len(rate_with_discrete_modif_lambda)],
                 rate_with_discrete_modif_lambda,
                 "b")
    linemdt, = ax.semilogx(all_h,
                 theorical_rate_discrete_modif,
                 "b--")

    """
    linedo, = ax.semilogx(all_h[:len(rate_with_discrete_lambda)],
                 rate_with_discrete_lambda,
                 "g")
    linedt, = ax.semilogx(all_h,
                 theorical_rate_discrete,
                 "g--")
     """
    lineco, = ax.semilogx(all_h[:len(rate_with_continuous_lambda)],
                 rate_with_continuous_lambda,
                 "r")
    linect, = ax.semilogx(all_h[:len(theorical_cont_rate)],
                 theorical_cont_rate,
                 "r--")

    linefmo, = ax.semilogx(all_h[:len(rate_with_fullmodif_lambda)],
                 rate_with_fullmodif_lambda,
                 "m")
    linefmt, = ax.semilogx(all_h[:len(theorical_rate_full_modif)],
                 theorical_rate_full_modif,
                 "m--")

    linetfmo, = ax.semilogx(all_h[:len(rate_with_theomodif_lambda)],
                 rate_with_theomodif_lambda,
                 "y")
    linetfmt, = ax.semilogx(all_h[:len(theorical_rate_theomodif)],
                 theorical_rate_theomodif,
                 "y--")

    if plot_perfect_performances:
        linept, = ax.semilogx(all_h,
                     theorical_rate_perfect,
                     "k--")
        if legend:
            linept.set_label("Taux théorique avec optimisation directement sur le taux observé")

    if legend:
        #linefdo.set_label("Taux observé avec $\\Lambda$ optimal discret en temps et en espace")
        linemdo.set_label("Taux observé avec $\\Lambda$ optimal semi-discret, modifié en temps")
        linemdt.set_label("Taux théorique avec $\\Lambda$ optimal semi-discret, modifié en temps")
        #linedo.set_label("Taux observé avec $\\Lambda$ optimal semi-discret")
        #linedt.set_label("Taux théorique avec $\\Lambda$ optimal semi-discret")
        lineco.set_label("Taux observé avec $\\Lambda$ optimal continu")
        linect.set_label("Taux théorique avec $\\Lambda$ optimal continu")
        linefmo.set_label("Optimisation numérique sur les équations modifiées : taux observé")
        linefmt.set_label("Optimisation numérique sur les équations modifiées : taux théorique")
        linetfmo.set_label("Optimisation analytique sur les équations modifiées : taux observé")
        linetfmt.set_label("Optimisation analytique sur les équations modifiées : taux théorique")
        fig.legend(loc="lower left", ncol=2)

    ax.set_xlabel("h")
    ax.set_ylabel("$\\hat{\\rho}$")
    ax.grid()
    fig.suptitle('Comparaison des différentes analyses' +
            ' (Robin one-sided), ' +
              discretization.name())
    ax.set_title('Nombre de Courant : $D_1\\frac{dt}{h^2}$ = ' + str(number_dt_h2))



def error_by_taking_continuous_rate_constant_number_dt_h2(fig, ax,
        discretization, T, number_dt_h2, steps=50, bounds_h=(0,2), legend=True, number_samples=500):
    """
        We keep the ratio D*dt/(h^2) constant and we watch the
        convergence rate as h decreases.
    """
    from scipy.optimize import minimize_scalar

    dt = discretization.DT
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
                             discretization.D1)
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
              #+str(discretization.D1)
              #+', $D_2$='
              #+str(discretization.D2)
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
    dt = discretizations[0].DT
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

def figMainErrorTermModified():
    Co = np.linspace(0, 20, 10000)
    eps_1 = .5 + 1/(24*Co)
    eps_2 = 1/6 + 1/(24*Co) + 1/(288*Co) - 1/(1920*Co)
    eps_1_dif = .5 + 1/(12*Co)
    eps_2_dif = 1/6 + 1/(12*Co) + 1/(72*Co) - 1/(360*Co)
    plt.plot(Co,eps_2 / (eps_1**2))
    plt.plot(Co,eps_2_dif / (eps_1_dif**2))
    plt.show()
    
def figRhoModifiedOperators(l=1):
    wmin, wmax = 0, 10
    w = np.linspace(wmin, wmax, 1000)
    alpha_cpx = np.sqrt(1j*w)
    alpha = np.sqrt(w)
    rho_initial = np.abs((l-alpha_cpx*np.exp(alpha_cpx))**2/(l+alpha_cpx*np.exp(alpha_cpx))**2)

    rho_avant_calcul = np.abs((l-alpha*np.exp(1j*np.pi/4 + np.sqrt(2)*(1+1j)/2 * alpha))/(l+alpha*np.exp(1j*np.pi/4 + np.sqrt(2)/2*(1+1j)*alpha)))**2
    alpha /= np.sqrt(2)
    l /= np.sqrt(2)
    co = np.cos(alpha + np.pi / 4)
    si = np.sin(alpha + np.pi / 4)

    rho_avant_calcul = np.abs((l-alpha*np.exp(1j*(alpha+ np.pi/4) + alpha))**2/(l+alpha*np.exp(1j*(alpha+np.pi/4) + alpha))**2)

    rho_avant_calcul = np.abs((l - alpha*np.exp(alpha)*co - 1j*alpha*np.exp(alpha)*si)**2/(l + alpha*np.exp(alpha)*co + 1j*alpha*np.exp(alpha)*si)**2)

    rho_avant_calcul = np.abs((l**2 - (alpha*np.exp(alpha))**2 - 2*l*1j*alpha*np.exp(alpha)*si)/((l + alpha*np.exp(alpha)*co)**2 + (alpha*np.exp(alpha)*si)**2))**2

    rho_apres_calcul = (((l**2 - (alpha*np.exp(alpha))**2)**2 + (2*l*alpha*np.exp(alpha)*si)**2)/((l + alpha*np.exp(alpha)*co)**2 + (alpha*np.exp(alpha)*si)**2)**2)
    
    rho_apres_calcul = (((l**2 - (alpha*np.exp(alpha))**2)**2 + (2*l*alpha*np.exp(alpha)*si)**2)/((l + alpha*np.exp(alpha)*co)**2 + (alpha*np.exp(alpha)*si)**2)**2)

    rho_apres_calcul = (((l**2 - (alpha*np.exp(alpha))**2)**2 + (2*l*alpha*np.exp(alpha)*si)**2)/((l**2 + 2*l*alpha*np.exp(alpha)*co + alpha**2*np.exp(2*alpha))**2))
    #https://www.wolframalpha.com/input/?i=%28%28l%5E2-alpha%5E2e%5E%282*alpha%29%29%5E2+%2B+4*l%5E2*alpha%5E2*e%5E%282*alpha%29*sin%28alpha%2Bpi%2F4%29%5E2%29%2F%28l%5E2%2B2*l*alpha*e%5Ealpha*cos%28alpha%2Bpi%2F4%29%2Balpha%5E2*e%5E%282*alpha%29%29

    #with x : https://www.wolframalpha.com/input/?i=d%2Fdx%28%28%28l%5E2-x%5E2*e%5E%282*x%29%29%5E2+%2B+4*l%5E2*x%5E2*e%5E%282*x%29*sin%28x%2Bpi%2F4%29%5E2%29%2F%28l%5E2%2B2*l*x*e%5Ex*cos%28x%2Bpi%2F4%29%2Bx%5E2*e%5E%282*x%29%29%29

    plt.plot(w, rho_initial)
    plt.plot(w, rho_avant_calcul)
    plt.plot(w, rho_apres_calcul)
    plt.show()

def figProjectionComplexPlan(l1=1.2,freqmin=0.1, freqmax=1.5*pi):
    import matplotlib.patches as patches
    # w*dt is between 0 and pi. The multiplicative factor is loosely between 1/2 and 1
    N = 1000
    Nx = 1000
    Ny = 1000
    fig, ax = plt.subplots(1,1,figsize=[6.4 , 4.8])
    xmin, xmax = -0, 2*pi
    ymin, ymax = 0, 3
    l1 = 1.2

    def function_to_plot(a, b): #a +ib
            #return np.clip(, 0, 1.2)
            return ((l1 -a)**2+b**2)/((l1 +a)**2+b**2)
    from mpl_toolkits.mplot3d import Axes3D
    plot_colorbar = fig is None
    X = np.ones((Ny, 1)) @ np.reshape(np.linspace(xmin, xmax, Nx), (1, Nx))
    Y = (np.ones((Nx, 1)) @ np.reshape(np.linspace(ymin, ymax, Ny), (1, Ny))).T
    Z = np.array([[function_to_plot(x, y) for x, y in zip(linex, liney)]
                  for linex, liney in zip(X, Y)])
    from matplotlib import cm

    cmap = cm.YlGnBu #reverse_colourmap(cm.YlGnBu)
    surf = ax.pcolormesh(X, Y, Z, cmap=cmap)
    fig.subplots_adjust(right=0.8, hspace=0.5)
    curve_xi = np.linspace(freqmin, freqmax, 1000)
    curve = np.sqrt(1j*curve_xi + curve_xi**2)
    curve_x = np.real(curve)
    curve_y = np.imag(curve)
    amin = curve_x[0]
    amax = curve_x[-1]
    bmin = curve_y[0]
    bmax = curve_y[-1]
    ax.add_patch(patches.Rectangle((amin, bmin), amax-amin, bmax-bmin, fill=False, color="r"))
    #                  linewidth=1, edgecolor='r', facecolor='none')
    ax.plot(curve_x, curve_y, "k")
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax.set_title("$\\widehat{\\rho}(\\Lambda=1.2, x+iy)$")
    ax.set_xlabel("$\\mathfrak{R}(\\sqrt{i\\xi+\\xi^2})$")
    ax.set_ylabel("$\\mathfrak{Im}(\\sqrt{i\\xi+\\xi^2})$")
    fig.colorbar(surf, shrink=0.5, aspect=5, cax=cbar_ax)
    plt.show()

def beauty_graph_finite(discretization,
                        lambda_min,
                        lambda_max,
                        steps,
                        courant_number,
                        fig, ax, legend,
                        **kwargs):
    PARALLEL = True
    LAMBDA_2 = discretization.LAMBDA_2

    D1 = discretization.D1
    D2 = discretization.D2

    M1 = discretization.M1
    M2 = discretization.M2

    SIZE_DOMAIN_1 = discretization.SIZE_DOMAIN_1
    SIZE_DOMAIN_2 = discretization.SIZE_DOMAIN_2

    NUMBER_DDT_H2 = courant_number
    T = 100.

    DT = NUMBER_DDT_H2 * (SIZE_DOMAIN_1 / M1)**2 / D1
    # should not be too different from the value with M2, Size_domain2, and D2
    TIME_WINDOW_LEN = int(T / DT)
    rate_func = functools.partial(cv_rate.rate_freq_slow, discretization,
                                  TIME_WINDOW_LEN,
                                  seeds = range(100),
                                  **kwargs)
    rate_func_normL2 = functools.partial(cv_rate.rate_freq_slow,
                                         discretization,
                                         TIME_WINDOW_LEN,
                                         function_to_use=np.linalg.norm,
                                         seeds = range(100),
                                         **kwargs)
    rate_func.__name__ = "bgf_rate_func_freq" + discretization.repr() + str(TIME_WINDOW_LEN)
    rate_func_normL2.__name__ = "bgf_rate_func_normL2_freq" + discretization.repr() + str(TIME_WINDOW_LEN)

    from scipy.optimize import minimize_scalar, minimize

    lambda_1 = np.linspace(lambda_min, lambda_max, steps)
    dt = DT
    print("> Starting frequency analysis.")
    rho = []
    for t in np.linspace(dt, T, TIME_WINDOW_LEN):
        rho += [[analytic_robin_robin(discretization,
                                      w=pi / t,
                                      Lambda_1=i, semi_discrete=True,
                                      **kwargs) for i in lambda_1]]
        rho += [[analytic_robin_robin(discretization,
                                      w=-pi / t,
                                      Lambda_1=i, semi_discrete=True,
                                      **kwargs) for i in lambda_1]]

    continuous_best_lam = cv_rate.continuous_best_lam_robin_neumann(
        discretization, TIME_WINDOW_LEN)

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
    # Then select the names beginning with fig.
    # Note that we don't check if it is a function,
    # So that a user can give a callable (for example, with functools.partial)
    if key[:3] == "fig":
        all_figures[key] = glob
