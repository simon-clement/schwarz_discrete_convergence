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

def fig_compareSettingsDirichletNeumann():
    from discretizations.finite_volumes_spline2 import FiniteVolumesSpline2
    from discretizations.rk2_finite_volumes_spline2 import Rk2FiniteVolumesSpline2
    from discretizations.finite_difference_naive_neumann import FiniteDifferencesNaiveNeumann
    from discretizations.finite_difference import FiniteDifferences


    discretizations = {}
    # discretizations["$\\eta^\\text{FV}, s_d^\\text{Euler}$"] = DEFAULT.new(FiniteVolumesSpline2)
    # discretizations["$\\eta^\\text{FV}, s_d^\\text{RK2}$"] = DEFAULT.new(Rk2FiniteVolumesSpline2)
    # discretizations["$\\eta^\\text{FD}, s_d^\\text{Euler}, Corr=0$"] = DEFAULT.new(FiniteDifferences)
    # discretizations["$\\eta^\\text{FD}, s_d^\\text{Euler}, Corr=1$"] = DEFAULT.new(FiniteDifferencesNaiveNeumann)

    discretizations["FV, Euler"] = DEFAULT.new(FiniteVolumesSpline2)
    #discretizations["FV, RK2"] = DEFAULT.new(Rk2FiniteVolumesSpline2)
    discretizations["FD, Euler, Corr=0"] = DEFAULT.new(FiniteDifferencesNaiveNeumann)
    discretizations["FD, Euler, Corr=1"] = DEFAULT.new(FiniteDifferences)
    convergence_factors = {}
    theorical_convergence_factors = {}

    N = 1000
    dt = DEFAULT.DT
    ###########
    # Computation of the frequency axis
    ###########
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    # w = 2 pi k T / (N)
    axis_freq = 2 * pi*all_k / N / dt

    ###########
    # for each discretization, a simulation
    ###########
    for name in discretizations:
        discretizations[name].LAMBDA_2 = 0. # lambda=0 is a Neumann condition
        discretizations[name].LAMBDA_1 = 1e9  # extremely high lambda is a Dirichlet condition
        
        alpha_w = memoised(frequency_simulation, discretizations[name], N, number_samples=1000)
        k = 1
        convergence_factors[name] = alpha_w[k+1] / alpha_w[k]
        theorical_convergence_factors[name] = \
                discretizations[name].analytic_robin_robin_modified(w=axis_freq,
                        order_time=float('inf'), order_operators=float('inf'),
                        order_equations=float('inf'))
        plt.semilogx(axis_freq * dt, convergence_factors[name], label=name)
        plt.semilogx(axis_freq * dt, theorical_convergence_factors[name], "--", label="Theorical " + name)

    plt.legend()
    show_or_save("fig_compareSettingsDirichletNeumann")




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
