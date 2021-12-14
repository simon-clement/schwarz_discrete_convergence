#!/usr/bin/python3
"""
    This module is the container of the generators of figures.
    The code is redundant, but it is necessary to make sure
    a future change in the default values won't affect old figures...
"""
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from memoisation import memoised
from simu1DEkman import Simu1dEkman
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath, amsfonts}"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = ':'
mpl.rcParams["grid.alpha"] = '0.7'
mpl.rcParams["grid.linewidth"] = '0.5'

DEFAULT_z_levels = np.linspace(0, 1500, 41)

def plot_FD(axes, sf_scheme, dt=60., N=1680,
        z_levels=DEFAULT_z_levels, name=None, style={}):
    if name == None:
        name = sf_scheme
    M = z_levels.shape[0] - 1
    simulator = Simu1dEkman(z_levels=z_levels,
            dt=dt, u_geostrophy=10.,
            K_mol=1e-4, C_D=1e-3, f=1e-4)
    u_0 = 10*np.ones(M)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    u, TKE, ustar, shear = simulator.FD(u_t0=u_0,
            sf_scheme=sf_scheme, forcing=forcing)

    axes[0].plot(np.real(u), simulator.z_half[:-1], **style)
    axes[1].plot(np.imag(u), simulator.z_half[:-1], **style)
    axes[2].plot(TKE, simulator.z_full, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)

def plot_FV(axes, sf_scheme, delta_sl, dt=60., N=1680,
        z_levels=DEFAULT_z_levels, name=None, style={}):
    if name == None:
        name = sf_scheme
    M = z_levels.shape[0] - 1
    z_star: float = .1
    simulator = Simu1dEkman(z_levels=z_levels,
            dt=dt, u_geostrophy=10.,
            K_mol=1e-4, C_D=1e-3, f=1e-4)
    # choosing u_0 linear so it can be the same FD, FV
    u_0 = 10*np.ones(M+1)
    forcing = 1j*simulator.f*simulator.u_g*np.ones((N+1, M))
    phi_0 = np.diff(u_0, append=11) / np.diff(z_levels, append=1300)
    phi_0[-1] = phi_0[-2] # correcting the last flux

    if sf_scheme in {"FV1 free", "FV2 free"}:
        k = bisect.bisect_right(z_levels[1:], delta_sl)
        zk = z_levels[k]
        h_tilde = z_levels[k+1] - delta_sl
        tau_sl = 1+z_star/delta_sl - \
                        1/np.log(1+delta_sl/z_star) \
                + (zk - (zk+z_star)*np.log(1+zk/z_star)) \
                / (delta_sl * np.log(1+delta_sl/z_star))

        u_constant = 10.
        u_deltasl = 10. # approximation
        phi_0[k] = np.abs(u_deltasl) / (z_star+delta_sl) / \
                np.log(1+delta_sl/z_star)
        u_tilde = u_constant - h_tilde/6 * phi_0[k]

        u_0[k] = (1+tau_sl) * u_tilde - tau_sl*h_tilde*phi_0[k]/3

    u, phi, TKE, ustar, shear = simulator.FV(u_t0=u_0[:-1],
            phi_t0=phi_0, sf_scheme=sf_scheme,
            delta_sl=delta_sl, forcing=forcing)

    z_fv, u_fv = simulator.reconstruct_FV(u,
            phi, sf_scheme, delta_sl=delta_sl)

    axes[0].plot(np.real(u_fv), z_fv, **style)
    axes[1].plot(np.imag(u_fv), z_fv, **style)
    axes[2].plot(TKE, simulator.z_half, **style, label=name)
    axes[3].plot(dt*np.array(range(len(ustar))), ustar, **style)

def fig_consistency_comparison():
    """
        Integrates for 1 day a 1D ekman equation
        with TKE turbulence scheme.
    """
    z_levels= np.linspace(0, 1500, 41)
    z_levels_les= np.linspace(0, 1500, 401)
    # for FV with FV interpretation of sf scheme,
    # the first grid level is divided by 2 so that
    # delta_{sl} is the same in all the schemes.
    dt = 60.
    N = 1680 # 28*60=1680

    fig, axes = plt.subplots(1,4, figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.7, right=0.99)
    sf_scheme_FV = "FV2 free"
    col_FDpure = "#488f31"
    col_FV1 = "#acb75b"
    col_FVfree = "#de425b"
    def style(col, linestyle='solid', **kwargs):
        return {"color": col, "linestyle": linestyle,
                "linewidth":0.8, **kwargs}

    plot_FD(axes, "FD pure", N=N, dt=dt, z_levels=z_levels,
            name="FD, M=40", style=style(col_FDpure))
    plot_FD(axes, "FD pure", N=N, dt=dt, z_levels=z_levels_les,
            name="FD, M=400", style=style(col_FDpure, "dashed"))
    plot_FV(axes, "FV1", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name="FV1, M=40", style=style(col_FV1))
    plot_FV(axes, "FV1", delta_sl=z_levels_les[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV1, M=400", style=style(col_FV1, "dashed"))
    plot_FV(axes, "FV2 free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels,
            name="FV2 free, M=40", style=style(col_FVfree))
    plot_FV(axes, "FV2 free", delta_sl=z_levels[1]/2,
            N=N, dt=dt, z_levels=z_levels_les,
            name="FV2 free, M=400", style=style(col_FVfree, "dashed"))

    axes[0].set_ylim(top=1500., bottom=0.)
    axes[1].set_ylim(top=1500., bottom=0.)
    axes[0].set_xlabel("wind speed (u, $m.s^{-1}$)")
    axes[0].set_ylabel("height (m)")
    axes[1].set_xlabel("wind speed (v, $m.s^{-1}$)")
    axes[1].set_ylabel("height (m)")
    axes[2].set_xlabel("energy (J)")
    axes[2].set_ylabel("height (m)")
    axes[3].set_ylabel("friction velocity (u*, $m.s^{-1}$)")
    axes[3].set_ylim(top=0.5, bottom=0.38)
    axes[3].set_xlabel("time (s)")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    show_or_save("fig_consistency_comparison")

def get_discrete_freq(N, dt, avoid_zero=True):
    """
        Computation of the frequency axis.
        Z transform gives omega = 2 pi k T / (N).
    """
    N = N + 1 # actually, the results of the simulator contains one more point
    if N % 2 == 0: # even
        all_k = np.linspace(-N/2, N/2 - 1, N)
    else: #odd
        all_k = np.linspace(-(N-1)/2, (N-1)/2, N)
    # Usually, we don't want the zero frequency so we use instead 1/T:
    if avoid_zero:
        all_k[int(N//2)] = .5
    return 2 * np.pi*all_k / N / dt

#############################################
# Utilities for saving, visualizing, calling functions
#############################################


def set_save_to_png():
    global SAVE_TO_PNG
    SAVE_TO_PNG = True
    assert not SAVE_TO_PDF and not SAVE_TO_PGF

def set_save_to_pdf():
    global SAVE_TO_PDF
    SAVE_TO_PDF = True
    assert not SAVE_TO_PGF and not SAVE_TO_PNG

def set_save_to_pgf():
    global SAVE_TO_PGF
    SAVE_TO_PGF = True
    assert not SAVE_TO_PDF and not SAVE_TO_PNG

SAVE_TO_PNG = False
SAVE_TO_PGF = False
SAVE_TO_PDF = False
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
    elif SAVE_TO_PDF:
        print("exporting to directory " + directory)
        import os
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + name_fig + '.pdf')
    else:
        try:
            import matplotlib as mpl
            import os
            os.makedirs(directory, exist_ok=True)
            mpl.rcParams['savefig.directory'] = directory
            fig = plt.get_current_fig_manager()
            fig.set_window_title(name_fig) 
        except:
            print("cannot set default directory or name")
        plt.show()

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
