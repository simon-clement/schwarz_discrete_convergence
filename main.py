#!/usr/bin/python3
"""
    main file of the project.
    You should have a PDF with the code with some figures inside.
    It is possible to replicate the figure 4 (for example) by using:
    ./main.py figure 4
    The code of a figure is in figures.py, in the function indexed
    by the file label_to_figure.py
    all function name begin with "fig_".
    you can also generate a figure named "fig_foo_bar" by using:
    ./main.py figname fig_foo_bar
    You can generate all figures (WILL TAKE A LOT OF TIME : >10 HOURS)
    by using the command "./main.py all_figures".

    All the results are stored in cache_npy to allow a fast re-generation
    of figures. You can clean cache with "./main.py clean".
"""
import figures

def main():
    import sys

    if len(sys.argv) == 1:
        print("to launch tests, use \"python3 main.py test\"")
        print("Usage: main.py {test, graph, optimize, debug, analytic}")
    else:
        #  example of use : ./main.py figure 17
        if sys.argv[1] == "figure":
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

        #  example of use : ./main.py figsave 17
        elif sys.argv[1] == "figsave":
            figures.set_save_to_pdf()
            from label_to_figure import ALL_LABELS
            if len(sys.argv) == 2:
                print("Please enter the id of the figure in the paper.")
                print("The following ids are allowed:")
                print(list(ALL_LABELS.keys()))
            else:
                if sys.argv[2] in ALL_LABELS:
                    print("Function found. Plotting figure...")
                    import matplotlib
                    matplotlib.use('Agg')
                    figures.all_figures[ALL_LABELS[sys.argv[2]]]()
                else:
                    print("id does not exist. Please use one of:")
                    print(list(ALL_LABELS.keys()))

        elif sys.argv[1] == "figsavepgf":
            # Does not work yet.
            figures.set_save_to_pgf()
            from label_to_figure import ALL_LABELS
            if len(sys.argv) == 2:
                print("Please enter the id of the figure in the paper.")
                print("The following ids are allowed:")
                print(list(ALL_LABELS.keys()))
            else:
                if sys.argv[2] in ALL_LABELS:
                    print("Function found. Plotting figure...")
                    import matplotlib
                    matplotlib.use('Agg')
                    figures.all_figures[ALL_LABELS[sys.argv[2]]]()
                else:
                    print("id does not exist. Please use one of:")
                    print(list(ALL_LABELS.keys()))


        #  example of use : ./main.py all_figures
        # WARNING THIS TAKES MULTIPLE HOURS IF YOUR CACHE IS EMPTY
        elif sys.argv[1] == "all_figures":
            try:
                from label_to_figure import ALL_LABELS
                import concurrent.futures
                if len(sys.argv) > 2:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        list(executor.map(global_launch_figsave, list(ALL_LABELS.keys())))
                else:
                    print("sequentially exporting all figures.")
                    # I prefer doing it sequentially
                    list(map(global_launch_figsave, list(ALL_LABELS.keys())))

            except:
                raise
                # We cannot plot them in parallel...
                # matplotlib won't work if you import it only once :/
                # if you want to do all figures in parallel,
                # you'll need an external script
                from label_to_figure import ALL_LABELS
                figures.set_save_to_png()
                for fig in ALL_LABELS.values():
                    figures.all_figures[fig]()

        #  example of use : ./main.py figname fig_rho_robin_neumann
        elif sys.argv[1] == "figname":
            if len(sys.argv) == 2:
                print("Please enter the name of the figure function.")
                print("The following names are allowed:")
                print('\n'.join(figures.all_figures))
            else:
                if sys.argv[2] in figures.all_figures:
                    print("Function found. Plotting figure...")
                    figures.all_figures[sys.argv[2]]()
                else:
                    print("This name does not exist. Please use one of:")
                    print('\n'.join(figures.all_figures))

        # clean cache. 
        # If you want to clean the cache of only one function named fun,
        # just delete the folder "cache_npy/fun/"
        # example of use : ./main.py clean
        elif sys.argv[1] == "clean":
            import memoisation
            memoisation.clean()
            print("Memoisation folder cleaned.")

        # Verify installation, and run non-regression tests
        # example of use : ./main.py test
        elif sys.argv[1] == "test":
            if len(sys.argv) > 2:
                if sys.argv[2] == "PadeLowTildeGamma": # ./main.py debug PadeLowTildeGamma
                    from tests.test_Manfredi import launch_all_tests
                    launch_all_tests()

                if sys.argv[2] == "PadeLowTildeGamma_rhs": # ./main.py debug PadeLowTildeGamma
                    from tests.test_Manfredi_rhs_c import launch_all_tests
                    launch_all_tests()

                if sys.argv[2] == "FV2":  # ./main.py debug 1
                    from tests.test_finite_volumes_spline2 import launch_all_tests
                    launch_all_tests()

                if sys.argv[2] == "FV4":  # ./main.py debug 1
                    from tests.test_finite_volumes import launch_all_tests
                    launch_all_tests()

                elif sys.argv[2] == "FD":  # ./main.py debug 2
                    from tests.test_finite_differences import launch_all_tests
                    launch_all_tests()
            else:
                from tests.test_finite_volumes_spline2 import launch_all_tests
                launch_all_tests()
                from tests.test_finite_volumes import launch_all_tests
                launch_all_tests()
                from tests.test_finite_differences import launch_all_tests
                launch_all_tests()
            import label_to_figure
            for val in label_to_figure.ALL_LABELS.values():
                try:
                    assert val in figures.all_figures
                except AssertionError:
                    print(val, "is not in figures.all_figures.")
                    raise
            print("All labels are reffering to an existing function.")

        # example of use : ./main.py debug 1
        # don't overuse this.
        # It is here to tests things with default parameters,
        # not to export figures
        elif sys.argv[1] == "debug":
            """ You can now test any function here, without impacting the program."""
            if len(sys.argv) > 2:
                if sys.argv[2] == "PadeLowTildeGamma": # ./main.py debug PadeLowTildeGamma
                    from tests.test_Manfredi import launch_all_tests
                    launch_all_tests()

                if sys.argv[2] == "FV2":
                    from tests.test_finite_volumes_spline2 import launch_all_tests
                    launch_all_tests()

                if sys.argv[2] == "FV4":
                    from tests.test_finite_volumes import launch_all_tests
                    launch_all_tests()

                elif sys.argv[2] == "FD":
                    from tests.test_finite_differences import launch_all_tests
                    launch_all_tests()

                elif sys.argv[2] == "PadeFV":
                    from cv_factor_pade import rho_Pade_FV
                    from figures import get_discrete_freq, Builder
                    builder = Builder()
                    builder.LAMBDA_1 = 1e9 # optimal parameters for corr=0, N=3000
                    builder.LAMBDA_2 = -0.
                    builder.M1 = 1000
                    builder.M2 = 1000
                    builder.D1 = 1.
                    builder.D2 = 2.
                    builder.R = .0
                    builder.DT = 1e-5
                    N=10000
                    w = get_discrete_freq(N, builder.DT)
                    rho_Pade_FV(builder, w)
                elif sys.argv[2] == "rootsPadeFV":
                    import numpy as np
                    from cv_factor_pade import rho_Pade_FV
                    from figures import get_discrete_freq, Builder
                    from memoisation import memoised, FunMem
                    from validation_pade_fv import invert_u_phi
                    from discretizations.space.quad_splines_fv import QuadSplinesFV
                    from discretizations.time.PadeSimpleGamma import PadeSimpleGamma
                    from simulator import firstlevels_errors
                    from simulator import linear_regression_cplx
                    builder = Builder()
                    builder.LAMBDA_1 = 1e9 # optimal parameters for corr=0, N=3000
                    builder.LAMBDA_2 = -0.
                    builder.M1 = 100
                    builder.M2 = 100
                    builder.D1 = 1.
                    builder.D2 = 2.
                    builder.R = .0
                    builder.DT = 1e-2
                    N=1000
                    w = get_discrete_freq(N, builder.DT)
                    phi_samples, u_samples = memoised(Builder.simulation_firstlevels,
                            builder, PadeSimpleGamma, QuadSplinesFV, 
                            N=N, number_samples=30)

                    # We give the 0 location for {phi,u}[sample, frequency, location]
                    A_k, A_kprime = invert_u_phi(builder, w, phi_samples[:, :, 0], u_samples[:, :, 0])
                    # then we do a 2D linear regression, with as input A_k A_kprime
                    lambda_1, lambda_2 = 1j*np.zeros(A_k.shape[1]), 1j*np.zeros(A_k.shape[1])
                    for i in range(A_k.shape[1]):
                        x = np.array([(A_k[k,i], A_kprime[k,i]) for k in range(A_k.shape[0])])
                        # and as output, phi_samples[:, :, 1]
                        lambda_1[i], lambda_2[i] = linear_regression_cplx(x, phi_samples[:, i, 1])
                    import matplotlib.pyplot as plt

                    from cv_factor_pade import lambda_Pade_FV
                    lam1, lam2, lam3, lam4 = lambda_Pade_FV(builder, w)

                    plt.semilogx(w, lambda_1)
                    plt.semilogx(w, lam1, "--")
                    plt.semilogx(w, lambda_2)
                    plt.semilogx(w, lam2, "--")
                    plt.semilogx(w, lam3, "--")
                    plt.semilogx(w, lam4, "--")
                    plt.show()


def global_launch_figsave(number_fig):
    """
        This function launch an external python context for figure number_fig.
        It allows to create a new matplotlib context, which is needed
            to create different figures.
        it is slightly better to launch this function sequentially
            because of the risk of concurrent access to cache.
        By launching multiple figures in parallel you will gain time
            but you may do some computations multiple times.
    """
    import os
    os.system('nice ./main.py figsave ' + str(number_fig))


if __name__ == "__main__":
    main()
