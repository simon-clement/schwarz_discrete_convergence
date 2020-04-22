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
            figures.set_save_to_png()
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
                if sys.argv[2] == "Manfredi": # ./main.py debug Manfredi
                    from tests.test_Manfredi import launch_all_tests
                    launch_all_tests()

                if sys.argv[2] == "Manfredi_rhs": # ./main.py debug Manfredi
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
                if sys.argv[2] == "Manfredi": # ./main.py debug Manfredi
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
