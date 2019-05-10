#!/usr/bin/python3
import figures

def main():
    import sys

    if len(sys.argv) == 1:
        print("to launch tests, use \"python3 cv_rate.py test\"")
        print("Usage: cv_rate {test, graph, optimize, debug, analytic}")
    else:
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

        elif sys.argv[1] == "all_figures":
            try:
                from label_to_figure import ALL_LABELS
                import concurrent.futures
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    list(executor.map(global_launch_figsave, list(ALL_LABELS.keys())))

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

        elif sys.argv[1] == "clean":
            import memoisation
            memoisation.clean()
            print("Memoisation folder cleaned.")

        elif sys.argv[1] == "test":
            import tests.test_linear_sys
            import tests.test_schwarz
            import tests.test_finite_volumes
            import tests.test_finite_differences
            import tests.test_finite_differences_no_corrective_term
            import tests.test_finite_differences_naive_neumann
            import tests.test_optimal_neumann_robin
            test_dict = {
                'linear_sys': tests.test_linear_sys.launch_all_tests,
                'schwarz': tests.test_schwarz.launch_all_tests,
                'fvolumes': tests.test_finite_volumes.launch_all_tests,
                'rate': tests.test_optimal_neumann_robin.launch_all_tests,
                'fdifferences': tests.test_finite_differences.launch_all_tests,
                'fdifferences_no_corr': tests.test_finite_differences_no_corrective_term.launch_all_tests,
                'fdifferences_naive': tests.test_finite_differences_naive_neumann.launch_all_tests
            }
            if len(sys.argv) > 2:
                test_dict[sys.argv[2]]()
            else:
                for test_func in test_dict.values():
                    test_func()
            import label_to_figure
            for val in label_to_figure.ALL_LABELS.values():
                try:
                    assert val in figures.all_figures
                except AssertionError:
                    print(val, "is not in figures.all_figures.")
                    raise
            print("All labels are reffering to an existing function.")

        elif sys.argv[1] == "debug":
            # defining discretizations:

            from discretizations.finite_difference import FiniteDifferences
            from discretizations.finite_difference_no_corrective_term \
                    import FiniteDifferencesNoCorrectiveTerm
            from discretizations.finite_difference_naive_neumann \
                    import FiniteDifferencesNaiveNeumann
            from discretizations.finite_volumes import FiniteVolumes
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


            finite_difference = FiniteDifferences(A_DEFAULT, C_DEFAULT, D1_DEFAULT,
                                                  D2_DEFAULT, M1_DEFAULT,
                                                  M2_DEFAULT, SIZE_DOMAIN_1,
                                                  SIZE_DOMAIN_2, LAMBDA_1_DEFAULT,
                                                  LAMBDA_2_DEFAULT, DT_DEFAULT)

            finite_difference_wout = \
                    FiniteDifferencesNoCorrectiveTerm(A_DEFAULT, C_DEFAULT, D1_DEFAULT,
                                                      D2_DEFAULT, M1_DEFAULT,
                                                      M2_DEFAULT, SIZE_DOMAIN_1,
                                                      SIZE_DOMAIN_2, LAMBDA_1_DEFAULT,
                                                      LAMBDA_2_DEFAULT, DT_DEFAULT)

            finite_difference_naive = \
                    FiniteDifferencesNaiveNeumann(A_DEFAULT, C_DEFAULT, D1_DEFAULT,
                                                      D2_DEFAULT, M1_DEFAULT,
                                                      M2_DEFAULT, SIZE_DOMAIN_1,
                                                      SIZE_DOMAIN_2, LAMBDA_1_DEFAULT,
                                                      LAMBDA_2_DEFAULT, DT_DEFAULT)

            finite_volumes = FiniteVolumes(A_DEFAULT, C_DEFAULT, D1_DEFAULT,
                                           D2_DEFAULT, M1_DEFAULT, M2_DEFAULT,
                                           SIZE_DOMAIN_1, SIZE_DOMAIN_2,
                                           LAMBDA_1_DEFAULT, LAMBDA_2_DEFAULT,
                                           DT_DEFAULT)

            """ You can now test any function here, without impacting the program."""
            if len(sys.argv) > 2:
                if sys.argv[2] == "1":  # ./cv_rate debug 1
                    pass
                elif sys.argv[2] == "2":  # ./cv_rate debug 2
                    pass

def global_launch_figsave(number_fig):
    import os
    os.system('nice ./main.py figsave ' + str(number_fig))
    os.system('sleep 4')
    print("ok")
    


if __name__ == "__main__":
    main()
