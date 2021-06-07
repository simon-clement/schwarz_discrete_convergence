#!/usr/bin/python3
import test_dir.ocean_pade_fv_test
import test_dir.ocean_pade_fd_test
import test_dir.atmo_pade_fv_test
import test_dir.atmo_pade_fd_test
import test_dir.atmo_be_fd_test

def launch_all_tests():
    test_dir.ocean_pade_fv_test.main()
    test_dir.ocean_pade_fd_test.main()
    test_dir.atmo_pade_fv_test.main()
    test_dir.atmo_be_fd_test.main()
    test_dir.atmo_pade_fd_test.main()
    print("The tests of the models are all successful")

if __name__ == "__main__":
    launch_all_tests()
