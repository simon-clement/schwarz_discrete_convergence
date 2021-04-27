#!/usr/bin/python3
import test_dir.ocean_fd_be_flux_test
import test_dir.atmo_fd_be_flux_test

def launch_all_tests():
    test_dir.ocean_fd_be_flux_test.main()
    test_dir.atmo_fd_be_flux_test.main()

if __name__ == "__main__":
    launch_all_tests()
