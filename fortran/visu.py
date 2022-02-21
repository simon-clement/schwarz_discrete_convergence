"""
    module created to work with fortran module output.
"""

import numpy as np
import matplotlib.pyplot as plt

def import_data(filename):
    """
        import data from fortran module.
        the files contain lines with 2 floats in each.
    """
    with open(filename, 'r') as file:
        array1 = []
        array2 = []
        for line in file:
            x1, x2 = line.split()
            array1 += [float(x1)]
            array2 += [float(x2)]
    return np.array(array1), np.array(array2)

def main(argv):
    """
        test function of the module. Opens and plots
        the data in the file given in argument.
    """
    array1, array2 = import_data(argv[1])
    plt.plot(array1, array2)
    plt.grid()
    plt.show()

if __name__ =="__main__":
    import sys
    main(sys.argv)
