# Schwarz TBC finder

This repository the main code of my MCS thesis, where the aim is to analyze transparent boundary conditions
(TBC) of a coupled system formed by 1D diffusions equations.
The diffusions equations may have a variable coefficient (in space) and non-uniform discretization.
There is an accelerated version written in Rust in a submodule. It should only work on unix system, due to the use of openBLAS (and probably MacOS but it wasn't tested).

### To install the repository:
```
    git clone --recursive https://github.com/nuftau/schwarz_tbc_finder
```
You can check your install is correct:
```
    ./cv_rate.py test
```

You can add this line to your "~/.bashrc":
```
    complete -W "test graph optimize debug analytic figure frequency raw_simu" ./cv_rate.py
```

The script can be launched with
```
    ./cv_rate.py [ARG]
```
Or
```
    ./main.py [ARG]
```

where ARG is one of the arguments {test, graph, optimize, debug, analytic, figure}
and where the completion can be used for the argument.

### use the repository (not yet implemented)
To make a figure that is inside the pdf, just use the argument figure:
example:
```
    ./cv_rate.py figure 2.3
```
will reproduce the figure 2.3.


### dependencies for the fast versions

Rust language must be installed: 
    https://www.rust-lang.org/tools/install

libopenblas-base must also be installed:
```
    apt install libopenblas-base
```

The python package cffi must be available:
```
    pip3 install cffi
```
