# Schwarz TBC finder

This repository is the main code of my MCS thesis, where the aim is to analyze transparent boundary conditions
(TBC) of a coupled system formed by 1D diffusion equations.
The diffusion equations may have variable coefficients (in space) and non-uniform discretization.
There is an accelerated version written in Rust in a submodule. The accelerated part should only work on unix systems and has additional dependencies: everything works without it, but the pure Python version is ~10 times slower.

### Install the repository:
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

where ARG is one of the arguments {test, graph, optimize, debug, analytic, figure, frequency, raw_simu}
and where the completion can be used for the argument.

### Use the repository (not yet implemented)
To make a figure that is inside the pdf, just use the argument figure:
example:
```
    ./cv_rate.py figure 2.3
```
will reproduce the figure 2.3.


### Dependencies for the fast versions

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
