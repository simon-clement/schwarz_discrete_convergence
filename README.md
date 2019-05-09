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

You may need to install scipy to make it work:
```
    pip3 install scipy
```
Or simply (Ubuntu systems):
```
    ./setup.py
```
You may be asked for sudo password because of the multiples installations.

You can add this line to your "~/.bashrc":
```
    complete -W "figure figname clean test" ./cv_rate.py
```

The script can be launched with
```
    ./cv_rate.py [ARG]
```
Or
```
    ./main.py [ARG]
```

Where ARG is one of the arguments {test, debug, figure, clean, figname}.

### Use the code
To make a figure that is inside the pdf, just use the argument figure:
example:
```
    ./cv_rate.py figure 2.3
```
will reproduce the figure 2.3.


### Dependencies for the fast versions
All dependencies can be satisfied with the setup script (Ubuntu systems):
```
    ./setup.py
```
You may be asked for sudo password because of the multiples installation
