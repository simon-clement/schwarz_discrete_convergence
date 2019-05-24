# Schwarz TBC finder

This repository is the main code of my MCS thesis, where the aim is to analyze transparent boundary conditions
(TBC) of a coupled system formed by 1D diffusion equations.
The diffusion equations may have variable coefficients (in space) and non-uniform discretization.

### Install the repository:
```
    git clone --recursive https://github.com/nuftau/schwarz_tbc_finder
```
You can check your install is correct:
```
    ./main.py test
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

The script can be launched with
```
    ./main.py [ARG]
```

Where ARG is one of the arguments {test, figure, figsave, figname, clean}.

### Use the code
To make a figure that is inside the pdf, just use the argument figure:
example:
```
    ./main.py figure 2.3
```
will reproduce the figure 2.3.

To export it, you can use:
```
    ./main.py figsave 2.3
```

The file "label\_to\_figure.py" gives the map between the id of the figure and the name of the function. The function is then inside the module "figures.py".

Any function in the "figures.py" prefixed by "fig\_" can be executed the following way:
```
    ./main.py figname fig_plot3D_function_to_minimize #  will execute the so-called function
```

Almost every computation goes to a persistent cache on the disk. If you change something in the computations, you might need to run:
```
    ./main.py clean
```


### Dependencies for the fast versions
There is an accelerated version written in Rust in a submodule. The accelerated part should only work on unix systems and has additional dependencies: everything works without it, but the pure Python version is ~10 times slower. The results with the rust version appear to have a greater variance. It may be better to avoid using it. 
All dependencies can be satisfied with the setup script (Ubuntu systems):
```
    ./setup.py
```
You may be asked for sudo password because of the multiples installation
