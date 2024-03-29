# Numerical analysis for the reconciliation of the discretizations of the air-sea exchanges and their parameterization

This repository is the main code of my PhD thesis, where the aim is to analyze air-sea exchanges
and Schwarz methods on the ocean-atmosphere coupled system.
The code of Figures in the manuscript can be found in some of the branches of this repository.

The script can be launched with
```
    ./main.py [ARG]
```

Where ARG is one of the arguments {figure, figsave, figname, clean}.

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

A accelerated version of the code (in Rust) can be found but it was eventually not used in the manuscript.
