# Schwarz TBC finder

You may need to install scipy and matplotlib to make it work:
```
    pip3 install scipy matplotlib
```

The script can be launched with
```
    ./main.py figsave [number of the Fig.]
```


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

