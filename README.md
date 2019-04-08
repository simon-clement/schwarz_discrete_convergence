# schwarz_tbc_finder

You can add theses lines to your "~/.bashrc":
    chmod +x cv_rate
    complete -W "test graph optimize debug analytic figure" ./cv_rate

with theses commands the script can be launched with "./cv_rate ARG"
where ARG is one of the arguments {test, graph, optimize, debug, analytic, figure}
and where the completion can be used for the argument.

To make a figure that is inside the pdf, just use the argument figure:
example:
    ./cv_rate figure 2.3
will reproduce the figure 2.3.

