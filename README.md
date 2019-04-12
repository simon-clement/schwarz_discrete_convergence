# schwarz_tbc_finder

You can add theses lines to your "~/.bashrc":
    chmod +x cv_rate
    complete -W "test graph optimize debug analytic figure" ./cv_rate

with theses commands the script can be launched with "./cv_rate ARG"
where ARG is one of the arguments {test, graph, optimize, debug, analytic, figure}
and where the completion can be used for the argument.

To make a figure that is inside the pdf, just use the argument figure:
example:
    ./cv_rate.py figure 2.3
will reproduce the figure 2.3.

To install the repository:
    git clone --recursive https://github.com/nuftau/rust_tbc_parab_schwarz
You can check your install is correct:
    ./cv_rate.py test


dependencies for the fast versions:
    rust installed
    libopenblas-base installed
    cffi installed
