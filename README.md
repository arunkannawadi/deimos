# pydeimos
A Python implementation of the DEIMOS shape measurement method (http://arxiv.org/abs/1008.1076) with an optional novel metacal technique (not yet implemented).

## How to use
There is no installation needed. Simply clone this repository onto your machine and include it as follows:

    import sys
    sys.path.append('path_to_pydeimos/pydeimos')
    from pydeimos import deimos

and you're good to go. Alternatively, you could import everything to access the intermediate quantitites.
It is recommended to run the scripts in the tests/ folder to see if everything is in order.

## Dependency
Standard Python packages such as NumPy, SciPy etc.
GalSim: https://github.com/GalSim-developers/GalSim/tree/releases/2.1/galsim (to be made optional, for non-metacal case)
