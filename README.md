# D-CG: Dynamic Coarse-Graining  

A collection of tools for developing coarse-grained models for molecular simulations with consistent dynamic properties.

## iomk_lib
A small library for tools to run IOMK and IOMK-Gauss-Newton(GN).
### iomk_gn.py
Implements the main interface to iteratively optimzize the drift matrix for the auxiliary variable generalized Langevin thermostat as implmented in LAMMPS, using the IOMK[[1]](#1)[[2]](#2) and IOMK-GN[3] ("iterative optimization of memory kernels (Gauss-Newton)") method.





## References
<a id="1">[1]</a> 
Klippenstein, V. and van der Vegt, N. F. A.
Journal of Chemical Theory and Computation 2023 19 (4), 1099-1110
https://10.1021/acs.jctc.2c00871 

<a id="2">[2]</a> 
 Klippenstein and Nico F. A. van der Vegt 
 J. Chem. Phys. 28 July 2022; 157 (4): 044103. https://doi.org/10.1021/acs.jctc.2c00871

 <a id="3">[3]</a> 
Klippenstein, V.; Wolf N. and van der Vegt, N. F. A. (in preperatiion)
