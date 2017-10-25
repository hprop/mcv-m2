Week 2 - Poisson Image Editing
==============================

Lena example: 2 implementations
-------------------------------
Run `start.m` on the folder week2 to run seamless cloning over Lena.

`start.m` use the sol_D**wd.m functions to compute the Laplacian of
the images. An alternative method, using Matlab's gradient function is
provided in the `start_gradient.m` script file.


Additional examples
-------------------
The scripts `start_img0*.m` run additional experiments with new
images.

Please, see the attached slides, where the experiments are explained
in detail.


Iterative methods
-----------------
Beyond the analytical solution implemented in `sol_Poisson_Equation_Axb.m`,
we provide two iterative methods for the shake of comparison:

- Gauss-Seidel with w-relaxation, in `sol_Poisson_Equation_GaussSeidel.m`
- Multigrid (2 levels grid implementation), in `sol_Poisson_Equation_Multigrid.m`
