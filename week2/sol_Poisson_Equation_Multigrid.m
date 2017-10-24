function [result] = sol_Poisson_Equation_Multigrid(f, dom, param)

result = sol_Multigrid_Step(f, param.driving, dom);