function [result] = sol_Multigrid_Step(f, driving, dom)

% Initial guess
param.iterations = 1;
param.driving = driving;
v0 = sol_Poisson_Equation_GaussSeidel(f, dom, param);

% Compute residual
df_xx = sol_DiBwd(sol_DiFwd(f, 1));
df_yy = sol_DjBwd(sol_DjFwd(f, 1));
driving_f = df_xx + df_yy;

r = driving - driving_f;

% Compute error
[ni, nj] = size(r);
f = zeros(ni, nj);
param.driving = driving;
error = sol_Poisson_Equation_Axb(f, dom, param);

% Compute solution
result = v0 + error;