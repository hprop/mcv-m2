function [result] = multigrid_Step(f, driving, dom)

%% Compute initial guess
param.iterations = 1;
param.driving = driving;
guess = sol_Poisson_Equation_GaussSeidel(f, dom, param);


%% Compute residual
guess_xx = sol_DiBwd(sol_DiFwd(guess, 1));
guess_yy = sol_DjBwd(sol_DjFwd(guess, 1));
guess_lap = guess_xx + guess_yy;

mask = logical(driving);
a = (guess_lap .* mask) + (guess .* ~mask);
b = (f .* ~mask) + driving;

res = b - a;


%% Compute error
res_h = res(1:2:end, 1:2:end);
dom_h = dom(1:2:end, 1:2:end);
p = struct();  % avoiding set driving force
err_h = sol_Poisson_Equation_Axb(res_h, dom_h, p);


%% Reconstruct solution
err = imresize(err_h, size(guess));  % interpolation
result = guess + err;


%% Refine with Gauss-Seidel
param.iterations = 1;
param.driving = driving;
result = sol_Poisson_Equation_GaussSeidel(result, dom, param);
