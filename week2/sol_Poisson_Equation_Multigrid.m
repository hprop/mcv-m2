function [result] = sol_Poisson_Equation_Multigrid(f, dom, param)
%% Poisson Equation solved by Multigrid method
% This code implements only the 2-grid version.
%
% Additional parameters:
% 'param.iterations': (default 5) number of multigrid iterations
% 'param.driving': (default zero-matrix) matrix specifying the driving force

if (isfield(param, 'iterations'))
    iter = param.iterations;
else
    iter = 4;
end

result = f;

for i = 1:iter
    result = multigrid_Step(result, param.driving, dom);
end
