function [result] = sol_Poisson_Equation_Multigrid(f, dom, param)

result = f;
for i = 1:5
    result = multigrid_Step(result, param.driving, dom);
end
