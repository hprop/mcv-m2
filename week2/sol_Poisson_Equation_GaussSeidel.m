function [result] = sol_Poisson_Equation_GaussSeidel(f, dom, param, w_param)
% dom -> logical matrix, if 1 then that pixel belongs to the domain
% param.iterations -> number of iterations applied by the algorithm

[ni, nj]=size(f);

%We add the ghost boundaries (for the boundary conditions)
f_ext = zeros(ni+2, nj+2);
f_ext(2:end-1, 2:end-1) = f;
dom_ext =zeros(ni+2, nj+2);
dom_ext(2:end-1, 2:end-1) = dom;

% Add padding to the driving parameter
b = zeros(ni+2, nj+2);
b(2:end-1, 2:end-1) = param.driving;

if (isfield(param, 'iterations'))
    iter = param.iterations;
else
    iter = 16;
end

for n=1:iter
    for i=1:ni  % TODO: boundary in the image border??
        for j=1:nj
            if dom_ext(i, j) == 1
                n = f_ext(i-1, j);
                e = f_ext(i, j+1);
                s = f_ext(i+1, j);
                w = f_ext(i, j-1);
                f_ext(i, j) = w_param * (-b(i, j) + n + e + s + w) / 4 + (1-w_param) * f_ext(i,j);
            end
        end
    end
end

result = f_ext(2:end-1, 2:end-1);