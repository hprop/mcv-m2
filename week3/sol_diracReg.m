function y = sol_diracReg( x, epsilon )
%  Dirac function of x
%    sol_diracReg( x, epsilon ) Computes the derivative of the heaviside
%    function of x with respect to x. Regularized based on epsilon.

% Equation 13 from paper
y = epsilon ./ (pi * (epsilon^2 + x.^2));
