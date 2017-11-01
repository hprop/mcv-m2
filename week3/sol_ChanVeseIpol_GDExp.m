function [ phi ] = sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni )
%Implementation of the Chan-Vese segmentation following the explicit
%gradient descent in the paper of Pascal Getreur "Chan-Vese Segmentation".
%It is the equation 19 from that paper

%I     : Gray color image to segment
%phi_0 : Initial phi
%mu    : mu lenght parameter (regularizer term)
%nu    : nu area parameter (regularizer term)
%eta   : epsilon for the total variation regularization
%lambda1, lambda2: data fidelity parameters
%tol   : tolerance for the sopping criterium
% epHeaviside: epsilon for the regularized heaviside.
% dt     : time step
%iterMax : MAximum number of iterations
%reIni   : Iterations for reinitialization. 0 means no reinitializacion

[ni,nj]=size(I);
hi=1;
hj=1;


phi=phi_0;
dif=inf;
nIter=0;
while dif>tol && nIter<iterMax

    phi_old=phi;
    nIter=nIter+1;


    % Fixed phi, Minimization w.r.t c1 and c2 (constant estimation)
    % c1: mean grey value for pixels inside the boundary (eq. 10)
    % c2: mean grey value for pixels outside the boundary (eq. 11)
    c1 = sum((phi(:) >= 0) .* I(:)) / sum(phi(:) >= 0);
    c2 = sum((phi(:) < 0) .* I(:)) / sum(phi(:) < 0);

    % Boundary conditions. Duplicate pixels near the borders
    % (equation 25 from the paper).
    phi(1,:)   = phi(2,:);
    phi(end,:) = phi(end-1,:);

    phi(:,1)   = phi(:,2);
    phi(:,end) = phi(:,end-1);

    % Regularized Dirac's Delta computation
    delta_phi = sol_diracReg(phi, epHeaviside);   %notice delta_phi=H'(phi)

    % Derivatives estimation
    % i direction, forward finite differences
    phi_iFwd  = DiFwd(phi);
    phi_iBwd  = DiBwd(phi);

    % j direction, forward finitie differences
    phi_jFwd  = DjFwd(phi);
    phi_jBwd  = DjBwd(phi);

    % centered finite diferences
    phi_icent = (phi_iFwd + phi_iBwd) ./ 2;
    phi_jcent = (phi_jFwd + phi_jBwd) ./ 2;

    % A and B estimation (A y B from the Pascal Getreuer's IPOL paper "Chan
    % Vese segmentation. Equation 18.
    A = mu ./ sqrt(eta.^2 + phi_jFwd.^2 + phi_icent.^2);
    B = mu ./ sqrt(eta.^2 + phi_jcent.^2 + phi_iFwd.^2);


    %%Equation 22, for inner points
    expr1 = A(2:end-1,2:end-1) .* phi(3:end,2:end-1) + ...
            A(1:end-2,2:end-1) .* phi(1:end-2,2:end-1) + ...
            B(2:end-1,2:end-1) .* phi(2:end-1,3:end) + ...
            B(2:end-1,2:end-1) .* phi(2:end-1,1:end-2) + ...
            -nu - lambda1 .* (I(2:end-1,2:end-1) - c1).^2 + ...
            lambda2 .* (I(2:end-1,2:end-1) - c2).^2;

    expr2 = 1 + dt .* delta_phi(2:end-1,2:end-1) .* ...
            (A(2:end-1,2:end-1) + A(1:end-2,2:end-1) + ...
             B(2:end-1,2:end-1) + B(2:end-1,1:end-2));

    phi(2:end-1,2:end-1) = ...
        (phi(2:end-1,2:end-1) + dt .* ...
         delta_phi(2:end-1,2:end-1) ...
         .* expr1) / expr2;

    %Reinitialization of phi
    if reIni>0 && mod(nIter, reIni)==0
        indGT = phi >= 0;
        indLT = phi < 0;

        phi=double(bwdist(indLT) - bwdist(indGT));

        %Normalization [-1 1]
        nor = min(abs(min(phi(:))), max(phi(:)));
        phi=phi/nor;
    end

    %Diference. This stopping criterium has the problem that phi can
    %change, but not the zero level set, that it really is what we are
    %looking for.
    dif = mean(sum( (phi(:) - phi_old(:)).^2 ));

    %Plot the level sets surface
    subplot(1,2,1)
        %The level set function
        surfc(phi);
        hold on
        %The zero level set over the surface
        contour(phi == 0);
        hold off
        title('Phi Function');

    %Plot the curve evolution over the image
    subplot(1,2,2)
        imagesc(I);
        colormap gray;
        hold on;
        contour(phi == 0);
        title('Image and zero level set of Phi')

        axis off;
        hold off
    drawnow;
    pause(.0001);
end