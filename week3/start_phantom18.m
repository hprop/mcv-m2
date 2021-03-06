%close all;
clearvars;
clc;

I=double(imread('phantom18.bmp'));
I=mean(I,3);
I=I-min(I(:));
I=I/max(I(:));

[ni, nj]=size(I);


%Lenght and area parameters
%Karim's parameters for phantom18: mu=0.2 mu=0.5
mu=1;  % Increase mu to improve the smoothness
nu=0;  % Avoid area penalty


%%Parameters
lambda1=1;
lambda2=1;

epHeaviside=1;
eta=1;
tol=0.1;
%dt=(10^-1)/mu;
dt=0.8;  % Increase dt to accelerate convergence
iterMax=200;
reIni=100;
[X, Y]=meshgrid(1:nj, 1:ni);


%%Initial phi
% This initialization allows a faster convergence for phantom 18
phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/4)).^2)+50);

%Normalization of the initial phi to [-1 1]
phi_0=phi_0-min(phi_0(:));
phi_0=2*phi_0/max(phi_0(:));
phi_0=phi_0-1;


%%Explicit Gradient Descent
seg=sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni );
