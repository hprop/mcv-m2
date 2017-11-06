%close all;
clearvars;
clc

I=double(imread('phantom19.bmp'));
I=mean(I,3);
I=I-min(I(:));
I=I/max(I(:));

[ni, nj]=size(I);


%Lenght and area parameters
mu=1;
nu=-0.01;


%%Parameters
lambda1=2;
lambda2=1;

epHeaviside=1;
eta=1
%tol=0.000001;
tol=0.001;
dt=1.5;
iterMax=3000;
reIni=100;
[X, Y]=meshgrid(1:nj, 1:ni);


%%Initial phi
% This initialization allows a faster convergence for phantom 18
phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/2)).^2)+50);
%phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/4)).^2)+50);

%Normalization of the initial phi to [-1 1]
phi_0=phi_0-min(phi_0(:));
phi_0=2*phi_0/max(phi_0(:));
phi_0=phi_0-1;


%%Explicit Gradient Descent
seg=sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni );
