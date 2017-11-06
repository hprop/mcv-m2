%close all;
clearvars;
clc

I=double(imread('circles.png'));
I=mean(I,3);
I=I-min(I(:));
I=I/max(I(:));

[ni, nj]=size(I);

%Lenght and area parameters
mu=1;
nu=0;


%%Parameters
lambda1=1;
lambda2=1;

epHeaviside=1;
eta=1
tol=0.1;
dt=(10^-1)/mu;
iterMax=130;
reIni=100;
[X, Y]=meshgrid(1:nj, 1:ni);


%%Initial phi
phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/2)).^2)+50);

phi_0=phi_0-min(phi_0(:));
phi_0=2*phi_0/max(phi_0(:));
phi_0=phi_0-1;


%%Explicit Gradient Descent
seg=sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni );
