%close all;
clearvars;
clc

I=double(imread('noisedCircles.tif'));
I=mean(I,3);
I=I-min(I(:));
I=I/max(I(:));

[ni, nj]=size(I);


%Lenght and area parameters
%noisedCircles.tif mu=0.1
mu=0.5;  % mu=0.1 produces shape irregularities (eg. holes inside the
         % circles). We increase the parameter to improve the
         % smoothness
nu=0;


%%Parameters
lambda1=1;
lambda2=1;

epHeaviside=1;
eta=1
tol=0.001;
dt=(10^-1)/mu;
iterMax=1150  % it takes about 1120 iterations with mu=0.5, nu=0
%reIni=0; %Try both of them
%reIni=500;
reIni=100;
[X, Y]=meshgrid(1:nj, 1:ni);


%%Initial phi
phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/2)).^2)+50);

phi_0=phi_0-min(phi_0(:));
phi_0=2*phi_0/max(phi_0(:));
phi_0=phi_0-1;


%%Explicit Gradient Descent
seg=sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni );
