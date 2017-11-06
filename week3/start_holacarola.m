%close all;
clearvars;
clc

I=double(imread('Image_to_Restore.png'));
I=mean(I,3);
I=I-min(I(:));
I=I/max(I(:));

[ni, nj]=size(I);


%Lenght and area parameters
%hola carola mu=1
mu=1;
nu=0;


%%Parameters
lambda1=10^-3; %Hola carola problem
lambda2=10^-3; %Hola carola problem

epHeaviside=1;
%eta=0.01;
eta=1
tol=0.1;
%dt=(10^-2)/mu;
dt=(10^-1)/mu;
iterMax=50;
%reIni=0; %Try both of them
%reIni=500;
reIni=100;
[X, Y]=meshgrid(1:nj, 1:ni);

%%Initial phi

phi_0=I; %For the Hola carola problem

phi_0=phi_0-min(phi_0(:));
phi_0=2*phi_0/max(phi_0(:));
phi_0=phi_0-1;



%%Explicit Gradient Descent
seg=sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni );
