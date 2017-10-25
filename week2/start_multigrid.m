clearvars;
dst = double(imread('lena.png'));
src = double(imread('girl.png')); % flipped girl, because of the eyes
[ni,nj, nChannels]=size(dst);

param.hi=1;
param.hj=1;
param.iterations = 4;

%masks to exchange: Eyes
mask_src=logical(imread('mask_src_eyes.png'));
mask_dst=logical(imread('mask_dst_eyes.png'));

for nC = 1: nChannels

    %TO DO: COMPLETE the ??
    drivingGrad_i = sol_DiBwd(sol_DiFwd(src(:,:,nC),param.hi));
    drivingGrad_j = sol_DjBwd(sol_DjFwd(src(:,:,nC),param.hj));

    driving_on_src = drivingGrad_i + drivingGrad_j;

    driving_on_dst = zeros(size(src(:,:,1)));
    driving_on_dst(mask_dst(:)) = driving_on_src(mask_src(:));

    param.driving = driving_on_dst;

    dst1(:,:,nC) = sol_Poisson_Equation_Multigrid(dst(:,:,nC), mask_dst,  param);
end

%Mouth
%masks to exchange: Mouth
mask_src=logical(imread('mask_src_mouth.png'));
mask_dst=logical(imread('mask_dst_mouth.png'));
for nC = 1: nChannels

    %TO DO: COMPLETE the ??
    drivingGrad_i = sol_DiBwd(sol_DiFwd(src(:,:,nC),param.hi));
    drivingGrad_j = sol_DjBwd(sol_DjFwd(src(:,:,nC),param.hj));

    driving_on_src = drivingGrad_i + drivingGrad_j;

    driving_on_dst = zeros(size(src(:,:,1)));
    driving_on_dst(mask_dst(:)) = driving_on_src(mask_src(:));

    param.driving = driving_on_dst;

    dst1(:,:,nC) = sol_Poisson_Equation_Multigrid(dst1(:,:,nC), mask_dst,  param);
end


imshow(dst1/256)