clearvars;
src = double(imread('snail.png'));
dst = double(imread('mona-lisa.jpg')); % flipped girl, because of the eyes
[ni,nj, nChannels]=size(dst);

param.hi=1;
param.hj=1;

mask_src=logical(zeros(size(src(:,:,1))));
mask_src(74:95, 60:128) = 1;
mask_dst=logical(zeros(size(dst(:,:,1))));
mask_dst(93:114, 78:146) = 1;

%masks to exchange: Eyes
% mask_src=logical(imread('mask_src_eyes.png'));
% mask_dst=logical(imread('mask_dst_eyes.png'));

for nC = 1: nChannels
    
    %TO DO: COMPLETE the ??
    drivingGrad_i = sol_DiFwd(sol_DiFwd(src(:,:,nC),param.hi));
    drivingGrad_j = sol_DjFwd(sol_DjFwd(src(:,:,nC),param.hj));

    driving_on_src = drivingGrad_i + drivingGrad_j;
    
    driving_on_dst = zeros(size(src(:,:,1)));   
    driving_on_dst(mask_dst(:)) = driving_on_src(mask_src(:));
    
    param.driving = driving_on_dst;

    dst1(:,:,nC) = sol_Poisson_Equation_Axb(dst(:,:,nC), mask_dst,  param);
end

imshow(dst1/256)