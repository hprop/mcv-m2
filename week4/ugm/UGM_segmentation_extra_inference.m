clear all;
close all;
clc;

im_name='3_12_s.bmp';

% TODO: Update library path
% Add  library paths

basedir='UGM';
maxflow='maxflow';
addpath(genpath(basedir));
addpath(genpath(maxflow));

%Set model parameters
%cluster color

%Pair-wise parameters
smooth_term=[4 4]; % Potts Model

%Load images
im = imread(im_name);


NumFils = size(im,1);
NumCols = size(im,2);

%Convert to LAB colors space
% TODO: Uncomment if you want to work in the LAB space
%
im = RGB2Lab(im);



%Preparing data for GMM fiting
%
% TODO: define the unary energy term: data_term
% nodePot = P( color at pixel 'x' | Cluster color 'c' )
K = 2;  % total of clusters

im = double(im);
x = reshape(im, [size(im,1) * size(im,2),size(im,3)]);
gmm_color = gmdistribution.fit(x,K);
mu_color = gmm_color.mu;

[nodePot, data_term]=gmm_color.posterior(x);

data_term = -nodePot;

%Building 4-grid
%Build UGM Model for 4-connected segmentation
disp('create UGM model');

% Create UGM data
[edgePot,edgeStruct] = CreateGridUGMModel(NumFils, NumCols, K ,smooth_term);

if ~isempty(edgePot)

    % color clustering
    [~,c] = min(reshape(data_term,[NumFils*NumCols K]),[],2);
    im_c= reshape(mu_color(c,:),size(im));
    
    % Call different UGM inference algorithms
    display('Loopy Belief Propagation'); tic;
    [nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);toc; 
    [~, im_lbp] = max(nodeBelLBP,[],2); 
    im_lbp = reshape(mu_color(im_lbp, :), size(im));
    
    % Max-sum
    display('Max-sum'); tic;
    decodeLBP = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
    im_bp= reshape(mu_color(decodeLBP,:),size(im));
    toc;
    
    %Graph Cuts
    display('Graph Cuts'); tic;
    decodeGC = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);
    im_gc= reshape(mu_color(decodeGC,:),size(im));
    toc;
    
    %Linear Programming
    %display('Linear Programming'); tic;
    %decodeLP = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
    %im_lp= reshape(mu_color(decodeLP,:),size(im));
    %toc;
    
    %Iterated Conditional Modes with restarts
    display('Linear Programming'); tic;
    nRestarts = 500;
    decodeICM = UGM_Decode_ICMrestart(nodePot,edgePot,edgeStruct,nRestarts);
    im_icm= reshape(mu_color(decodeICM,:),size(im));
    toc;
    
    % TODO: apply other inference algorithms and compare their performance
    %
    % - Graph Cut
    % - Linear Programing Relaxation
    
    figure
    subplot(2,3,1),imshow(Lab2RGB(im));xlabel('Original');
    subplot(2,3,2),imshow(Lab2RGB(im_c),[]);xlabel('Clustering without GM');
    subplot(2,3,3),imshow(Lab2RGB(im_bp),[]);xlabel('Max-Sum');
    subplot(2,3,4),imshow(Lab2RGB(im_lbp),[]);xlabel('Loopy Belief Propagation');
    subplot(2,3,5),imshow(Lab2RGB(im_gc),[]);xlabel('Graph Cuts');
    subplot(2,3,6),imshow(Lab2RGB(im_icm),[]);xlabel('Iterated Conditional Modes');

else
   
    error('You have to implement the CreateGridUGMModel.m function');

end