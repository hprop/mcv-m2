function [edgePot,edgeStruct]=CreateGridUGMModel(NumFils, NumCols, K, lambda)
%
%
% NumFils, NumCols: image dimension
% K: number of states
% lambda: smoothing factor



tic

nNodes = NumFils*NumCols;
nEdges = ((NumFils-1)*NumCols)+((NumCols-1)*NumFils);

adj_each = [];
adj_neigh = [];

for i = 1:nNodes
    if i > NumFils
        adj_each = [adj_each, i];
        adj_neigh = [adj_neigh, i-NumFils];
    end    
    if i <= (nNodes - NumFils)
        adj_each = [adj_each, i];
        adj_neigh = [adj_neigh, i+NumFils];
    end
    if mod(i-1, NumFils)>0
        adj_each = [adj_each, i];
        adj_neigh = [adj_neigh, i-1];
    end
    if mod(i, NumFils)>0
        adj_each = [adj_each, i];
        adj_neigh = [adj_neigh, i+1];
    end    
end
adj = sparse(adj_each, adj_neigh, ones([length(adj_each),1]));

edgeStruct = UGM_makeEdgeStruct(adj,K);

edgePot = zeros(K,K,nEdges);
for e = 1:edgeStruct.nEdges
   n1 = edgeStruct.edgeEnds(e,1);
   n2 = edgeStruct.edgeEnds(e,2);
   pot_same = lambda;
   edgePot(:,:,e) = ones(K) + sparse([1:K],[1:K], (pot_same-1).*ones(K,1));
end

toc;