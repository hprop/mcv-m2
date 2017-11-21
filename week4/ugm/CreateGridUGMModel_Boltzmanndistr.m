function [edgePot,edgeStruct]=CreateGridUGMModel(NumFils, NumCols, K, lambda, X)
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
   diag_vector = exp(lambda(1) + lambda(2)*1/(1+abs(X(n1)-X(n2))))*ones(K,1);
   edgePot(:,:,e) = ones(K) + diag(diag_vector) - diag(ones(K,1));
end

toc;