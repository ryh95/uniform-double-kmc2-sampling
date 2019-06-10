function [label, energy] = knKmeans(init, K)
% Perform kernel kmeans clustering.
% Input:
%   K: n x n kernel matrix
%   init: initial label (1xn)
% Output:
%   label: 1 x n clustering result label
%   energy: optimization target value
%   model: trained model structure
% Reference: Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Mo Chen (sth4nth@gmail.com).
n = size(K,2);
label = init;
last = zeros(1,n);
while any(label ~= last)
    [~,~,last(:)] = unique(label);   % remove empty clusters
    E = sparse(last,1:n,1);
    E = E./sum(E,2);
    T = E*K;
    [val, label] = max(T-dot(T,E,2)/2,[],1);
end
energy = trace(K)-2*sum(val); 