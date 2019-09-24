function [label, energy, Distance, dis_calculator] = knKmeans(init, K, t_max, dis_calculator)
% Perform kernel kmeans clustering.
% Input:
%   K: n x n kernel matrix
%   init: initial label (1xn)
% Output:
%   label: 1 x n clustering result label
%   energy: optimization target value, label->energy 
%   model: trained model structure
% Reference: Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Mo Chen (sth4nth@gmail.com).
n = size(K,2);
label = init;
last = zeros(1,n);
t = 0;
while any(label ~= last) && t<t_max
    [~,~,last(:)] = unique(label);   % remove empty clusters
    E = sparse(last,1:n,1);
    E = E./sum(E,2);
    T = E*K;
    % distance evaluations don't mean time complexity here
    dis_calculator = dis_calculator + numel(T);
    [val, label] = max(T-dot(T,E,2)/2,[],1);
    t = t+1;
end
E = sparse(label,1:n,1);
E = E./sum(E,2);
T = E*K;
Distance = diag(K) - 2*(T-dot(T,E,2)/2)';
energy = trace(K)-2*sum(val);