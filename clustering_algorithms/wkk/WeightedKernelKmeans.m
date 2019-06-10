function varargout = WeightedKernelKmeans(W_diag,K,k,t_max,init_labels)
% W_diag: column vector, weights of each point
% K: kernel matrix
% k: num of clusters
% t_max: max iteration number
% init_labels: initial labels for each point, [1,size(W_diag,1)]
% ------------------------------------------------
% output:
% p: label vector, [size(K,1),1]
% obj: sum of kernelized squared distances
% D: all kernelized squared distances matrix, [size(K,1),size(K,1)],
% min(D,[],2) = p

% todo: generate a better start seed
% todo: tackle with empty cluster problem

% 1. init indicators
num_points = size(W_diag,1);

I = 1:k == init_labels';

assert(all(sum(I)) == true,'init cluster has empty cluster');
assert(all(sum(I,2) == ones(num_points,1)) == true,'at least one point has more than one cluster'); 
nOutputs = nargout;
varargout = cell(1,nOutputs);

% 2. compute distances
t=0;
obj_last = realmax;
while t<t_max
    
    % compute distance with kernel_distance
    cluster_num = sum(I,1);
    % I [s k] -> I [s*k 1]
    I_stretch = find(I);
    reduce_m = num_points*repelem(0:(k-1),cluster_num)';
    I_stretch = I_stretch - reduce_m;
    j_block = mat2cell(I_stretch,cluster_num,1);
    D = kernel_distance(1:num_points,k,W_diag,K,j_block);
    
    % check distance positivity
    if any(any(D<0)) == false
        disp('Distance have negative elements!');
    end
%     assert(any(any(D<0)) == false,'Distance have negative elements!');
    
    % update indicators
    [v,p] = min(W_diag.*D,[],2);
    % todo: make I sparse to save memory
    I = 1:k == p;
    
    obj = sum(v);
    
    if obj_last - obj <= eps
        break
    end
    
    obj_last = obj;
    t = t+1;
end

varargout{1} = p;
if nOutputs > 1
    varargout{2} = obj;
end
if nOutputs > 2
    varargout{3} = D;
end