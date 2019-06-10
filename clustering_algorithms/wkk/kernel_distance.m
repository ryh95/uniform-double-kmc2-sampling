function Distance = kernel_distance(row_num,k,W_diag,K,I)
% i: index that to compute distance
% c: cluster index
% row_num: row id to compute distance, vector
% k: number of clusters
% W_diag: diagonal vector of weight matrix, col vector
% K: kernel matrix
% I: partition cell array [k 1], for each cell matrix, index should not
% exceed size of W
% ---------------------------
% d: real positive number

% compute the denominator of second term 
    second_denominator = zeros(1,k);
    for i = 1:k
        second_denominator(i) = sum(W_diag(I{i,1}));
    end

    % compute the third term
    third_term = zeros(1,k);
    for i = 1:k
        j_l = I{i,1};
        third_numerator = sum(sum(W_diag(j_l)*W_diag(j_l)'.*K(j_l,j_l)));
        third_term(i) = third_numerator/(second_denominator(i)^2);
    end
    

    % compute distances
    Distance = zeros(size(W_diag,1),k);
    
    [j_matrix,mask] = padcat(I{:});
    cluster_num = cellfun(@length,I);
    
    if max(cluster_num) == 1
        j_matrix = j_matrix';
        mask = mask';
    end
    if any(cluster_num == 0)
        error('cluster num should not be 0');
    end
    
    
    j_matrix(~mask) = 1;
    assert(size(j_matrix,1) == max(cluster_num),'the first dim of j_matrix should be max cluster num');

    W_diag_mask = W_diag(j_matrix);
    W_diag_mask(~mask) = 0;
    
    if size(W_diag_mask,1) ~= max(cluster_num)
        W_diag_mask = W_diag_mask';
    end
    assert(size(W_diag_mask,1) == max(cluster_num),'the first dim of W_diag_mask should be max cluster num');
    
    % todo: speed up the following
    for i = row_num
        
        second_numerator = sum(W_diag_mask'.*reshape(K(i,j_matrix(:)),max(cluster_num),k)',2);
        
        Distance(i,:) = K(i,i) - 2*second_numerator'./second_denominator + third_term;

    end