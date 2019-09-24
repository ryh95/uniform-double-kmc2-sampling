function [centroid,dis_calculator] = DoubleKMC2(X,s,k,m,dis_calculator)
    % X: dxn
    % s: sample number
    % m: chain length
    % k: number of clusters
    
%     [C1,sample_index1] = KMC2(X,s,m);
    [C1,sample_index1,dis_calculator] = KMC2(X,s,m,dis_calculator);
    
    % remove the sampled points and rerun KMC2 to compute weights
    sel_index = setdiff(1:size(X,2),sample_index1);
    X_rest = X(:,sel_index);
    
%     [C2,~] = KMC2(X_rest,s,m);
    [C2,sample_index2,dis_calculator] = KMC2(X_rest,s,m,dis_calculator);
    
    % compute weights
    D = pdist2(C1',C2'); % C1xC2
    dis_calculator = dis_calculator + numel(D);
    [~,p] = min(D);
    I = zeros(size(D)); % I: C2xC1 -> sxs
    I(sub2ind(size(D),1:s,p)) = 1; % here s is the size of C2
    w = sum(I); % C1 size
    w = w + 1;
    
    % run weighted k-means
    options.weight = w';
    options.careful = true;
    [label, centroid, dis,dis_calculator] = fkmeans(C1', k,dis_calculator,options);
    
