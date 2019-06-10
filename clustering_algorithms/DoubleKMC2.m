function [centroid] = DoubleKMC2(X,s,k,m)
    % X: dxn
    % s: sample number
    % m: chain length
    % k: number of clusters
    
%     [C1,sample_index1] = KMC2(X,s,m);
    n = size(X,2);
    X_np = py.numpy.array(transpose(X(:)));
    rt_centers_indices = py.kmc2caller.call_kmc2(X_np,int32(n),int32(s),int32(m),false);
    C1 = reshape(double(py.array.array('d',py.numpy.nditer(rt_centers_indices{1}))),[],s);
    sample_index1 = reshape(double(py.array.array('d',py.numpy.nditer(rt_centers_indices{2}))),[],s);
    
    % remove the sampled points and rerun KMC2 to compute weights
    sel_index = setdiff(1:size(X,2),sample_index1);
    X_rest = X(:,sel_index);
    
%     [C2,~] = KMC2(X_rest,s,m);
    
    n = size(X_rest,2);
    X_np = py.numpy.array(transpose(X_rest(:)));
    rt_centers_indices = py.kmc2caller.call_kmc2(X_np,int32(n),int32(s),int32(m),false);
    C2 = reshape(double(py.array.array('d',py.numpy.nditer(rt_centers_indices{1}))),[],s);
    
    % compute weights
    D = pdist2(C1.',C2.');
    [~,p] = min(D);
    I = zeros(size(D));
    I(sub2ind(size(D),1:s,p)) = 1;
    w = sum(I);
    w = w + 1;
    
    % run weighted k-means
    options.weight = w.';
    options.careful = true;
    [label, centroid, dis] = fkmeans(C1.', k, options);
    
