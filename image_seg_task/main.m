
image_dir = '../data/jpg_images/';
addpath(genpath('../utils'))
addpath(genpath('../clustering_algorithms/'));
addpath(genpath('../clustering_algorithms/wkk'));

% read image, change color image to brightness image, resize to 160x160
picture = '2';
nr = 120; nc = 120;
I = imread_ncut(strcat(image_dir,picture,'.jpg'),nr,nc);

[A,imageEdges] = ICgraph(I);

% make A matrix sparse
valeurMin = 1e-6;
A = sparsifyc(A,valeurMin);

% check for matrix symmetry
if max(max(abs(A-A'))) > 1e-10 %voir (-12) 
    %disp(max(max(abs(A-A'))));
    error('A not symmetric');
end

n = size(A,1);

% degrees and regularization
offset = 5e-1;
d = sum(abs(A),2);
dr = 0.5 * (d - sum(A,2));
d = d + offset * 2;
dr = dr + offset;
A = A + spdiags(dr,0,n,n);


% construct kernel
D = sum(A,2);
% K = sigmoid*D^(-1) + D^(-1)*A*D^(-1);
% K = D^(-1)*A*D^(-1);
% K = (1./D)*(1./D)'.*A;
% K = A;

% p = 1;
% k = 0;
% while p ~= 0
%   [R,p] = chol(K);
%   k = k + 1;
%   if p ~= 0
%     % Ahat failed the chol test. It must have been just a hair off,
%     % due to floating point trash, so it is simplest now just to
%     % tweak by adding a tiny multiple of an identity matrix.
%     
%     % TODO: find min eigen value more efficiently
%     % I tried the following, but got an error
%     % Warning: Matrix is close to singular or badly scaled, solve this in
%     % the future
%     %     mineig_eigs = eigs(Ahat,1,'SM');
%     % another issiue is eig on sparse matrix seems to be very slow, figure
%     % it out in the future
%     mineig = min(eig(full(K)));
%     K = K + (-mineig*k.^2 + eps(mineig))*speye(size(A));
%   end
% end


% K = nearestSPD(K);

assert(issymmetric(K) == true,'K is not symmetric');
if full(any(any(K<0)))
    disp('K has negative elements');
end

n = size(A,1);
k = 5;
n_repeat = 20;

% uniform sampling based clustering
s = floor(0.3*log(n)^4); % number to sample
n_iter = 100;
min_ncut = realmax;
Ncuts = zeros(1,n_repeat);
ps = zeros(n_repeat,n);
running_time = zeros(1,n_repeat);
% W = D;
W = ones(n,1);
% weights of samples for weighted kernel kmeans
% W_s = D(sample_index);
% % W_s = ones(s,1);
for i = 1:n_repeat
    tic;
    sample_index = randperm(n,s);
    K_s = K(sample_index,sample_index);

%     [R,p] = chol(K_s);
%     assert(p == 0,'K_s is not positive semi definite');

    init_method.name = 'random-balance';
    init_method.K = K_s;
    init_labels = init_label(s,k,init_method); % initial labels for sample points

    
    % [p_s,obj_s,Distance_s] = WeightedKernelKmeans(W_s,K_s,k,n_iter,init_labels);
    % P_s_one_hot = 1:k == p_s;
    % assert(any(sum(P_s_one_hot) == 0) == false, 'empty cluster');

    % test kernel kmeans++ seeding
    % Distance_wkk = zeros(n,k);
    % for i = 1:n
    %     Distance_wkk(i,:) = K(i,i) - 2*K(i,init_idxs) + K(sub2ind(size(K),init_idxs,init_idxs));
    % end
    % [~,p_wkk] = min(Distance_wkk,[],2);
    % P_wkk = 1:k == p_wkk;
    % Z = P_wkk*diag(diag(1./sqrt(P_wkk'*diag(D)*P_wkk)));
    % Nassoc_wkk = trace(Z'*A*Z);

    % use Mo chen's kk code to check wkk implementation
    [p_s,obj_s] = knKmeans(init_labels, K_s);
    P_s_one_hot = 1:k == p_s';
    assert(any(sum(P_s_one_hot) == 0) == false, 'empty cluster');

    % diffuse
    cluster_num = sum(P_s_one_hot,1);

    I_stretch_origin = find(P_s_one_hot);
    reduce_m = length(I_stretch_origin)*repelem(0:(k-1),cluster_num)';
    I_stretch = I_stretch_origin - reduce_m;
    I_stretch_sample = sample_index(I_stretch)';
    j_block = mat2cell(I_stretch_sample,cluster_num,1);

    % Distance of the rest points to the centers, it is not weighted, and we
    % don't need weights here

    
    Distance = kernel_distance(setdiff(1:n,sample_index),k,W,K,j_block); 
    [~,p] = min(Distance,[],2);
    p(sample_index) = p_s;
    P = 1:k == p;

    Z = P*diag(diag(1./sqrt(P'*diag(D)*P)));
    Ncut = k - trace(Z'*A*Z);
    assert(Ncut>=0,'ncut is negative');
    if Ncut < min_ncut
        min_ncut = Ncut;
        min_p = p;
    end
    Ncuts(i) = Ncut;
    ps(i,:) = p';
    running_time(i) = toc;
end

save(strcat(['uni-sample','_',picture,'_',num2str(nr),'_',num2str(nc),'.mat']),'Ncuts','ps','running_time','min_ncut','min_p');

% show segmentation of the picture
show_seg(I,1:k == min_p);

% kernel-kmc^2 seeding
m = 200; % chain length
% W_kmc2 = D;
W_kmc2 = ones(n,1);
min_ncut = realmax;
Ncuts = zeros(1,n_repeat);
ps = zeros(n_repeat,n);
running_time = zeros(1,n_repeat);
for j = 1:n_repeat
    tic;
    idx = WeightedKernelKMC2(W_kmc2,K,k,m);
    Distance = zeros(n,k);
    for i = 1:n
        Distance(i,:) = K(i,i) - 2*K(i,idx) + K(sub2ind(size(K),idx,idx));
    end
    [~,p] = min(W_kmc2.*Distance,[],2);
    P = 1:k == p;

    Z = P*diag(diag(1./sqrt(P'*diag(D)*P)));
    Ncut = k - trace(Z'*A*Z);
    assert(Ncut>=0,'ncut is negative');
    if Ncut < min_ncut
        min_ncut = Ncut;
        min_p = p;
    end
    Ncuts(j) = Ncut;
    ps(j,:) = p';
    running_time(j) = toc;
end
save(strcat(['kmc2','_',picture,'_',num2str(nr),'_',num2str(nc),'.mat']),'Ncuts','ps','running_time','min_ncut','min_p');

% show segmentation of the picture
show_seg(I,1:k == min_p);


% Double-kmc2-sampling
s2 = floor(0.5*s);
for j = 1:n_repeat
    tic;
    idx1 = WeightedKernelKMC2(W_kmc2,K,s2,m);
    sel_index = setdiff(1:n,idx1);
    idx2 = WeightedKernelKMC2(W_kmc2(sel_index),K(sel_index,sel_index),s2,m);
    idx2 = sel_index(idx2);
    
    Distance = zeros(s2,s2);
    for i = 1:s2
        Distance(i,:) = K(idx2(i),idx2(i)) - 2*K(idx2(i),idx1) + K(sub2ind(size(K),idx1,idx1));
    end
    [~,p] = min(Distance,[],2);
    P = 1:s2 == p;
    w = sum(P);
    w = w+1;
    
    init_method.name = 'random-balance';
    init_method.K = K(idx1,idx1);
    init_labels = init_label(s2,k,init_method);
    [p_s,obj_s,Distance_s] = WeightedKernelKmeans(w.',K(idx1,idx1),k,n_iter,init_labels);
    P_s_one_hot = 1:k == p_s;
    assert(any(sum(P_s_one_hot) == 0) == false, 'empty cluster');

    % diffuse
    cluster_num = sum(P_s_one_hot,1);

    I_stretch_origin = find(P_s_one_hot);
    reduce_m = length(I_stretch_origin)*repelem(0:(k-1),cluster_num)';
    I_stretch = I_stretch_origin - reduce_m;
    I_stretch_sample = idx1(I_stretch)';
    j_block = mat2cell(I_stretch_sample,cluster_num,1);

    % Distance of the rest points to the centers, it is not weighted, and we
    % don't need weights here
    
    Distance = kernel_distance(setdiff(1:n,idx1),k,W,K,j_block); 
    [~,p] = min(Distance,[],2);
    p(idx1) = p_s;
    P = 1:k == p;
    
    
    Z = P*diag(diag(1./sqrt(P'*diag(D)*P)));
    Ncut = k - trace(Z'*A*Z);
    assert(Ncut>=0,'ncut is negative');
    if Ncut < min_ncut
        min_ncut = Ncut;
        min_p = p;
    end
    Ncuts(j) = Ncut;
    ps(j,:) = p';
    running_time(j) = toc;
end

save(strcat(['Double-kmc2','_',picture,'_',num2str(nr),'_',num2str(nc),'.mat']),'Ncuts','ps','running_time','min_ncut','min_p');

% show segmentation of the picture
show_seg(I,1:k == min_p);
