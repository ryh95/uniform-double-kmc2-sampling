function [label,dis_calculator] = init_label(n,k,method,dis_calculator)
    if strcmp(method.name,'random-balance')

        label = zeros(1,n);
        cluster_num_points = floor(n/k);
        chose_points = [];
        for p = 1:k
            cluster_points = datasample(setdiff(1:n,chose_points),cluster_num_points,'Replace',false);
            label(cluster_points) = p;
            chose_points = [chose_points cluster_points];
        end
        rest_points = setdiff(1:n,chose_points);
        label(rest_points) = randi(k,1,length(rest_points));

    elseif strcmp(method.name,'random')
        
        label = ceil(k*rand(1,n));
        
    elseif strcmp(method.name,'knkmeans++')
        
        % The kernel k-means++ initialization.
        % can be weighted kernel k-means++ initialization if weights
        % specified
        C = zeros(1,k);
        C(1) = datasample(1:n,1);
        L = ones(1,n);
        K = method.K;
        W = ones(1,n);
        if isfield(method,'W')
            W = method.W;
        end
        for i = 2:k
            % D is the distance that a point to its nearest center 
            D = W.*(diag(K)' -2 * K(sub2ind(size(K),1:n,C(L))) + K(sub2ind(size(K),C(L),C(L))));
            dis_calculator = dis_calculator + numel(D);
            assert(all(D>=0),'negative distance');
            C(i) = datasample(1:n,1,'Weights',full(D));
            % todo: optimize the following
            [~,L] = max(bsxfun(@minus,2*K(1:n,C(1:i))',K(sub2ind(size(K),C(1:i),C(1:i)))'));
        end
        label = L;
    end