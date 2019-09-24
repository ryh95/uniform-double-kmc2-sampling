function [idx,dis_calculator] = KernelKMC2(K,k,m,dis_calculator)
% W: weights for points, col vector

    n = size(K,1);
    idx = zeros(1,k);

    idx(1) = randsample(1:n,1);
    
    for i = 2:k
        
        cand_ind = randsample(1:n,m);
        D_ys = zeros(m,i-1);
        for l = 1:m
            x = cand_ind(l);
            D_ys(l,:) = K(x,x) - 2*K(x,idx(1:i-1)) + K(sub2ind(size(K),idx(1:i-1),idx(1:i-1)));
        end
        d_ys = min(D_ys,[],2);
        dis_calculator = dis_calculator + numel(D_ys);
        
        for j = 1:m
            y_id = cand_ind(j);
            d_y = d_ys(j);
            if j==1 || d_y/d_x > rand
                x_id = y_id;
                d_x = d_y;
            end
        end
        idx(i) = x_id;
    end