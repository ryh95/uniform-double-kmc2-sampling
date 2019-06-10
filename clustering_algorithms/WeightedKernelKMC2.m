function [idx] = WeightedKernelKMC2(W,K,k,m)
% W: weights for points, col vector

    n = size(K,1);
    idx = zeros(1,k);

    idx(1) = randsample(1:n,1);
    
    for i = 2:k
        x = randsample(1:n,1);
        d_x = W(x)*min(K(x,x) - 2*K(x,idx(1:i-1)) + K(sub2ind(size(K),idx(1:i-1),idx(1:i-1))));

        for j = 2:m
            y = randsample(1:n,1);
            d_y = W(y)*min(K(y,y) - 2*K(y,idx(1:i-1)) + K(sub2ind(size(K),idx(1:i-1),idx(1:i-1))));
            if d_y/d_x > rand
                x = y;
                d_x = d_y;
            end
        end
        idx(i) = x;
    end