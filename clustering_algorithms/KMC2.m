function [C,sample_index,dis_calculator] = KMC2(X,k,m,dis_calculator)
    % X: dxn
    
    n = size(X,2);
    first_id = 1+round(rand*(n-1));
    C = zeros(size(X,1),k);
    C(:,1) = X(:,first_id);
    
    sample_index = zeros(1,k);
    sample_index(1) = first_id;

    for i = 2:k

        cand_ind = randsample(1:n,m);
        D_ys = pdist2(X(:,cand_ind)',C(:,1:i-1)','squaredeuclidean');
        dis_calculator = dis_calculator + numel(D_ys);
        d_ys = min(D_ys,[],2);
        
        for j = 1:m

            y_id = cand_ind(j);
            d_y = d_ys(j);
            if j==1 || d_y/d_x > rand
                x_id = y_id;
                d_x = d_y;
            end
        end
        C(:,i) = X(:,x_id);
        sample_index(i) = x_id;
    end