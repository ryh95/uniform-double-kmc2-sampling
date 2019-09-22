%% do not use this function, it's too slow
function [C,sample_index] = KMC2_2(X,k,m)
    % X: dxn
    n = size(X,2);
    first_id = 1+round(rand*(n-1));
    C = zeros(size(X,1),k);
    C(:,1) = X(:,first_id);
    
    sample_index = zeros(1,k);
    sample_index(1) = first_id;
    
    for i = 2:k
        x_id = randsample(1:n,1);
        x = X(:,x_id);
        
        D_x = x - C(:,1:i-1);
        d_x = min(dot(D_x,D_x,1));

        for j = 2:m
            y_id = randsample(1:n,1);
            y = X(:,y_id);
            D_y = y - C(:,1:i-1);
            d_y = min(dot(D_y,D_y,1));
            if d_y/d_x > rand
                x = y;
                x_id = y_id;
                d_x = d_y;
            end
        end
        C(:,i) = x;
        sample_index(i) = x_id;
    end