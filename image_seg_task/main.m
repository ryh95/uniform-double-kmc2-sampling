
image_dir = '../data/jpg_images/';
addpath(genpath('../utils'))
addpath(genpath('../clustering_algorithms/'));
addpath(genpath('../clustering_algorithms/wkk'));

% read image, change color image to brightness image, resize to 160x160
% kitten 60 60 / 3 30 30 / 2 120 120
% pictures = ["3","kitten","2"];
pictures = ["kitten"];
algorithms = ["uni-sample","Double-kmc2","kmc2"];
% algorithms = ["uni-sample"];

for picture = pictures
    
    for algorithm = algorithms
        
        if strcmp(picture,'3')
            nr = 30; nc = 30;
        elseif strcmp(picture,'kitten')
            nr = 60; nc = 60;
        elseif strcmp(picture,'2')
            nr = 120; nc = 120;
        end
    
        %% prepare image and kernel

        I = imread_ncut(strcat(image_dir,char(picture),'.jpg'),nr,nc);

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
        
        if strcmp(picture,"2")
            % load pre-computed kernel to save time
            K = load('../data/image_kernels/K_2_120_120.mat');
            K = K.K;
        else
            K = A;
            K = nearestSPD(K);
        end
        
        assert(issymmetric(K) == true,'K is not symmetric');


        n = size(A,1);
        k = 5;
        n_repeat = 30;
    
        ps = zeros(n_repeat,n);
        running_time = zeros(1,n_repeat);
        obj = zeros(1,n_repeat);
        
        
        %% uniform sampling based clustering
        if strcmp(algorithm,'uni-sample')

            s = floor(0.4*log(n)^4); % number to sample
            Ncuts = zeros(1,n_repeat);
            min_ncut = inf;
            init_method = struct();
            n_iter = 10;
            
            for i = 1:n_repeat
                
                dis_calculator = 0;
                sample_index = randperm(n,s);
                K_s = K(sample_index,sample_index);

                init_method.name = 'knkmeans++';
                init_method.K = K_s;
                % initial labels for sample points
                [init_labels,dis_calculator] = init_label(s,k,init_method,dis_calculator);


                % use Mo chen's kk code to check wkk implementation
                [p_s,obj_s,Distance_s,dis_calculator] = knKmeans(init_labels, K_s, n_iter, dis_calculator);
                
                P_s_one_hot = 1:k == p_s';
                assert(all(sum(P_s_one_hot) ~= 0), 'empty cluster');

                % diffuse
                cluster_num = sum(P_s_one_hot,1);

                I_stretch_origin = find(P_s_one_hot);
                reduce_m = length(I_stretch_origin)*repelem(0:(k-1),cluster_num)';
                I_stretch = I_stretch_origin - reduce_m;
                I_stretch_sample = sample_index(I_stretch)';
                j_block = mat2cell(I_stretch_sample,cluster_num,1);

                % Distance of the rest points to the centers
                % diffuse time is not included
                
                Distance = kernel_distance(1:n,k,ones(n,1),K,j_block);
                assert(all(all(Distance>=0)),'Distance has negative value');
                % the label p is at the iteration of t_max + 1
                % can ignore p if ncut is not needed
                [d,p] = min(Distance,[],2);
                
                P = 1:k == p;
                Z = P*diag(diag(1./sqrt(P'*diag(D)*P)));
                Ncut = k - trace(Z'*A*Z);
                assert(Ncut>=0,'ncut is negative');
                if Ncut < min_ncut
                    min_ncut = Ncut;
                    min_p = p;
                end
                Ncuts(i) = Ncut;
                obj(i) = sum(d);
                ps(i,:) = p';
                running_time(i) = dis_calculator;
            end

        end
        
        %% kernel-kmc^2 seeding
        if strcmp(algorithm,'kmc2')
            
            m = 200; % chain length
            Ncuts = zeros(1,n_repeat);
            min_ncut = inf;
            
            for j = 1:n_repeat
            %     tic;
                dis_calculator = 0;
                [idx,dis_calculator] = KernelKMC2(K,k,m,dis_calculator);
                
                % diffuse
                Distance = zeros(n,k);
                for i = 1:n
                    Distance(i,:) = K(i,i) - 2*K(i,idx) + K(sub2ind(size(K),idx,idx));
                end
                assert(all(all(Distance>=0)),'Distance has negative value');
                [d,p] = min(Distance,[],2);

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
                obj(j) = sum(d);
                running_time(j) = dis_calculator;
            end
            
        end
        
        %% kernel-Double-kmc2-sampling
        if strcmp(algorithm,'Double-kmc2')   
            
            s2 = floor(0.25*log(n)^2);
            m = 200;
            n_iter = 100;
            Ncuts = zeros(1,n_repeat);
            min_ncut = inf;
            init_method = struct();
            
            for j = 1:n_repeat
%                 tic;
                dis_calculator = 0;
                
                [idx1,dis_calculator] = KernelKMC2(K,s2,m,dis_calculator);
                sel_index = setdiff(1:n,idx1);
                [idx2,dis_calculator] = KernelKMC2(K(sel_index,sel_index),s2,m,dis_calculator);
                idx2 = sel_index(idx2);

                Distance = zeros(s2,s2);
                dis_calculator = dis_calculator + numel(Distance);
                for i = 1:s2
                    Distance(i,:) = K(idx2(i),idx2(i)) - 2*K(idx2(i),idx1) + K(sub2ind(size(K),idx1,idx1));
                end
                assert(all(all(Distance>=0)),'Distance has negative value');
                [~,p] = min(Distance,[],2);
                P = 1:s2 == p;
                w = sum(P);
                w = w+1;

                % weighted kernel kmeans++
                init_method.name = 'knkmeans++';
                init_method.K = K(idx1,idx1);
                init_method.W = w;
                [init_labels,dis_calculator] = init_label(s2,k,init_method,dis_calculator);
                
                
                [p_s,dis_calculator] = WeightedKernelKmeans(w.',K(idx1,idx1),k,n_iter,init_labels,dis_calculator);
                P_s_one_hot = 1:k == p_s;
                assert(all(sum(P_s_one_hot) ~= 0), 'empty cluster');

                % diffuse
                cluster_num = sum(P_s_one_hot,1);

                I_stretch_origin = find(P_s_one_hot);
                reduce_m = length(I_stretch_origin)*repelem(0:(k-1),cluster_num)';
                I_stretch = I_stretch_origin - reduce_m;
                I_stretch_sample = idx1(I_stretch)';
                j_block = mat2cell(I_stretch_sample,cluster_num,1);

                % Distance of the rest points to the centers
                % diffuse time is not included

                Distance = kernel_distance(1:n,k,ones(n,1),K,j_block);
                assert(all(all(Distance>=0)),'Distance has negative value');
                [d,p] = min(Distance,[],2);
                

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
                obj(j) = sum(d);
                running_time(j) = dis_calculator;
            end
            
        end
        
        
        save(join([algorithm,picture,num2str(nr),num2str(nc),'.mat'],"_"),'obj','ps','running_time','Ncuts');

        % show segmentation of the picture
        show_seg(I,1:k == min_p);
        
    end
end

