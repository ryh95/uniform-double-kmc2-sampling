addpath(genpath('../clustering_algorithms/wkk'));
addpath(genpath('../clustering_algorithms/SNC'));
addpath(genpath('../utils'));
addpath(genpath('../data'));

% data_names = ["MnistData_05_uni","MnistData_10_uni",...
%     "isolet5","USPSdata_uni","letter-recognition"];
data_names = ["USPSdata_uni","letter-recognition"];
for data_name = data_names
    load(strcat(data_name,".mat"));
    n = size(X,1);
    if strcmp(data_name,'segment_uni')
        k = 7;
    elseif strcmp(data_name,'MnistData_05_uni')
        k = 10;
    elseif strcmp(data_name,'MnistData_10_uni')
        k = 10;
    elseif strcmp(data_name,'isolet5')
        k = 26;
    elseif strcmp(data_name,'USPSdata_uni')
        k = 10;
    elseif strcmp(data_name,'letter-recognition')
        k = 26;
    end
    
    
    % create the graph
    m = ceil(0.2*n);
    B = constructAnchorDistance_PKN(X',m, k);
    P=B*diag(sum(B,1))^-0.5;
    A = P*P';
    d = ones(n,1);

    algorithms = ["sample_wkk","wkk"];
    n_repeat = 20;
    for algorithm = algorithms
        if strcmp(algorithm,'sample_wkk')
            % to sampled wkk problem

            Ncuts = zeros(1,n_repeat);
            running_time = zeros(1,n_repeat);
            for j = 1:n_repeat
                tic;
                s = ceil(0.2*n);
                s_id = datasample(1:n,s,'Replace',false);
                % K = (1./d)*(1./d)'.*A;
                K = A;
                w = d;
                K_s = K(s_id,s_id);
                w_s = w(s_id);


                init_method.name = 'knkmeans++';
                init_method.K = K_s;
                init_method.W = w_s';
                dis_calculator = 0;
                [init_labels,dis_calculator] = init_label(s,k,init_method,dis_calculator);

                n_iter = 30;
                [p_s,dis_calculator] = WeightedKernelKmeans(w_s,K_s,k,n_iter,init_labels,dis_calculator);
                P_s_one_hot = 1:k == p_s;
                assert(all(sum(P_s_one_hot) ~= 0), 'empty cluster');

                % diffuse
                cluster_num = sum(P_s_one_hot,1);

                I_stretch_origin = find(P_s_one_hot);
                reduce_m = length(I_stretch_origin)*repelem(0:(k-1),cluster_num)';
                I_stretch = I_stretch_origin - reduce_m;
                I_stretch_sample = s_id(I_stretch)';
                j_block = mat2cell(I_stretch_sample,cluster_num,1);

                % Distance of the rest points to the centers
                Distance = kernel_distance(1:n,k,ones(n,1),K,j_block);
                assert(all(all(Distance>=0)),'Distance has negative value');
                [d_min,p] = min(Distance,[],2);
                P = 1:k == p;
                time_elapsed = toc;

                assert(all(sum(P) ~= 0), 'empty cluster');
                Z = P*diag(diag(1./sqrt(P'*diag(d)*P)));
                Ncut = k - trace(Z'*A*Z);
                assert(Ncut>=0,'ncut is negative');

                Ncuts(j) = Ncut;
                running_time(j) = time_elapsed;
                % 
            end
        end
        if strcmp(algorithm,'wkk')
            for j = 1:n_repeat
                tic;
                K = A;
                w = d;

                init_method.name = 'knkmeans++';
                init_method.K = K;
                init_method.W = w';
                dis_calculator = 0;
                [init_labels,dis_calculator] = init_label(n,k,init_method,dis_calculator);

                n_iter = 30;
                [p_s,dis_calculator] = WeightedKernelKmeans(w,K,k,n_iter,init_labels,dis_calculator);
                P = 1:k == p_s;
                time_elapsed = toc;

                assert(all(sum(P) ~= 0), 'empty cluster');
                Z = P*diag(diag(1./sqrt(P'*diag(d)*P)));
                Ncut = k - trace(Z'*A*Z);
                assert(Ncut>=0,'ncut is negative');

                Ncuts(j) = Ncut;
                running_time(j) = time_elapsed;
            end
        end
        save(join([data_name,algorithm,'.mat'],"_"),'running_time','Ncuts');
    end
end

