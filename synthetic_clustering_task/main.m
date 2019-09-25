
addpath(genpath('../clustering_algorithms/'));
addpath(genpath('../clustering_algorithms/kmc2'));

% data_names = ["a2","a3","b2-random-10","b2-random-15","b2-random-20"];
% data_names = ["b2-random-20"];
% algorithms = ["kmc2","uniform-wo-wk","Double-kmc2"];
algorithms = ["kmc2","uniform-wo-wk","Double-kmc2"];
data_names = ["poker-hand","bio_train","rna"];

for data_name = data_names
    
    % todo: feature standardize 
    if  strcmp(data_name,'a2')
        X = importdata(strcat('../data/',data_name,'.txt'));
        k = 35;
    elseif strcmp(data_name,'a3')
        X = importdata(strcat('../data/',data_name,'.txt'));
        k = 50;
    elseif strcmp(data_name,'b2-random-10') || strcmp(data_name,'b2-random-15') || strcmp(data_name,'b2-random-20')
        X = importdata(strcat('../data/',data_name,'.txt'));
        k = 100;
    elseif strcmp(data_name,'dim15')
        % dimension is 15
        X = importdata(strcat('../data/',data_name,'.txt'));
        k = 9;
    elseif strcmp(data_name,'bio_train')
        X = load(strcat('../data/',data_name,'.dat'));
        X = X(:,4:end);
        k = 200;
    elseif strcmp(data_name,'rna')
        X = importdata(strcat('../data/',data_name,'.txt'));
        X = X(:,2:end);
        k = 200;
    elseif strcmp(data_name,'poker-hand')
        X = importdata(strcat('../data/',data_name,'.data'));
        X = X(:,1:end-1);
        k = 10;
    end
    
    X = (X-mean(X))./std(X);
    assert(~any(any(isnan(X))));
    X = X';
    n = size(X,2);
    
    for algorithm = algorithms
        
        chain_length = 200;

        % sample number 0.35(Double-kmc2)/0.7(uniform-wo-wk)
        if strcmp(algorithm,'uniform-wo-wk')
            m = floor(0.7*log(n)^4);
            n_iter = 2000;
        elseif strcmp(algorithm,'Double-kmc2')
            m = floor(1.5*log(n)^2);
            n_iter = 2000;
        end
        
        if ~strcmp(algorithm,'kmc2')
            assert(m>=k);
        end
        
        repeat = 40;
        sum_squared_distances = zeros(1,repeat);
        running_time = zeros(1,repeat);

        for i = 1:repeat
            dis_calculator = 0;
            if strcmp(algorithm,'kmc2')
%                 tic;
                [centers,sample_index,dis_calculator] = KMC2(X,k,chain_length,dis_calculator);
                running_time(i) = dis_calculator;
            elseif strcmp(algorithm,'uniform-wo-wk')
%                 tic;
                X_sample = datasample(X,m,2,'Replace',false);
                options.careful = true;
                [label, centers, dis, dis_calculator] = fkmeans(X_sample.',k,n_iter,dis_calculator,options);
                centers = centers';
                running_time(i) = dis_calculator;
            elseif strcmp(algorithm,'Double-kmc2')
%                 tic;
                [centers,dis_calculator] = DoubleKMC2(X,m,k,chain_length,n_iter,dis_calculator);
                centers = centers.';
                running_time(i) = dis_calculator;
            end

            distances = dot(X,X)' + dot(centers,centers) - 2*X'*centers;
            sum_squared_distances(i) = sum(min(distances,[],2));
        end
        save(join([data_name,algorithm,'running-time','.mat'],"_"),'running_time');
        save(join([data_name,algorithm,'sum-squared-distances','.mat'],"_"),'sum_squared_distances');
    end
end