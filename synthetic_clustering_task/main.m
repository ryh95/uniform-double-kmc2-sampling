
addpath(genpath('../clustering_algorithms/'));
addpath(genpath('../clustering_algorithms/kmc2'));

data_name = 'b2-random-20.txt'; % a2/a3/b2-random-10/b2-random-15/b2-random-20
algorithm = 'Double-kmc2'; % kmc2/uniform-wo-wk/Double-kmc2

if  strcmp(data_name,'a2.txt')
    X = importdata(strcat('../data/',data_name));
    X = X';
    k = 35;
elseif strcmp(data_name,'a3.txt')
    X = importdata(strcat('../data/',data_name));
    X = X';
    k = 50;
elseif strcmp(data_name,'b2-random-10.txt') || strcmp(data_name,'b2-random-15.txt') || strcmp(data_name,'b2-random-20.txt')
    X = importdata(strcat('../data/',data_name));
    X = X';
    k = 100;
elseif strcmp(data_name,'dim15.txt')
    % dimension is 15
    X = importdata(strcat('../data/',data_name));
    X = X';
    k = 9;
end

n = size(X,2);

chain_length = 200;

m = floor(0.35*log(n)^4); % sample number 0.35(Double-kmc2)/0.7(uniform-wo-wk)

repeat = 20;
sum_squared_distances = zeros(1,repeat);
running_time = zeros(1,repeat);
for i = 1:repeat
    
    if strcmp(algorithm,'kmc2')
        tic;
        X_np = py.numpy.array(transpose(X(:)));
        rt_centers_indices = py.kmc2caller.call_kmc2(X_np,int32(n),int32(k),int32(chain_length),false);
        centers = reshape(double(py.array.array('d',py.numpy.nditer(rt_centers_indices{1}))),[],k);
        running_time(i) = toc;
%         tic;
%         centers = KMC2(X,k,chain_length); % too slow
%         running_time(i) = toc;
    elseif strcmp(algorithm,'uniform-wo-wk')
        tic;
        X_sample = datasample(X,m,2,'Replace',false);
        options.careful = true;
        [label, centers, dis] = fkmeans(X_sample.', k, options);
        centers = centers.';
        running_time(i) = toc;
    elseif strcmp(algorithm,'Double-kmc2')
        tic;
        centers = DoubleKMC2(X,m,k,chain_length);
        centers = centers.';
        running_time(i) = toc;
    end
    
    distances = dot(X,X)' + dot(centers,centers) - 2*X'*centers;
    sum_squared_distances(i) = sum(min(distances,[],2));
end
save(join([algorithm,'_','running-time','.mat'],"_"),'running_time');
save(join([algorithm,'_','sum-squared-distances','.mat'],"_"),'sum_squared_distances');