clc;
clear;

% load data
dim = 100;
sigma1 = eye(dim) * 0.1;

mu1 = mvnrnd(zeros(1, dim), sigma1, 1)
mu2 = mvnrnd(zeros(1, dim), sigma1, 1);
mu3 = mvnrnd(zeros(1, dim), sigma1, 1);

sigma = eye(dim);
num = 500;
num2 = num * 3;

data1 = mvnrnd(mu1, sigma, num);
data2 = mvnrnd(mu2, sigma, num);
data3 = mvnrnd(mu3, sigma, num);
data = [data1; data2; data3];

label = [ones(num, 1) ; 2*ones(num,1) ; 3*ones(num,1)];
data_size = size(data);


% perform k_means algorithm
clusters = 3;
[cluster_label, step] = k_means(data, clusters);

% evaluate the clutter result according to given label
eval = zeros(3, clusters);
for i = 1:3
    for j = 1:clusters
        for k = 1:num2
            if ((label(k)==i) && (cluster_label(k)==j))
                eval(i, j) = eval(i, j) + 1;
            end
        end
    end
end

disp(eval);

% Perform spectral relaxation clusering
[cluster_label, s] = spectral(data, clusters, 3);

% evaluate the clutter result according to given label
eval = zeros(3, clusters);
for i = 1:3
    for j = 1:clusters
        for k = 1:num2
            if ((label(k)==i) && (cluster_label(k)==j))
                eval(i, j) = eval(i, j) + 1;
            end
        end
    end
end

disp(eval);
