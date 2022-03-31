clear; close all; clc;

% load spam data
data = load('spam_email/data.txt');
data = [data , ones(size(data,1),1)];
labels = load('spam_email/labels.txt');

% split data to train and test
x_train = data(1:2001,:);
y_train = labels(1:2001);
x_test = data(2001:4601,:);
y_test = labels(2001:4601);
n_test = size(y_test);

% train the model with different train data size
train_number_list = [200, 500, 800, 1000, 1500, 2000];
for i = 1:6
    
    n = train_number_list(i);
    x = x_train(1:n,:);
    y = y_train(1:n);

    % train model
    weights = logistic_train(x, y);
    w(i,:) = weights;
    
    % evaluate model
    pred = sigmf(x_test * weights, [1 0]);
    pred = (pred > 0.5);
    acc = 1 - sum((pred - y_test).'*(pred - y_test)) / n_test(1);
    disp(acc);
end

