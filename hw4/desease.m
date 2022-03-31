clear; close all; clc;

% load data
data = load('alzheimers/ad_data.mat');
x_train = data.X_train;
x_test = data.X_test;
y_train = data.y_train;
y_test = data.y_test;
n_test = size(y_test);

par_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];


for i = 1:numel(par_list)
    
    % train the sparse logistic regression model
    [weight, bias] = logistic_l1_train(x_train, y_train, par_list(i));
    
    % nonzero feature numbers
    feat_num = sum(weight ~= 0);
    disp(feat_num);

    % test the model on test set
    pred = (x_test * weight) + bias;
    
    % Compute the accuracy and AUC
    pred2 = (pred > 0.5);
    y_test = (y_test > 0);
    acc = 1 - sum((pred2 - y_test).'*(pred2 - y_test)) / n_test(1);
    [far,gar,thres,auc] = perfcurve(y_test, pred,1);

    disp([feat_num, acc, auc]);
end

