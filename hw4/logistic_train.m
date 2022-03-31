function [weights] = logistic_train(data, labels, epsilon, maxiter)

epsilon = 1e-5;
maxiter = 1000;
weights = zeros(size(data,2),1);

i = 1;
cont = true;
while ((i < maxiter) && cont)
    
    % feed the data into model
    y = sigmf(data * weights,[1 0]);
    R = diag( y .* (1 - y) );
    
    % prevent singularity
    a = 0.01;
    R = R + a * eye(length(R));
    
    % update weight
    z = (data * weights) - (R^(-1) * (y - labels));
    weights = (data' * R * data)^(-1) * data' * R * z;
        
    % get the new loss value
    y_new = sigmf(data * weights,[1 0]);
    diff = mean(abs(y_new - y));    
    
    % check training loss
    cont = diff > epsilon;
    
    % update step
    i = i + 1;
end
end
