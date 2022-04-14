clear;
close all;

% Load the Data
load USPS.mat

% SVD for matrix decomposition
[U,S,V] = svd(A);
[n,d] = size(S);


pc_nums = [10, 50, 100, 200];
recons_errs = zeros(numel(pc_nums),1);
for i = 1:numel(pc_nums)
   
    p = pc_nums(i);
    mask = [ones(1,p) zeros(1,d - p)];
    S1 = S * diag(mask);

    %S1 = mask.* S;

    % Reconstruct the images
    pca_imgs = U* S1 * V.';

    % Get the reconstruction errors for each image
    n = size(A,1);
    diff = A - pca_imgs;
    err = zeros(n,1);
    for i = 1:n
        err(i) = norm(diff(1,:), 2)^2;
    end
    
    % Reshape the images
    img1 = reshape(pca_imgs(1,:), 16, 16);
    img2 = reshape(pca_imgs(2,:), 16, 16);
    
    % Plot the reconstructed images
    imshow(img1);
    saveas(gcf,'img1.png')

    % Plot the reconstructed images
    imshow(img2);
    saveas(gcf,'img2.png')

    % Save the reconstruction errors
    recons_errs(i) = sum(err);
    disp(sum(err));
end
avg_recons_errs = (recons_errs / size(A,1));

