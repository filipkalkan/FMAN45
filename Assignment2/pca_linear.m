function projected = pca_linear(X)
% Map to zero mean images
X = X - mean(X, 2);

[U, ~, ~] = svd(X);

% Project onto first and second pricipal component
principal_components = U(:, 1:2);
projected = principal_components' * X;
end

