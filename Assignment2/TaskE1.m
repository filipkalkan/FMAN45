load A2_data.mat

% Use the training data. each column represents an image.
X = train_data_01;

projected = pca_linear(X);

%% Plot
gscatter(projected(1, :), projected(2, :), train_labels_01);
xlabel('First Principal Component');
ylabel('Second Principal Component');
title 'Principal Component Analysis'