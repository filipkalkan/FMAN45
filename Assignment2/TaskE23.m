close all;
load A2_data.mat;
X = train_data_01;

%% 2 Clusters
K = 2;
[y_2, C_2] = K_means_clustering(X, K);
projected_2 = pca_linear(X);

%% Plot
figure
gscatter(projected_2(1, :), projected_2(2, :), y_2, 'br', 'ox');
xlabel('First Principal Component');
ylabel('Second Principal Component');
title 'K Means Clustering (K = 2)'

%% 5 Clusters
K = 5;
[y_5, C_5] = K_means_clustering(X, K);
projected_5 = pca_linear(X);

%% Plot
figure
gscatter(projected_5(1, :), projected_5(2, :), y, 'brgcm', 'oxv.*');
xlabel('First Principal Component');
ylabel('Second Principal Component');
title 'K Means Clustering (K = 5)'

%% 2 Clusters with Centroids
img_C_2a = reshape(C_2(:, 1), [28 28]);
img_C_2b = reshape(C_2(:, 2), [28 28]);

figure
hold on
subplot(1,2,1);
imshow(img_C_2a);
title('Cluster 1')
subplot(1,2,2);
imshow(img_C_2b);
title('Cluster 2')
hold off

%% 5 Clusters with Centroids
img_C_5a = reshape(C_5(:, 1), [28 28]);
img_C_5b = reshape(C_5(:, 2), [28 28]);
img_C_5c = reshape(C_5(:, 3), [28 28]);
img_C_5d = reshape(C_5(:, 4), [28 28]);
img_C_5e = reshape(C_5(:, 5), [28 28]);

figure
hold on
subplot(1,5,1);
imshow(img_C_5a);
title('Cluster 1')
subplot(1,5,2);
imshow(img_C_5b);
title('Cluster 2')
subplot(1,5,3);
imshow(img_C_5c);
title('Cluster 3')
subplot(1,5,4);
imshow(img_C_5d);
title('Cluster 4')
subplot(1,5,5);
imshow(img_C_5e);
title('Cluster 5')
hold off