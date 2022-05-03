% clear;
load A2_data.mat

X_train = train_data_01;
X_test = test_data_01;
labels_train = train_labels_01;
labels_test = test_labels_01;

% K = 8;

[y_train, C_train] = K_means_clustering(X_train, K);
cluster_labels_train = k_means_classifier(labels_train, y_train, K);

[y_test, C_test] = K_means_clustering(X_test, K);
cluster_labels_test = k_means_classifier(labels_test, y_test, K);

for i=1:K
    n_zeros_train(i) = sum(labels_train(y_train == i) == 0);
    n_ones_train(i) = sum(labels_train(y_train == i) == 1);
    n_zeros_test(i) = sum(labels_test(y_test == i) == 0);
    n_ones_test(i) = sum(labels_test(y_test == i) == 1);

    if cluster_labels_train(i) == 1
        misclassified_train(i) = n_zeros_train(i);
    else
        misclassified_train(i) = n_ones_train(i);
    end

    if cluster_labels_test(i) == 1
        misclassified_test(i) = n_zeros_test(i);
    else
        misclassified_test(i) = n_ones_test(i);
    end

end

misclassified_sum_train = sum(misclassified_train);
misclassification_rate_train = misclassified_sum_train ./ size(X_train, 2);

misclassified_sum_test = sum(misclassified_test);
misclassification_rate_test = misclassified_sum_test ./ size(X_test, 2);

% Analysis of different K performances
% Uncomment, define K and set the variables below to initial value []
% load E5data
% misclassification_rate_trains = [misclassification_rate_trains, misclassification_rate_train];
% misclassification_rate_tests = [misclassification_rate_tests, misclassification_rate_test];
% 
% save E5data misclassification_rate_tests misclassification_rate_trains

%% Produce plot
hold on
scatter(1:10, log(misclassification_rate_tests), 'red');
scatter(1:10, log(misclassification_rate_trains), 'blue');
xlabel 'K'
ylabel 'log(Misclassification rate (%))'
legend('Test Data Set', 'Training Data Set');
hold off




