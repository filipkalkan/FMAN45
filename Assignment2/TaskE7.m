clear;
load A2_data.mat

svm = fitcsvm(train_data_01', train_labels_01, 'KernelFunction','gaussian');

prediction_train = predict(svm, train_data_01');
prediction_test = predict(svm, test_data_01');

performance_train = evaluate_svm(prediction_train, train_labels_01');
performance_test = evaluate_svm(prediction_test, test_labels_01');

%% Finding optimal beta
clear;
load A2_data.mat

performance_test = {};
i = 1;
for beta=1:0.2:8
    svm = fitcsvm(train_data_01', train_labels_01, 'KernelFunction','gaussian', 'KernelScale', beta);
    
    prediction_train = predict(svm, train_data_01');
    prediction_test = predict(svm, test_data_01');
    
    performance_train = evaluate_svm(prediction_train, train_labels_01');
    performance_test{i} = evaluate_svm(prediction_test, test_labels_01');
    i = i + 1;
end

for j = 1:size(performance_test, 2)
    misclassified(j) = performance_test{j}(1, 2) + performance_test{j}(2, 1);
end
save E7data performance_test misclassified

%% Produce Plot
hold on
scatter(1:0.2:8, misclassified);
xlabel('\beta')
ylabel('Number of misclassified samples')
hold off
