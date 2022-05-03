clear;
load A2_data.mat

svm = fitcsvm(train_data_01', train_labels_01);

prediction_train = predict(svm, train_data_01');
prediction_test = predict(svm, test_data_01');

performance_train = evaluate_svm(prediction_train, train_labels_01');
performance_test = evaluate_svm(prediction_test, test_labels_01');