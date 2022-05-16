load models/cifar10_modified.mat

%% Find misclassified images
addpath(genpath('./'));
[X_train, labels_train, X_test, labels_test, classes] = load_cifar10(1);
data_mean = mean(mean(mean(X_test, 1), 2), 4); % mean RGB triplet
X_test_norm = bsxfun(@minus, X_test, data_mean);

predictions = zeros(numel(labels_test), 1);
batch_size = 20;
for filter_index = 1:batch_size:size(labels_test)
    data_indices = filter_index : min(filter_index + batch_size - 1, numel(labels_test));
    output_by_layer = evaluate(net, X_test_norm(:,:,:,data_indices), labels_test(data_indices));
    probabilities = output_by_layer{end-1};
    [~, prediction] = max(probabilities, [], 1);
    predictions(data_indices) = prediction;
end

%% Plot misclassified images
misclassified_indices = find(predictions ~= labels_test, 9);
for filter_index = 1:9
    subplot(3, 3, filter_index)
    imshow(X_test(:, :, :, misclassified_indices(filter_index))/255);
    text = strcat([...
        'True label: ',...
        classes(labels_test(misclassified_indices(filter_index))),...
        newline,...
        'Predicted label: ',...
        classes(predictions(misclassified_indices(filter_index)))...
        ]);
    title(text);
end

%% Confusion matrix
confusion_matrix = confusionmat(double(labels_test), predictions);
confusionchart(confusion_matrix);

precision = zeros(1,10);
recall = zeros(1,10);
for filter_index=1:10
    n_correct_predictions = confusion_matrix(filter_index, filter_index);
    n_true = sum(confusion_matrix(:, filter_index));
    precision(filter_index) = n_correct_predictions / n_true;

    n_predictions = sum(confusion_matrix(filter_index, :));
    recall(filter_index) = n_correct_predictions / n_predictions; 
end 

%% Get number of params in net
weights = zeros(1, numel(net.layers));
biases = zeros(1, numel(net.layers));

for layer_index=1:numel(net.layers)
    layer = net.layers{1, layer_index};
    weights(layer_index) = 0;
    biases(layer_index) = 0;
    if isfield(layer, 'params')
        if isfield(layer.params, 'weights')
            weights(layer_index) = numel(layer.params.weights);
        end
        if isfield(layer.params, 'biases')
            biases(layer_index) = numel(layer.params.biases);
        end
    end
end

weights_total = sum(weights);
biases_total = sum(biases);

%% Plot filters
conv_weights = net.layers{1, 2}.params.weights;

n_filters = 16;
for filter_index = 1:n_filters
    subplot(4, 4, filter_index)
    imshow(conv_weights(:,:,:,filter_index)*10);
end



