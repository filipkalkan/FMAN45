load models/network_trained_with_momentum.mat

%% Plot filters
conv_layer = net.layers{1, 2};
conv_weights = conv_layer.params.weights;

batch_size = 16;
for label = 1:batch_size
    subplot(4, 4, label)
    imshow(conv_weights(:,:,label))
end

%% Find misclassified images
addpath(genpath('./'));
X_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');

X_test = reshape(X_test, [28, 28, 1, 10000]);
labels_test(labels_test==0) = 10;

predictions = zeros(numel(labels_test), 1);
batch_size = 20;
for label = 1:batch_size:size(labels_test)
    data_indices = label : min(label + batch_size - 1, numel(labels_test));
    output_by_layer = evaluate(net, X_test(:,:,:,data_indices), labels_test(data_indices));
    probabilities = output_by_layer{end-1};
    [~, prediction] = max(probabilities, [], 1);
    predictions(data_indices) = prediction;
end
predictions(predictions == 10) = 0;

%% Plot misclassified images
predictions(predictions == 10) = 0;
labels_test(labels_test == 10) = 0;
misclassified_indices = find(predictions ~= labels_test, 9);
for label = 1:9
    subplot(3, 3, label)
    imshow(X_test(:, :, :, misclassified_indices(label)));
    text = strcat([...
        'True label: ',...
        num2str(labels_test(misclassified_indices(label))),...
        newline,...
        'Predicted label: ',...
        num2str(predictions(misclassified_indices(label)))...
        ]);
    title(text);
end

%% Confusion matrix
confusion_matrix = confusionmat(labels_test, predictions);
confusionchart(confusion_matrix);

precision = zeros(1,10);
recall = zeros(1,10);
for label=1:10
    n_correct_predictions = confusion_matrix(label, label);
    n_true = sum(confusion_matrix(:, label));
    precision(label) = n_correct_predictions / n_true;

    n_predictions = sum(confusion_matrix(label, :));
    recall(label) = n_correct_predictions / n_predictions; 
end 

%% Get number of params in net
weights = zeros(1, 9);
biases = zeros(1, 9);

for layer_index=1:9
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





