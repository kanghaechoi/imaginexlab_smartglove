function [net] = train_lstm(features, labels)
%% Long short-term memory network training sequence

x = features';
y = labels;
%y_vector = ind2vec(y);
x_cell = mat2cell(x, 1);
y_categories = categorical(y);

input_size = 11;
hidden_layer_size = 100;
output_size = 2;

layers = [sequenceInputLayer(input_size), lstmLayer(hidden_layer_size, 'OutputMode', 'last'), ...
    fullyConnectedLayer(output_size), softmaxLayer, classificationLayer];

max_epochs = 100;
mini_batch_size = 100;

options = trainingOptions('adam', 'ExecutionEnvironment', 'auto', ...,
    'MaxEpochs', max_epochs, 'MiniBatchSize', mini_batch_size, ...
    'GradientThreshold', 1, 'Verbose', false, 'Plots', 'training-progress');

net = trainNetwork(x_cell', y_categories, layers, options);

end

