function [net] = exp2TrainLstmLoop(features, labels, NumberFeatures)
%% Long short-term memory network training sequence

labelMat = labels;
featureMat = features;
labelCategories = categorical(labelMat);
codeContinue = true;

% Sort the data by length
numObservations = numel(featureMat);

for i = 1:numObservations
    CELL_SEQUENCE = featureMat{i,1};
    m = size(CELL_SEQUENCE, 2);
    sequenceLengths(i) = m;
end

ii = randperm(size(featureMat,1), 4);
xValidation = featureMat(ii);
featureMat(ii) = [];
yValidation = labelCategories(ii);
labelCategories(ii) =[];

%% LSTM network options
if codeContinue == true
    maxEpochs = 100;
    miniBatchSize = 9; % Total iteration count = maxEpochs * miniBatchSize

    options = trainingOptions('adam', ...
        'ExecutionEnvironment','auto', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'Verbose',1, ...
        'ValidationData',{xValidation, yValidation}, ...
        'ValidationFrequency', 18, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'Plots','training-progress'); % LSTM network training options


    numFeatures = NumberFeatures; % The number of input nodes
    numHiddenUnits1 = 125; % The number of layer 1 nodes 
    %numHiddenUnits2 = 75; % The number of layer 2 nodes
    %numHiddenUnits3 = 500; % The number of layer 2 nodes
    numClasses = 8; % The number of output nodes
    layers = [ ...
        sequenceInputLayer(numFeatures)
        bilstmLayer(numHiddenUnits1,'OutputMode','last')
        dropoutLayer(0.5)
        %lstmLayer(numHiddenUnits2,'OutputMode','sequence')
        %bilstmLayer(numHiddenUnits3,'OutputMode','last')
        %dropoutLayer(0.2)
        fullyConnectedLayer(numClasses)
        %dropoutLayer(0.2)
        softmaxLayer
        classificationLayer]; % Layer configuration

    [net] = trainNetwork(featureMat, labelCategories, layers, options);
end
    
end
