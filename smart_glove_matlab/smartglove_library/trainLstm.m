function [net] = trainLstm(features, reducedFeatures, labels, numOfK)
%% Long short-term memory network training sequence

reducedFeatureMat = reducedFeatures;
labelMat = labels;
featureMat = features;
%yVector = ind2vec(y);
%xCell = mat2cell(x, 1);
labelCategories = categorical(labelMat);

% Sort the data by length
numObservations = numel(reducedFeatureMat);

for i = 1:numObservations
    CELL_SEQUENCE = reducedFeatureMat{i,1};
    m = size(CELL_SEQUENCE, 2);
    sequenceLengths(i) = m;
end

[sequenceLengths,idx] = sort(sequenceLengths);
reducedFeatureMat = reducedFeatureMat(idx);
labelCategories = labelCategories(idx);

ii = randperm(size(reducedFeatureMat,1), 20);
xValidation = reducedFeatureMat(ii);
yValidation = labelCategories(ii);

% for iC = 1:numel(x)
%   x{iC} = rot90(x{iC});
% end

%% Signal plotting

% figure
% plot(x{1}')
% xlabel("Time Step")
% title("Training Observation 1")
% numFeatures = size(x{1},1);
% legend("Feature " + string(1:numFeatures), 'Location', 'northeastoutside')
% hold on;

%% Sequence length plotting

% figure
% bar(sequenceLengths)
% ylim([0 6000])
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")

%% LSTM network options

maxEpochs = 100;
miniBatchSize = 17; % Total iteration count = maxEpochs * miniBatchSize

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Verbose',1, ...
    'ValidationData',{xValidation, yValidation}, ...
    'ValidationFrequency', 30, ...
    'LearnRateSchedule', 'piecewise', ...
    'Plots','training-progress'); % LSTM network training options


numFeatures = numOfK; % The number of input nodes
%numHiddenUnits1 = 125; % The number of layer 1 nodes
numHiddenUnits2 = 100; % The number of layer 2 nodes
numClasses = 2; % The number of output nodes
layers = [ ...
    sequenceInputLayer(numFeatures)
    %lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    %dropoutLayer(0.2)
    bilstmLayer(numHiddenUnits2,'OutputMode','last')
    %dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]; % Layer configuration

net = trainNetwork(reducedFeatureMat, labelCategories, layers, options);

end
