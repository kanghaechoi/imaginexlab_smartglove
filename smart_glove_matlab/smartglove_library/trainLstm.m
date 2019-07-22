function [net] = trainLstm(features, labels)
%% Long short-term memory network training sequence

%reducedFeatureMat = reducedFeatures;
labelMat = labels;
featureMat = features;
%yVector = ind2vec(y);
%xCell = mat2cell(x, 1);
labelCategories = categorical(labelMat);
codeContinue = false;

% Sort the data by length
numObservations = numel(featureMat);

for i = 1:numObservations
    CELL_SEQUENCE = featureMat{i,1};
    m = size(CELL_SEQUENCE, 2);
    sequenceLengths(i) = m;
end

% [sequenceLengths,idx] = sort(sequenceLengths);
% featureMat = featureMat(idx);
% labelCategories = labelCategories(idx);

ii = randperm(size(featureMat,1), 21);
xValidation = featureMat(ii);
featureMat(ii) = [];
yValidation = labelCategories(ii);
labelCategories(ii) =[];

% for iC = 1:numel(x)
%   x{iC} = rot90(x{iC});
% end

%% Signal plotting

% figure
% plot(reducedFeatureMat{1}')
% xlabel("Time Step")
% title("Training Observation 1")
% numFeatures = size(reducedFeatureMat{1},1);
% legend("Feature " + string(1:numFeatures), 'Location', 'northeastoutside')
% hold on;

%% Sequence length plotting

% figure
% bar(sequenceLengths)
% ylim([0 10000])
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")

%% Check if to continue training neural network 

trainingPrompt = 'Will you continue to train network? (Y/N): ';
Ans = input(trainingPrompt, 's');
if ~isempty(Ans)
    if Ans == 'Y'
        codeContinue = true;
        fprintf("\n");
    elseif Ans == 'N'
        fprintf("See ya!\n\n");
    else
        fprintf("You inserted a wrong letter\n\n");
    end
end

%% LSTM network options
if codeContinue == true
    maxEpochs = 100;
    miniBatchSize = 10; % Total iteration count = maxEpochs * miniBatchSize

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
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'Plots','training-progress'); % LSTM network training options


    numFeatures = 26; % The number of input nodes
    numHiddenUnits1 = 125; % The number of layer 1 nodes
%     numHiddenUnits2 = 75; % The number of layer 2 nodes
    numHiddenUnits3 = 100; % The number of layer 2 nodes
    numClasses = 3; % The number of output nodes
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits1,'OutputMode','sequence')
        %dropoutLayer(0.2)
        %lstmLayer(numHiddenUnits2,'OutputMode','sequence')
        bilstmLayer(numHiddenUnits3,'OutputMode','last')
        %dropoutLayer(0.2)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer]; % Layer configuration

    net = trainNetwork(featureMat, labelCategories, layers, options);
end
    
end
