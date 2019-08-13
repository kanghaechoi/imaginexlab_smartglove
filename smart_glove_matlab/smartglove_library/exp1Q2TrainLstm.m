function [net] = exp1Q2TrainLstm(features, labels)
%% Long short-term memory network training sequence

labelMat = labels;
featureMat = features;
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

ii = randperm(size(featureMat,1), 10);
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

trainingPrompt = 'Will you continue to train network? (y/n): ';
Ans = input(trainingPrompt, 's');
if ~isempty(Ans)
    if Ans == 'y'
        codeContinue = true;
        fprintf("\n");
    elseif Ans == 'n'
        fprintf("See ya!\n\n");
    else
        fprintf("You inserted a wrong letter\n\n");
    end
end

%% LSTM network options
if codeContinue == true
    maxEpochs = 100;
    miniBatchSize = 11; % Total iteration count = maxEpochs * miniBatchSize

    options = trainingOptions('adam', ...
        'ExecutionEnvironment','auto', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'Verbose',1, ...
        'ValidationData',{xValidation, yValidation}, ...
        'ValidationFrequency', 22, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'Plots','training-progress'); % LSTM network training options


    numFeatures = 26; % The number of input nodes
    numHiddenUnits1 = 125; % The number of layer 1 nodes 
    %numHiddenUnits2 = 75; % The number of layer 2 nodes
    %numHiddenUnits3 = 500; % The number of layer 2 nodes
    numClasses = 18; % The number of output nodes
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
    
    %[XTest,YTest] = digitTest4DArrayData;
    YPredicted = classify(net,xValidation);
    plotconfusion(yValidation,YPredicted)
    
end
