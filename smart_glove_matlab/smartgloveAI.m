%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO
%% Include smartglove function library
addpath('smartglove_library');

%% Initialization
codeInitialize;

% Network selection logics (0: PCA, 1:CNN, 2:LSTM)
NETSELECT = uint8(0);

%% 5 Hz low-pass filter
SAMPLE_FREQ = single(100); %Sample frequency 100Hz
[B1, A1] = butterworth5hz(SAMPLE_FREQ); % [B1,A1] =butterworth5hz(FS): 5Hz low-pass filter

[fileCount20, age20] = inputCount(20); %[fileCount, AGE] = inputCount(AGE)
[fileCount60, age60] = inputCount(60); %[fileCount, AGE] = inputCount(AGE)

if(NETSELECT == 0)
%% Smart glove data read & feature extraction
    %Subjects in 20s
    [features20, labels20] = featureExtraction(B1, A1, age20, fileCount20); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
    %Subjects in 60s
   
    [features60, labels60] = featureExtraction(B1, A1, age60, fileCount60); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction

    %Subjects in total
    featuresTotal = [features20; features60]; %Features from 20s + Features from 60s
    labelsTotal = [labels20; labels60]; %Labels from 20s + Labels from 60s
    
%% Principle Component Analysis
    [reducedFeatures, numOfK] = featureReduction(featuresTotal);

end
    
if(NETSELECT == 1)
%% Smart glove data read & feature extraction
    %Subjects in 20s
    [features20, labels20] = featureExtraction(B1, A1, age20, fileCount20); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
    %Subjects in 60s
   
    [features60, labels60] = featureExtraction(B1, A1, age60, fileCount60); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction

    %Subjects in total
    featuresTotal = [features20; features60]; %Features from 20s + Features from 60s
    labelsTotal = [labels20; labels60]; %Labels from 20s + Labels from 60s
    
%% Smart glove data read & feature extraction in 3D array
%     Subjects in 20s
%     [features20, labels20] = featureExtraction3D(B1, A1, age20, fileCount20); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%         
%     Subjects in 60s
%     [features60, labels60] = featureExtraction3D(B1, A1, age60, fileCount60); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction

%% Signal analyzing

%     signalAnalyzer;

%% Convolutional neural network training

    %[net, tr, ranks, weights] = trainCnn(featuresTotal, labelsTotal); %Pattern network

end

if(NETSELECT == 2)
%% Smart glove data read & feature extraction

    %Subjects in 20s
    [features20Cell, labels20Cell] = featureExtractionToCell(B1, A1, age20, fileCount20); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction

    %Subjects in 60s
    [features60Cell, labels60Cell] = featureExtractionToCell(B1, A1, age60, fileCount60); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
    %Subjects in total
    featuresTotalCell = [features20Cell; features60Cell]; %Features from 20s + Features from 60s
    labelsTotalCell = [labels20Cell; labels60Cell]; %

%% Signal analyzing

    %signalAnalyzer;
    
%% Long short-term memory network training

    %[net] = trainLstm(featuresTotalCell, labelsTotalCell); %LSTM network
    
end