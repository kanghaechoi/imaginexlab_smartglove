%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO
%% Include smartglove function library
addpath('smartglove_library');

%% Initialization
codeInitialize;

% Network selection logics
LSTM = true;
CNN = false;

%% 5 Hz low-pass filter
SAMPLE_FREQ = 100; %Sample frequency 100Hz
[B1, A1] = butterworth5hz(SAMPLE_FREQ); % [B1,A1] =butterworth5hz(FS): 5Hz low-pass filter

[fileCount20, age20] = inputCount(20); %[fileCount, AGE] = inputCount(AGE)
[fileCount60, age60] = inputCount(60); %[fileCount, AGE] = inputCount(AGE)

if(CNN)
%% Smart glove data read & feature extraction

    %Subjects in 20s
    [features20, labels20] = featureExtraction(B1, A1, age20, fileCount20); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
    %Subjects in 60s
   
    [features60, labels60] = featureExtraction(B1, A1, age60, fileCount60); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction

    %Subjects in total
    featuresTotal = [features20; features60]; %Features from 20s + Features from 60s
    labelsTotal = [labels20; labels60]; %Labels from 20s + Labels from 60s

%% Signal analyzing

    %signalAnalyzer;

%% Convolution neural network training

    %[net, tr] = trainCnn(featuresTotal, labelsTotal); %Pattern network

%% Long short-term memory network training

    %[net] = trainLstm(featuresTotal, labelsTotal); %LSTM network

end

if(LSTM)
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

    [net] = trainLstm(featuresTotalCell, labelsTotalCell); %LSTM network
    
end