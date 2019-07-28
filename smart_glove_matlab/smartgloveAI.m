%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO
% Last update: July 20th, 2019

%% Include smartglove function library
addpath('smartglove_library');

%% Initialization
codeInitialize;

% Network selection logics (0: PCA, 1:reliefF, 2:Pattern Net, 3:LSTM)
NETSELECT = uint8(1);

%% 5 Hz low-pass filter
SAMPLE_FREQ = single(100); %Sample frequency 100Hz
[B1, A1] = butterworth5hz(SAMPLE_FREQ); % [B1,A1] =butterworth5hz(FS): 5Hz low-pass filter

%% File count & Age extraction 
[fileCount20, age20] = inputCount(20); %[fileCount, AGE] = inputCount(AGE)
[fileCount40, age40] = inputCount(40); %[fileCount, AGE] = inputCount(AGE)
[fileCount60, age60] = inputCount(60); %[fileCount, AGE] = inputCount(AGE)

%% Network selection

if(NETSELECT == 0)
%% Smart glove data read & feature extraction
    %Subjects in 20s
    for n20 = 1 : fileCount20
        if n20 == 1
            [features20Data] = featureExtractionPca2dfd(age20, n20); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features20 = features20Data;
            continue;
        else
            [features20Data] = ...
                featureExtractionPca(age20, n20); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features20 = [features20; features20Data];
        end
    end
    
    %Subjects in 40s
    for n40 = 1 : fileCount40
        if n40 == 1
            [features40Data] = featureExtractionPca2(age40, n40); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features40 = features40Data;
            continue;
        else
            [features40Data] = ...
                featureExtractionPca(age40, n40); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features40 = [features40; features40Data];
        end
    end
        
    %Subjects in 60s
    for n60 = 1 : fileCount60
        if n60 == 1
            [features60Data] = featureExtractionPca2(age60, n60); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features60 = features60Data;
            continue;
        else
            [features60Data] = ...
                featureExtractionPca(age60, n60); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features60 = [features60; features60Data];
        end
    end
    
    %Subjects in total
    featuresTotal = [features20; features40; features60]; %Features from 20s + Features from 60s
    scaledFeaturesTotal = featureScale(featuresTotal);
    
%% Principle Component Analysis
    [pcaMat, numOfK, varRetain] = featureReduction(featuresTotal);

    fprintf("The ideal number of K = %d\n\n", numOfK);
    
    save pcaResult.mat pcaMat numOfK varRetain ...
        fileCount20 age20 fileCount40 age40 fileCount60 age60 ...
        A1 B1;
    
%% Network training selection after PCA   
    prompt1 = 'Will you continue? (Y/N): ';
    continueAns = input(prompt1, 's');
    if ~isempty(continueAns)
        if continueAns == 'Y'
            codeInitialize;
            fprintf("Okay!\n\n");
            prompt2 = "Will you continue to train neural network?\n" + ...
               "If so, choose '1'(CNN) or '2'(LSTM): ";
            NETSELECT = input(prompt2);
            if NETSELECT ~= 1 && NETSELECT ~= 2
                fprintf("You inserted a wrong number\n");
            end
            fprintf("\n");
        elseif continueAns == 'N'
            fprintf("See ya!\n");
        else
            fprintf("You inserted a wrong letter\n");
        end
    end

end

if(NETSELECT == 1)
%% Smart glove data read & feature extraction & rms

    %Subjects in 20s
    features20 = [];
    labels20 = [];
    rms20 = [];
    
    for n20 = 1 : fileCount20
        [features20Data, labels20Data, rms20Data] ... 
            = featureExtraction(B1, A1, age20, fileCount20); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        features20 = [features20; features20Data];
        labels20 = [labels20; labels20Data];
        rms20 = [rms20; rms20Data];
    end
    
    %Subjects in 40s
    features40 = [];
    labels40 = [];
    rms40 = [];    

    for n40 = 1 : fileCount40
        [features40Data, labels40Data, rms40Data] ...
            = featureExtraction(B1, A1, age40, fileCount40); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        features40 = [features40; features40Data];
        labels40 = [labels40; labels40Data];
        rms40 = [rms40; rms40Data];
    end
    
    %Subjects in 60s
    features60 = [];
    labels60 = [];
    rms60 = [];    
    
    for n60 = 1 : fileCount60
        [features60Data, labels60Data, rms60Data] ...
            = featureExtraction(B1, A1, age60, fileCount60); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        features60 = [features60; features60Data];
        labels60 = [labels60; labels60Data];
        rms60 = [rms60; rms60Data];
    end

    rmsTotal = [rms20; rms40; rms60];
    labelsTotal = [labels20; labels40; labels60];
    
    [ranks, weights] = relieff(rmsTotal, labelsTotal, 4, ...
        'method', 'classification');
    
    bar(weights(ranks))
    xlabel('Predictor rank');
    ylabel('Predictor important weight');
    
end

if(NETSELECT == 2)
%% Smart glove data read & feature extraction
%     %Subjects in 20s
%     [features20, labels20] = featureExtraction(B1, A1, age20, fileCount20); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%     %Subjects in 40s
%     [features40, labels40] = featureExtraction(B1, A1, age40, fileCount40); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%     %Subjects in 60s
%     [features60, labels60] = featureExtraction(B1, A1, age60, fileCount60); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
% 
%     %Subjects in total
%     featuresTotal = [features20; features40; features60]; %Features from 20s + Features from 60s
%     labelsTotal = [labels20; labels40; labels60]; %Labels from 20s + Labels from 60s
%     
%% Smart glove data read & feature extraction in 3D array

    %Subjects in 20s
    features20 = [];
    labels20 = [];
    rms20 = [];
    
    for n20 = 1 : fileCount20
        [features20Data, labels20Data, rms20Data] ... 
            = featureExtraction(B1, A1, age20, fileCount20); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        features20 = [features20; features20Data];
        labels20 = [labels20; labels20Data];
        rms20 = [rms20; rms20Data];
    end
    
    %Subjects in 40s
    features40 = [];
    labels40 = [];
    rms40 = [];    

    for n40 = 1 : fileCount40
        [features40Data, labels40Data, rms40Data] ...
            = featureExtraction(B1, A1, age40, fileCount40); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        features40 = [features40; features40Data];
        labels40 = [labels40; labels40Data];
        rms40 = [rms40; rms40Data];
    end
    
    %Subjects in 60s
    features60 = [];
    labels60 = [];
    rms60 = [];    
    
    for n60 = 1 : fileCount60
        [features60Data, labels60Data, rms60Data] ...
            = featureExtraction(B1, A1, age60, fileCount60); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        features60 = [features60; features60Data];
        labels60 = [labels60; labels60Data];
        rms60 = [rms60; rms60Data];
    end

%% Signal analyzing

%     signalAnalyzer;

%% Convolutional neural network training

    %[net, tr] = trainCnn(featuresTotal, labelsTotal); %Pattern network

end

if(NETSELECT == 3)
%% Load PCA matrix
  
    %load('pcaResult.mat');

%% Smart glove data read & feature extraction

    features20Cell = {};
    reducedFeatures20Cell = {};
    features40Cell = {};
    reducedFeatures40Cell = {};
    features60Cell = {};
    reducedFeatures60Cell = {};

    %Subjects in 20s
    for n20 = 1 : fileCount20
        if n20 == 1
            [features20Data, labels20Data, lengths20] = ...
                featureExtractionToCell(B1, A1, age20, n20); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features20Cell = {features20Data};
            %reducedFeatures20Cell = {reducedFeatures20Data};
            lengthss20 = lengths20;
            labels20Cell = labels20Data;
            continue;
        else
            [features20Data, labels20Data, lengths20] = ...
                featureExtractionToCell(B1, A1, age20, n20); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features20Cell = [features20Cell; {features20Data}];
            %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
            lengthss20 = [lengthss20; lengths20];
            labels20Cell = [labels20Cell; labels20Data];
        end
    end

    %Subjects in 40s
    for n40 = 1 : fileCount40
        if n40 == 1
            [features40Data, labels40Data, lengths40] = ...
                featureExtractionToCell(B1, A1, age40, n40); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features40Cell = {features40Data};
            %reducedFeatures40Cell = {reducedFeatures40Data};
            lengthss40 = lengths40;
            labels40Cell = labels40Data;
            continue;
        else
            [features40Data, labels40Data, lengths40] = ...
                featureExtractionToCell(B1, A1, age40, n40); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features40Cell = [features40Cell; {features40Data}];
            %reducedFeatures40Cell = [reducedFeatures40Cell; {reducedFeatures40Data}];
            lengthss40 = [lengthss40; lengths40];
            labels40Cell = [labels40Cell; labels40Data];
        end
    end
    
    %Subjects in 60s
    for n60 = 1 : fileCount60
        if n60 == 1
            [features60Data, labels60Data, lengths60] = ...
                featureExtractionToCell(B1, A1, age60, n60); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features60Cell = {features60Data};
            %reducedFeatures60Cell = {reducedFeatures60Data};
            lengthss60 = lengths60;
            labels60Cell = labels60Data;
            continue;
        else
            [features60Data, labels60Data, lengths60] = ...
                featureExtractionToCell(B1, A1, age60, n60); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            features60Cell = [features60Cell; {features60Data}];
            %reducedFeatures60Cell = [reducedFeatures60Cell; {reducedFeatures60Data}];
            lengthss60 = [lengthss60; lengths60];
            labels60Cell = [labels60Cell; labels60Data];
        end
    end
    
    %Subjects in total
    featuresTotalCell = [features20Cell; ...
        features40Cell; ...
        features60Cell]; %Features from 20s + Features from 60s
    %reducedFeaturesTotalCell = [reducedFeatures20Cell; ...
    %    reducedFeatures40Cell; ...
    %    reducedFeatures60Cell];
    labelsTotalCell = [labels20Cell; ... 
        labels40Cell; ...
        labels60Cell];

%% Signal analyzing

    %signalAnalyzer;
    
%% Long short-term memory network training

%    [net] = trainLstm(featuresTotalCell, ...
%        labelsTotalCell); %LSTM network
    
end