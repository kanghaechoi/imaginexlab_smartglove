%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO

%% Include smartglove function library
addpath('smartglove_library');

%readFolders()

%% Initialization
codeInitialize;

% Network selection logics (0: PCA, 1:reliefF, 2:Pattern Net, 3:LSTM)
NETSELECT = uint8(0);

%% 5 Hz low-pass filter
SAMPLE_FREQ = single(100); %Sample frequency 100Hz
[B1, A1] = butterworth5hz(SAMPLE_FREQ); % [B1,A1] =butterworth5hz(FS): 5Hz low-pass filter

%% File count & Age extraction 
maleID = 1;
femaleID = 2;

gesture1ID = 01;
gesture2ID = 02;
gesture3ID = 03;
gesture4ID = 04;
gesture5ID = 05;
gesture6ID = 06;

[fileCountM20, ageM20] = inputCount(20, maleID); %[fileCount, AGE] = inputCount(AGE)
[fileCountF20, ageF20] = inputCount(20, femaleID); %[fileCount, AGE] = inputCount(AGE)
[fileCountM40, ageM40] = inputCount(40, maleID); %[fileCount, AGE] = inputCount(AGE)
[fileCountF40, ageF40] = inputCount(40, femaleID); %[fileCount, AGE] = inputCount(AGE)
[fileCountM60, ageM60] = inputCount(60, maleID); %[fileCount, AGE] = inputCount(AGE)
[fileCountF60, ageF60] = inputCount(60, femaleID); %[fileCount, AGE] = inputCount(AGE)

ages = [ageM20 ageF20 ageM40 ageF40 ageM60 ageF60];
fileCounts = [fileCountM20 fileCountF20 fileCountM40 fileCountF40 ... 
    fileCountM60 fileCountF60];

% [fileCountMG1, ageMG1] = inputCount2(20, maleID, gesture1ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountFG1, ageFG1] = inputCount2(20, femaleID, gesture1ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountMG2, ageMG2] = inputCount2(20, maleID, gesture2ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountFG2, ageFG2] = inputCount2(20, femaleID, gesture2ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountMG3, ageMG3] = inputCount2(20, maleID, gesture3ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountFG3, ageFG3] = inputCount2(20, femaleID, gesture3ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountMG4, ageMG4] = inputCount2(20, maleID, gesture4ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountFG4, ageFG4] = inputCount2(20, femaleID, gesture4ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountMG5, ageMG5] = inputCount2(20, maleID, gesture5ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountFG5, ageFG5] = inputCount2(20, femaleID, gesture5ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountMG6, ageMG6] = inputCount2(20, maleID, gesture6ID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountFG6, ageFG6] = inputCount2(20, femaleID, gesture6ID); %[fileCount, AGE] = inputCount(AGE)


%% Network selection

promptNetSelect = 'Which process do you want to continue? (0: PCA, 1:reliefF, 2:Pattern Net, 3:LSTM): ';
netSelectAns = input(promptNetSelect);
    
switch netSelectAns 
    case 0
        NETSELECT = NETSELECT + 0;
    case 1
        NETSELECT = NETSELECT + 1;
    case 2
        NETSELECT = NETSELECT + 2;
    case 3
        NETSELECT = NETSELECT + 3;
    otherwise
        fprintf('You inserted a wrong number.\n');
end

if(NETSELECT == 0)
%% Principle Component Analysis
    [pcaMat, numOfK, varRetain] = pcaResult(ages, fileCounts);
      
    fprintf("\n");
    prompt1 = 'Will you continue? (y/n): ';
    continueAns = input(prompt1, 's');
    if ~isempty(continueAns)
        if continueAns == 'y'
            codeInitialize;
            fprintf("Okay!\n\n");
            prompt2 = "Will you continue to train neural network?\n" + ...
               "If so, choose '2'(CNN) or '3'(LSTM): ";
            NETSELECT = input(prompt2);
            if NETSELECT ~= 1 && NETSELECT ~= 2
                fprintf("You inserted a wrong number.\n");
            end
            fprintf("\n");
        elseif continueAns == 'n'
            fprintf("See ya!\n");
        else
            fprintf("You inserted a wrong letter.\n");
        end
    end

end

if(NETSELECT == 1)
%% reliefF for feature reduction
    [ranks, weights] = relieffResult(A1, B1, maleID, femaleID, ...
    ages, fileCounts);
    
    fprintf("\n");
    prompt1 = 'Will you continue? (y/n): ';
    continueAns = input(prompt1, 's');
    if ~isempty(continueAns)
        if continueAns == 'y'
            fprintf("Okay!\n\n");
            prompt2 = "Will you continue to train neural network?\n" + ...
               "If so, choose '2'(CNN) or '3'(LSTM): ";
            NETSELECT = input(prompt2);
            if NETSELECT ~= 1 && NETSELECT ~= 2
                fprintf("You inserted a wrong number.\n");
            end
            fprintf("\n");
        elseif continueAns == 'n'
            fprintf("See ya!\n");
        else
            fprintf("You inserted a wrong letter.\n");
        end
    end
    
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

%     featuresM20Cell = {};
%     reducedFeaturesM20Cell = {};
%     featuresF20Cell = {};
%     reducedFeaturesF20Cell = {};
%     featuresM40Cell = {};
%     reducedFeaturesM40Cell = {};
%     featuresF40Cell = {};
%     reducedFeaturesF40Cell = {};
%     featuresM60Cell = {};
%     reducedFeaturesM60Cell = {};
%     featuresF60Cell = {};
%     reducedFeaturesF60Cell = {};
% 
%     %Subjects in 20s
%     for nM20 = 1 : fileCountM20
%         if nM20 == 1
%             [featuresM20Data, labelsM20Data] = ...
%                 featureExtractionToCell(B1, A1, ageM20, nM20, maleID); ...
%                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresM20Cell = {featuresM20Data};
%             %reducedFeatures20Cell = {reducedFeatures20Data};
%             labelsM20Cell = labelsM20Data;
%             continue;
%         else
%             [featuresM20Data, labelsM20Data] = ...
%                 featureExtractionToCell(B1, A1, ageM20, nM20, maleID); ...
%                  %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresM20Cell = [featuresM20Cell; {featuresM20Data}];
%             %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
%             labelsM20Cell = [labelsM20Cell; labelsM20Data];
%         end
%     end
% 
%     for nF20 = 1 : fileCountF20
%         if nF20 == 1
%             [featuresF20Data, labelsF20Data] = ...
%                 featureExtractionToCell(B1, A1, ageF20, nF20, femaleID); ...
%                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresF20Cell = {featuresF20Data};
%             %reducedFeatures20Cell = {reducedFeatures20Data};
%             labelsF20Cell = labelsF20Data;
%             continue;
%         else
%             [featuresF20Data, labelsF20Data] = ...
%                 featureExtractionToCell(B1, A1, ageF20, nF20, femaleID); ...
%                  %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresF20Cell = [featuresF20Cell; {featuresF20Data}];
%             %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
%             labelsF20Cell = [labelsF20Cell; labelsF20Data];
%         end
%     end
%     
%     %Subjects in 40s
%     for nM40 = 1 : fileCountM40
%         if nM40 == 1
%             [featuresM40Data, labelsM40Data] = ...
%                 featureExtractionToCell(B1, A1, ageM40, nM40, maleID); ...
%                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresM40Cell = {featuresM40Data};
%             %reducedFeatures40Cell = {reducedFeatures40Data};
%             labelsM40Cell = labelsM40Data;
%             continue;
%         else
%             [featuresM40Data, labelsM40Data] = ...
%                 featureExtractionToCell(B1, A1, ageM40, nM40, maleID); ...
%                  %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresM40Cell = [featuresM40Cell; {featuresM40Data}];
%             %reducedFeatures40Cell = [reducedFeatures40Cell; {reducedFeatures40Data}];
%             labelsM40Cell = [labelsM40Cell; labelsM40Data];
%         end
%     end
%     
%     for nF40 = 1 : fileCountF40
%         if nF40 == 1
%             [featuresF40Data, labelsF40Data] = ...
%                 featureExtractionToCell(B1, A1, ageF40, nF40, femaleID); ...
%                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresF40Cell = {featuresF40Data};
%             %reducedFeatures40Cell = {reducedFeatures40Data};
%             labelsF40Cell = labelsF40Data;
%             continue;
%         else
%             [featuresF40Data, labelsF40Data] = ...
%                 featureExtractionToCell(B1, A1, ageF40, nF40, femaleID); ...
%                  %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresF40Cell = [featuresF40Cell; {featuresF40Data}];
%             %reducedFeatures40Cell = [reducedFeatures40Cell; {reducedFeatures40Data}];
%             labelsF40Cell = [labelsF40Cell; labelsF40Data];
%         end
%     end
%     
%     %Subjects in 60s
%     for nM60 = 1 : fileCountM60
%         if nM60 == 1
%             [featuresM60Data, labelsM60Data] = ...
%                 featureExtractionToCell(B1, A1, ageM60, nM60, maleID); ...
%                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresM60Cell = {featuresM60Data};
%             %reducedFeatures60Cell = {reducedFeatures60Data};
%             labelsM60Cell = labelsM60Data;
%             continue;
%         else
%             [featuresM60Data, labelsM60Data] = ...
%                 featureExtractionToCell(B1, A1, ageM60, nM60, maleID); ...
%                  %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresM60Cell = [featuresM60Cell; {featuresM60Data}];
%             %reducedFeatures60Cell = [reducedFeatures60Cell; {reducedFeatures60Data}];
%             labelsM60Cell = [labelsM60Cell; labelsM60Data];
%         end
%     end
%     
%     for nF60 = 1 : fileCountF60
%         if nF60 == 1
%             [featuresF60Data, labelsF60Data] = ...
%                 featureExtractionToCell(B1, A1, ageF60, nF60, femaleID); ...
%                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresF60Cell = {featuresF60Data};
%             %reducedFeatures60Cell = {reducedFeatures60Data};
%             labelsF60Cell = labelsF60Data;
%             continue;
%         else
%             [featuresF60Data, labelsF60Data] = ...
%                 featureExtractionToCell(B1, A1, ageF60, nF60, femaleID); ...
%                  %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
%             featuresF60Cell = [featuresF60Cell; {featuresF60Data}];
%             %reducedFeatures60Cell = [reducedFeatures60Cell; {reducedFeatures60Data}];
%             labelsF60Cell = [labelsF60Cell; labelsF60Data];
%         end
%     end
%     
%     %Subjects in total
%     featuresTotalCell = [featuresM20Cell; featuresF20Cell;...
%         featuresM40Cell; featuresF40Cell;...
%         featuresM60Cell; featuresF60Cell]; %Features from 20s + Features from 60s
%     %reducedFeaturesTotalCell = [reducedFeatures20Cell; ...
%     %    reducedFeatures40Cell; ...
%     %    reducedFeatures60Cell];
%     labelsTotalCell = [labelsM20Cell; labelsF20Cell;... 
%         labelsM40Cell; labelsF40Cell;...
%         labelsM60Cell; labelsF60Cell];

    featuresMG1Cell = {};
    reducedFeaturesM20Cell = {};
    featuresFG1Cell = {};
    reducedFeaturesF20Cell = {};
    featuresMG2Cell = {};
    reducedFeaturesM40Cell = {};
    featuresFG2Cell = {};
    reducedFeaturesF40Cell = {};
    featuresMG3Cell = {};
    reducedFeaturesM60Cell = {};
    featuresFG3Cell = {};
    reducedFeaturesF60Cell = {};
    featuresMG4Cell = {};
    reducedFeaturesM20Cell = {};
    featuresFG4Cell = {};
    reducedFeaturesF20Cell = {};
    featuresMG5Cell = {};
    reducedFeaturesM40Cell = {};
    featuresFG5Cell = {};
    reducedFeaturesF40Cell = {};
    featuresMG6Cell = {};
    reducedFeaturesM60Cell = {};
    featuresFG6Cell = {};
    reducedFeaturesF60Cell = {};

    %Subjects in 20s
    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG1Data, labelsMG1Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG1, nM20, maleID, iter, gesture1ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG1Cell = {featuresMG1Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG1Cell = labelsMG1Data;
                continue;
            else
                [featuresMG1Data, labelsMG1Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG1, nM20, maleID, iter, gesture1ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG1Cell = [featuresMG1Cell; {featuresMG1Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG1Cell = [labelsMG1Cell; labelsMG1Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG1Data, labelsFG1Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG1, nF20, femaleID, iter, gesture1ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG1Cell = {featuresFG1Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG1Cell = labelsFG1Data;
                continue;
            else
                [featuresFG1Data, labelsFG1Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG1, nF20, femaleID, iter, gesture1ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG1Cell = [featuresFG1Cell; {featuresFG1Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG1Cell = [labelsFG1Cell; labelsFG1Data];
            end
        end
    end

     for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG2Data, labelsMG2Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG2, nM20, maleID, iter, gesture2ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG2Cell = {featuresMG2Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG2Cell = labelsMG2Data;
                continue;
            else
                [featuresMG2Data, labelsMG2Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG2, nM20, maleID, iter, gesture2ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG2Cell = [featuresMG2Cell; {featuresMG2Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG2Cell = [labelsMG2Cell; labelsMG2Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG2Data, labelsFG2Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG2, nF20, femaleID, iter, gesture2ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG2Cell = {featuresFG2Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG2Cell = labelsFG2Data;
                continue;
            else
                [featuresFG2Data, labelsFG2Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG2, nF20, femaleID, iter, gesture2ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG2Cell = [featuresFG2Cell; {featuresFG2Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG2Cell = [labelsFG2Cell; labelsFG2Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG3Data, labelsMG3Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG3, nM20, maleID, iter, gesture3ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG3Cell = {featuresMG3Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG3Cell = labelsMG3Data;
                continue;
            else
                [featuresMG3Data, labelsMG3Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG3, nM20, maleID, iter, gesture3ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG3Cell = [featuresMG3Cell; {featuresMG3Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG3Cell = [labelsMG3Cell; labelsMG3Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG3Data, labelsFG3Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG3, nF20, femaleID, iter, gesture3ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG3Cell = {featuresFG3Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG3Cell = labelsFG3Data;
                continue;
            else
                [featuresFG3Data, labelsFG3Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG3, nF20, femaleID, iter, gesture3ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG3Cell = [featuresFG3Cell; {featuresFG3Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG3Cell = [labelsFG3Cell; labelsFG3Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG4Data, labelsMG4Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG4, nM20, maleID, iter, gesture4ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG4Cell = {featuresMG4Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG4Cell = labelsMG4Data;
                continue;
            else
                [featuresMG4Data, labelsMG4Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG4, nM20, maleID, iter, gesture4ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG4Cell = [featuresMG4Cell; {featuresMG4Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG4Cell = [labelsMG4Cell; labelsMG4Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG4Data, labelsFG4Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG4, nF20, femaleID, iter, gesture4ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG4Cell = {featuresFG4Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG4Cell = labelsFG4Data;
                continue;
            else
                [featuresFG4Data, labelsFG4Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG4, nF20, femaleID, iter, gesture4ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG4Cell = [featuresFG4Cell; {featuresFG4Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG4Cell = [labelsFG4Cell; labelsFG4Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG5Data, labelsMG5Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG5, nM20, maleID, iter, gesture5ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG5Cell = {featuresMG5Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG5Cell = labelsMG5Data;
                continue;
            else
                [featuresMG5Data, labelsMG5Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG5, nM20, maleID, iter, gesture5ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG5Cell = [featuresMG5Cell; {featuresMG5Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG5Cell = [labelsMG5Cell; labelsMG5Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG5Data, labelsFG5Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG5, nF20, femaleID, iter, gesture5ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG5Cell = {featuresFG5Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG5Cell = labelsFG5Data;
                continue;
            else
                [featuresFG5Data, labelsFG5Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG5, nF20, femaleID, iter, gesture5ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG5Cell = [featuresFG5Cell; {featuresFG5Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG5Cell = [labelsFG5Cell; labelsFG5Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG6Data, labelsMG6Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG6, nM20, maleID, iter, gesture6ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG6Cell = {featuresMG6Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG6Cell = labelsMG6Data;
                continue;
            else
                [featuresMG6Data, labelsMG6Data] = ...
                    featureExtractionToCell2(B1, A1, ageMG6, nM20, maleID, iter, gesture6ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG6Cell = [featuresMG6Cell; {featuresMG6Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG6Cell = [labelsMG6Cell; labelsMG6Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3 
            if nF20 == 1 && iter == 1
                [featuresFG6Data, labelsFG6Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG6, nF20, femaleID, iter, gesture6ID); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG6Cell = {featuresFG6Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG6Cell = labelsFG6Data;
                continue;
            else
                [featuresFG6Data, labelsFG6Data] = ...
                    featureExtractionToCell2(B1, A1, ageFG6, nF20, femaleID, iter, gesture6ID); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG6Cell = [featuresFG6Cell; {featuresFG6Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG6Cell = [labelsFG6Cell; labelsFG6Data];
            end
        end
    end

    %Subjects in total
    featuresTotalCell = [featuresMG1Cell; featuresFG1Cell;...
        featuresMG2Cell; featuresFG2Cell;...
        featuresMG3Cell; featuresFG3Cell;...
        featuresMG4Cell; featuresFG4Cell;...
        featuresMG5Cell; featuresFG5Cell;...
        featuresMG6Cell; featuresFG6Cell]; %Features from 20s + Features from 60s
    %reducedFeaturesTotalCell = [reducedFeatures20Cell; ...
    %    reducedFeatures40Cell; ...
    %    reducedFeatures60Cell];
    labelsTotalCell = [labelsMG1Cell; labelsFG1Cell;... 
        labelsMG2Cell; labelsFG2Cell;...
        labelsMG3Cell; labelsFG3Cell;...
        labelsMG4Cell; labelsFG4Cell;...
        labelsMG5Cell; labelsFG5Cell;...
        labelsMG6Cell; labelsFG6Cell];



%% Signal analyzing

    %signalAnalyzer;
    
%% Long short-term memory network training

    [net] = trainLstm(featuresTotalCell, ...
        labelsTotalCell); %LSTM network
    
end