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
genderID = [maleID femaleID];

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
            if NETSELECT ~= 2 && NETSELECT ~= 3
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
            if NETSELECT ~= 2 && NETSELECT ~= 3
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

     %signalAnalyzer;

%% Convolutional neural network training

    %[net, tr] = trainCnn(featuresTotal, labelsTotal); %Pattern network

end

if(NETSELECT == 3)
%% Smart glove data read & feature extraction

    %Results from experiment 1
    [featuresTotalCell, labelsTotalCell] = exp1LstmResult(A1, B1, ages, fileCounts, genderID); 
    
    %Results from experiment 2
    %[featuresTotalCell, labelsTotalCell] = exp2LstmResult(A1, B1, ages, fileCounts, genderID); 



%% Signal analyzing
    %Features = cell2mat(featuresTotalCell) ; 
    %signalAnalyzer;
    
%% Long short-term memory network training

   [ net] = trainLstm(featuresTotalCell, ...
        labelsTotalCell); %LSTM network
    
end