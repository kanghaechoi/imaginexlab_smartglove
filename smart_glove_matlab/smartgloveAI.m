%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO

%% Include smartglove function library
addpath('smartglove_library', 'exp1_data', 'exp2_data');

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

gestureID = [01 02 03 04 05 06];

% [fileCountM20, ageM20] = inputCount(20, maleID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountF20, ageF20] = inputCount(20, femaleID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountM40, ageM40] = inputCount(40, maleID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountF40, ageF40] = inputCount(40, femaleID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountM60, ageM60] = inputCount(60, maleID); %[fileCount, AGE] = inputCount(AGE)
% [fileCountF60, ageF60] = inputCount(60, femaleID); %[fileCount, AGE] = inputCount(AGE)
% 
% ages = [ageM20 ageF20 ageM40 ageF40 ageM60 ageF60];
% fileCounts = [fileCountM20 fileCountF20 fileCountM40 fileCountF40 ... 
%     fileCountM60 fileCountF60];

[fileCountM, ~] = exp2InputCount(20, maleID, gestureID(1,6)); %[fileCount, AGE] = inputCount(AGE)
[fileCountF, ~] = exp2InputCount(20, femaleID, gestureID(1,6)); %[fileCount, AGE] = inputCount(AGE)

ages = 20;
fileCounts = [fileCountM fileCountF];

%% Network selection

promptNetSelect = ... 
    'Which process do you want to continue?\n(1:reliefF, 2:LSTM_Exp1, 3:LSTM_Exp2): ';
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
    case 4
        NETSELECT = NETSELECT + 4;
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
    %Experiment 1
%     [ranks, weights] = relieffResult(A1, B1, maleID, femaleID, ...
%     ages, fileCounts);
    
    %Experiment 2
    [ranks, weights] = relieffResultExp2(A1, B1, maleID, femaleID, ...
    ages, gestureID);
    
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

    %Results from experiment 1
    [featuresTotalCell, labelsTotalCell] = ... 
        exp1LstmResult(A1, B1, ages, fileCounts, genderID); 

%% Signal analyzing
    %Features = cell2mat(featuresTotalCell) ; 
    %signalAnalyzer;
    
%% Long short-term memory network training

   [exp1Net] = exp1TrainLstm(featuresTotalCell, ...
        labelsTotalCell); %LSTM network

end

if(NETSELECT == 3)
%% Smart glove data read & feature extraction

    %Results from experiment 2
    [featuresTotalCell, labelsTotalCell] = ... 
        exp2LstmResult(A1, B1, ages, genderID, gestureID); 

%% Signal analyzing
    %Features = cell2mat(featuresTotalCell) ; 
    %signalAnalyzer;
    
%% Long short-term memory network training

   [exp2Net] = exp2TrainLstm(featuresTotalCell, ...
        labelsTotalCell); %LSTM network
    
end