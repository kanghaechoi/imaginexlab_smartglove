function smartgloveRuntime(NUMBER)
%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO

%% Initialization
%codeInitialize;
Value = NUMBER;
%readFolders()

%% 5 Hz low-pass filter
SAMPLE_FREQ = single(100); %Sample frequency 100Hz
[B1, A1] = butterworth5hz(SAMPLE_FREQ); % [B1,A1] =butterworth5hz(FS): 5Hz low-pass filter

%% File count & Age extraction 
maleID = 1;
femaleID = 2;
genderID = [maleID femaleID];

gestureID = [01 02 03 04 05 06];

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

% ages = 20;
% fileCounts = [fileCountMG1 fileCountFG1 fileCountMG2 fileCountFG2 ... 
%     fileCountMG3 fileCountFG3 fileCountMG4 fileCountFG4 ...
%     fileCountMG5 fileCountFG5 fileCountMG6 fileCountFG6];

%% Smart glove data read & feature extraction

    %Results from experiment 1
    [featuresTotalCell, labelsTotalCell] = ... 
        exp1LstmResultRuntime(A1, B1, ages, fileCounts, genderID, Value); 
    
    %Results from experiment 2
    %[featuresTotalCell, labelsTotalCell] = ... 
    %    exp2LstmResult(A1, B1, ages, fileCounts, genderID); 

%% Signal analyzing
    %Features = cell2mat(featuresTotalCell) ; 
    %signalAnalyzer;
    
%% Long short-term memory network training

[net] = trainLstm(featuresTotalCell, ...
        labelsTotalCell,Value); %LSTM network
 
end

