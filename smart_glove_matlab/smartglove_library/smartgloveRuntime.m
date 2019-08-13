function smartgloveRuntime(NUMBER)
%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO

%% Initialization
%codeInitialize;
numberOfFeature = NUMBER;
%readFolders()

%% 5 Hz low-pass filter
SAMPLE_FREQ = single(100); %Sample frequency 100Hz
[B1, A1] = butterworth5hz(SAMPLE_FREQ); % [B1,A1] =butterworth5hz(FS): 5Hz low-pass filter

%% File count & Age extraction 
maleID = 1;
femaleID = 2;
genderID = [maleID femaleID];

gestureID = [01 02 03 04 05 06];

%[fileCountM20, ageM20] = inputCount(20, maleID); %[fileCount, AGE] = inputCount(AGE)
%[fileCountF20, ageF20] = inputCount(20, femaleID); %[fileCount, AGE] = inputCount(AGE)
%[fileCountM40, ageM40] = inputCount(40, maleID); %[fileCount, AGE] = inputCount(AGE)
%[fileCountF40, ageF40] = inputCount(40, femaleID); %[fileCount, AGE] = inputCount(AGE)
%[fileCountM60, ageM60] = inputCount(60, maleID); %[fileCount, AGE] = inputCount(AGE)
%[fileCountF60, ageF60] = inputCount(60, femaleID); %[fileCount, AGE] = inputCount(AGE)

%ages = [ageM20 ageF20 ageM40 ageF40 ageM60 ageF60];
%fileCounts = [fileCountM20 fileCountF20 fileCountM40 fileCountF40 ... 
%    fileCountM60 fileCountF60];

[fileCountM, ~] = exp2InputCount(20, maleID, gestureID(1,4)); %[fileCount, AGE] = inputCount(AGE)
[fileCountF, ~] = exp2InputCount(20, femaleID, gestureID(1,4)); %[fileCount, AGE] = inputCount(AGE)

ages = 20;
fileCounts = [fileCountM fileCountF];

%% Long short-term memory network training (Loop)

    %Results from experiment 1, Question 1
    %[featuresTotalCell, labelsTotalCell] = ... 
    %    exp1Q1LstmResultLoop(A1, B1, ages, fileCounts, genderID, numberOfFeature); 
    %Experiment 1, Question 1 LSTM network
    %[exp1Q1Net] = exp1Q1TrainLstmLoop(featuresTotalCell, ...
    %    labelsTotalCell, numberOfFeature);
    
    %Results from experiment 1, Question 2
    %[featuresTotalCell, labelsTotalCell] = ... 
    %    exp1Q2LstmResultLoop(A1, B1, ages,genderID, numberOfFeature); 
    %Experiment 1, Question 2 LSTM network
    %[exp1Q2Net] = exp1Q2TrainLstmLoop(featuresTotalCell, ...
    %    labelsTotalCell, numberOfFeature);    
    
    %Results from experiment 2
    [featuresTotalCell, labelsTotalCell] = ... 
        exp2LstmResultLoop(A1, B1, ages, genderID, gestureID, numberOfFeature);
    %Experiment 2 LSTM network
    [exp2Net] = exp2TrainLstmLoop(featuresTotalCell, ...
        labelsTotalCell, numberOfFeature);

end

