%% Smartglove LSTM network training automation process

%% Include smartglove function library
addpath('smartglove_library', 'exp1_data', 'exp2_data');

%% Initialization
codeInitialize;

%% Directory path modification
path = "\exp1_data\";

%% Network training iteration
MAXFEATURES = 26; %The number of features
NUMBEROFITERATION = 5; %The number of training iteration per feature

for numberOfFeature = 1 : MAXFEATURES
    for loop = 1 : NUMBEROFITERATION
        logFileName = strcat('LSTM_', int2str(numberOfFeature),"_", int2str(loop), '.log');
        f = fullfile('training_log',logFileName);
        fid =  fopen(f, 'w');
        fclose(fid);
        diary(f);
        smartgloveRuntime(numberOfFeature)
    end
end

logFileName = strcat('LSTMFINISH_','.log');
f = fullfile('training_log',logFileName);
fid = fopen(f, 'w');
fclose(fid);
diary(f);