%% Smartglove LSTM network training automation process

%% Include smartglove function library
addpath('smartglove_library');

%% Initialization
codeInitialize;

%% Directory path modification
path = "\training_log\";

%% Network training iteration
MAXFEATURES = 26; %The number of features
NUMBEROFITERATION = 5; %The number of training iteration per feature

for NumberFeature = 1 : MAXFEATURES
    for loop = 1 : NUMBEROFITERATION
        logFileName = strcat('LSTM_', int2str(NumberFeature),"_", int2str(loop), '.log');
        f = fullfile('training_log',logFileName);
        fid =  fopen(f, 'w');
        fclose(fid);
        diary(f);
        smartgloveRuntime(NumberFeature)
    end
end

logFileName = strcat('LSTMFINISH_','.log');
f = fullfile('training_log',logFileName);
fid = fopen(f, 'w');
fclose(fid);
diary(f);