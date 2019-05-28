%% Code for Imagine x Lab artificial intelligence system for smart glove: MANOVIVO
%% Include smartglove function library
addpath('smartglove_library');

%% Initialization
code_initialize;

%% 5 Hz low-pass filter
fs = 100; %Sample frequency 100Hz
[b1, a1] = butterworth_5hz(fs); % =[b1,a1] =butterworth_5hz(fs): 5Hz low-pass filter

%% Smart glove data read & feature extraction

%Subjects in 20s
[file_count_20, age_20] = input_count(20); %[file_count, age] = input_count(age)
[features_20, labels_20] = feature_extraction(b1, a1, age_20, file_count_20); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction

%Subjects in 60s
[file_count_60, age_60] = input_count(60); %[file_count, age] = input_count(age)
[features_60, labels_60] = feature_extraction(b1, a1, age_60, file_count_60); %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction

%Subjects in total
features_total = [features_20; features_60]; %Features from 20s + Features from 60s
labels_total = [labels_20; labels_60]; %


%% Signal analyzing

%signalAnalyzer;

%% Convolution neural network training

[net, tr] = train_cnn(features_total, labels_total); %Pattern network

%% Long short-term memory network training

%[net] = train_lstm(features_total, labels_total); %LSTM network
