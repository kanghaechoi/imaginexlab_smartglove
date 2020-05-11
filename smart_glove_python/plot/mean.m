clear;
clc;

data = readmatrix('./resnet152/q2/0/result.txt');

data_ = data(:, 2:end);

data_mean = sum(data_, 2)/5;