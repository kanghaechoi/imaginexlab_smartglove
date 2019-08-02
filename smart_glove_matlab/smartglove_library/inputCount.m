function [fileCount, AGE] = inputCount(AGE, GENDER)
%% Input txt file number count

dataList = dir(sprintf('*Hand_IMU_%d_%d*.txt', AGE, GENDER));
fileCount =(numel(dataList));

end

