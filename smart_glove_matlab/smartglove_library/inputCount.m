function [fileCount, AGE] = inputCount(AGE)
%% Input txt file number count

dataList = dir(sprintf('*Hand_IMU_%d*.txt', AGE));
fileCount =(numel(dataList));

end

