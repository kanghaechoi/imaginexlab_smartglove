function [fileCount, AGE] = inputCount2(AGE, GENDER, gestureID)
%% Input txt file number count

dataList = dir(sprintf('*Hand_IMU_%d_%d_*_%d_*.txt', AGE, GENDER, gestureID));
fileCount =(numel(dataList));

end

