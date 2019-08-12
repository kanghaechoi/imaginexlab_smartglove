function [fileCount, AGE] = exp2InputCount(AGE, GENDER, gestureID)
%% Input txt file number count

for subjectNum = 1 : 5
    dataList = dir(sprintf('*Hand_IMU_%d_%d_%d_0%d*.txt', AGE, GENDER, subjectNum, gestureID));
end

fileCount =(numel(dataList));

end

