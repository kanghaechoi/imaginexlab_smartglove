function [scaledFeature] = featureScale(inputColumn)
%% Feature scaling

meanColumn = mean(inputColumn, 1); %Mean of input variables
stdColumn = std(inputColumn, 0, 1); %Standard deviation of input variables

scaledFeature = (inputColumn - meanColumn) / stdColumn; %(Matrix of input variables - Mean) / Std

end