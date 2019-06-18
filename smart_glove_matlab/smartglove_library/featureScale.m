function [scaledFeature] = featureScale(inputColumn)
%% Feature scaling

meanColumn = mean(inputColumn, 1); %Mean
stdColumn = std(inputColumn, 0, 1); %Standard deviation

scaledFeature = (inputColumn - meanColumn) / stdColumn; %(Variables - Mean) / Std

end