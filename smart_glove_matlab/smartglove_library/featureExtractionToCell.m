function [features, labels] = featureExtractionToCell(B1, A1, AGE, fileCount)
%% Feature extraction

for n = 1:fileCount

    if n == 1
        %hand_data = ones(1, 12);
        %wrist_data = ones(1, 6);
         
         handRead = dlmread(sprintf('Hand_IMU_%d_%d.txt', AGE, n)); %Read hand data
         wristRead = dlmread(sprintf('Wrist_IMU_%d_%d.txt', AGE, n)); %Read wrist
         handData = {handRead};
         wristData = {wristRead};
        continue;
    else  
        handRead = dlmread(sprintf('Hand_IMU_%d_%d.txt', AGE, n)); %Read hand data
        wristRead = dlmread(sprintf('Wrist_IMU_%d_%d.txt', AGE, n)); %Read wrist
        handData= [handData; {handRead}];
        wristData= [wristData; {wristRead}];
    end
end 

handData(1:length(handData));
wristData(1:length(wristData));

m = size(handData, 1);
features = {};

for k = 1:m
    
%Feature matrix length definition
if length(handData{k}) >= length(wristData{k}), featureLength = length(wristData{k});
elseif length(handData{k}) < length(wristData{k}), featureLength = length(handData{k});
end
    
thumbAcc = handData{k}(:,7:9); %Thumb acceleration (X, Y, Z)
indexAcc = handData{k}(:,10:12); %Index acceleration (X, Y, Z)
wristAcc = wristData{k}(:,4:6); %Wrist acceleration (X, Y, Z)

thumbAccMag = sum((thumbAcc .* thumbAcc),2); %Thumb acc magnitude + Downsizing
indexAccMag = sum((indexAcc .* indexAcc),2); %Index acc magnitude + Downsizing
wristAccMag = sum((wristAcc .* wristAcc),2); %Wrist acc magnitude + Downsizing

%Finger Euler angle
thumbAngleX = handData{k}(:,4); %Thumb x-axis Euler angle
indexAngleX = handData{k}(:,5); %Index x-axis Euler angle

%Hand Euler angle
handAngleX = handData{k}(:,1); %Hand x-axis Euler angle
handAngleY = handData{k}(:,2); %Hand y-axis Euler angle
handAngleZ = handData{k}(:,3); %Hand z-axis Euler angle

%Wrist Euler angle
wristAngleX = wristData{k}(:,1); %Wrist x-axis Euler angle
wristAngleY = wristData{k}(:,2); %Wrist y-axis Euler angle
wristAngleZ = wristData{k}(:,3); %Wrist z-axis Euler angle

%% Feature scaling

%Acceleration
scaledThumbAccMag = featureScale(thumbAccMag);
scaledIndexAccMag = featureScale(indexAccMag);
scaledWristAccMag = featureScale(wristAccMag);

%Euler angle
%Degree to Radian
% scaledThumbAngleX = deg2rad(thumbAngleX);
% scaledIndexAngleX = deg2rad(indexAngleX);
% scaledHandAngleX = deg2rad(handAngleX);
% scaledHandAngleY = deg2rad(handAngleY);
% scaledHandAngleZ = deg2rad(handAngleZ);
% scaledWristAngleX = deg2rad(wristAngleX);
% scaledWristAngleY = deg2rad(wristAngleY);
% scaledWristAngleZ = deg2rad(wristAngleZ);

scaledThumbAngleX = featureScale(deg2rad(thumbAngleX));
scaledIndexAngleX = featureScale(deg2rad(indexAngleX));
scaledHandAngleX = featureScale(deg2rad(handAngleX));
scaledHandAngleY = featureScale(deg2rad(handAngleY));
scaledHandAngleZ = featureScale(deg2rad(handAngleZ));
scaledWristAngleX = featureScale(deg2rad(wristAngleX));
scaledWristAngleY = featureScale(deg2rad(wristAngleY));
scaledWristAngleZ = featureScale(deg2rad(wristAngleZ));

%% Data filtering

%Acceleration
zplThumbAccMag = (filtfilt(B1,A1,scaledThumbAccMag'))'; %Zero-phase filtering + Matrix transpose
zplIndexAccMag = (filtfilt(B1,A1,scaledIndexAccMag'))'; %Zero-phase filtering + Matrix transpose
zplWristAccMag = (filtfilt(B1,A1,scaledWristAccMag'))'; %Zero-phase filtering + Matrix transpose

%Euler angle
zplThumbAngleX = (filtfilt(B1,A1,scaledThumbAngleX'))'; %Zero-phase filtering + Matrix transpose
zplIndexAngleX = (filtfilt(B1,A1,scaledIndexAngleX'))'; %Zero-phase filtering + Matrix transpose
zplHandAngleX = (filtfilt(B1,A1,scaledHandAngleX'))'; %Zero-phase filtering + Matrix transpose
zplHandAngleY = (filtfilt(B1,A1,scaledHandAngleY'))'; %Zero-phase filtering + Matrix transpose
zplHandAngleZ = (filtfilt(B1,A1,scaledHandAngleZ'))';%Zero-phase filtering + Matrix transpose
zplWristAngleX = (filtfilt(B1,A1,scaledWristAngleX'))'; %Zero-phase filtering + Matrix transpose
zplWristAngleY = (filtfilt(B1,A1,scaledWristAngleY'))'; %Zero-phase filtering + Matrix transpose
zplWristAngleZ = (filtfilt(B1,A1,scaledWristAngleZ'))'; %Zero-phase filtering + Matrix transpose

%% Data resizing

%Acceleration
resizedThumbAccMag = zplThumbAccMag(1:featureLength,1);
resizedIndexAccMag = zplIndexAccMag(1:featureLength,1);
resizedWristAccMag = zplWristAccMag(1:featureLength,1);

%Euler angle
resizedThumbAngleX = zplThumbAngleX(1:featureLength,1);
resizedIndexAngleX = zplIndexAngleX(1:featureLength,1);
resizedHandAngleX = zplHandAngleX(1:featureLength,1);
resizedHandAngleY = zplHandAngleY(1:featureLength,1);
resizedHandAngleZ = zplHandAngleZ(1:featureLength,1);
resizedWristAngleX = zplWristAngleX(1:featureLength,1);
resizedWristAngleY = zplWristAngleY(1:featureLength,1);
resizedWristAngleZ = zplWristAngleZ(1:featureLength,1);

%% Create feature columns (feature_length x 11)

featuresData = [resizedThumbAccMag, resizedIndexAccMag, resizedWristAccMag, resizedThumbAngleX, resizedIndexAngleX, ...
    resizedHandAngleX, resizedHandAngleY, resizedHandAngleZ, resizedWristAngleX, resizedWristAngleY, resizedWristAngleZ]';

features= [features; {featuresData}];
end

%features(1:length(features));

%% Create a label column (number of files, so each roz of the cell array x 1) 

labels = ones(m, 1);
labels = (AGE * labels) / 10;

end