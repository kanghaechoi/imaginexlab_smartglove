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

%Velocity
thumbVel = cumtrapz(thumbAcc); %Thumb velocity (X, Y, Z)
indexVel = cumtrapz(indexAcc); %Index velocity (X, Y, Z)
wristVel = cumtrapz(wristAcc); %Wrist velocity (X, Y, Z)

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
scaledThumbAccX = featureScale(thumbAcc(:,1));
scaledThumbAccY = featureScale(thumbAcc(:,2));
scaledThumbAccZ = featureScale(thumbAcc(:,3));
scaledIndexAccX = featureScale(indexAcc(:,1));
scaledIndexAccY = featureScale(indexAcc(:,2));
scaledIndexAccZ = featureScale(indexAcc(:,3));
scaledWristAccX = featureScale(wristAcc(:,1));
scaledWristAccY = featureScale(wristAcc(:,2));
scaledWristAccZ = featureScale(wristAcc(:,3));

%Velocity
scaledThumbVelX = featureScale(thumbVel(:,1));
scaledThumbVelY = featureScale(thumbVel(:,2));
scaledThumbVelZ = featureScale(thumbVel(:,3));
scaledIndexVelX = featureScale(indexVel(:,1));
scaledIndexVelY = featureScale(indexVel(:,2));
scaledIndexVelZ = featureScale(indexVel(:,3));
scaledWristVelX = featureScale(wristVel(:,1));
scaledWristVelY = featureScale(wristVel(:,2));
scaledWristVelZ = featureScale(wristVel(:,3));

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

% %Acceleration
% filtThumbAccMag = (filtfilt(B1,A1,scaledThumbAccMag'))'; %Zero-phase filtering + Matrix transpose
% filtIndexAccMag = (filtfilt(B1,A1,scaledIndexAccMag'))'; %Zero-phase filtering + Matrix transpose
% filtWristAccMag = (filtfilt(B1,A1,scaledWristAccMag'))'; %Zero-phase filtering + Matrix transpose
% 
% %Euler angle
% filtThumbAngleX = (filtfilt(B1,A1,scaledThumbAngleX'))'; %Zero-phase filtering + Matrix transpose
% filtIndexAngleX = (filtfilt(B1,A1,scaledIndexAngleX'))'; %Zero-phase filtering + Matrix transpose
% filtHandAngleX = (filtfilt(B1,A1,scaledHandAngleX'))'; %Zero-phase filtering + Matrix transpose
% filtHandAngleY = (filtfilt(B1,A1,scaledHandAngleY'))'; %Zero-phase filtering + Matrix transpose
% filtHandAngleZ = (filtfilt(B1,A1,scaledHandAngleZ'))';%Zero-phase filtering + Matrix transpose
% filtWristAngleX = (filtfilt(B1,A1,scaledWristAngleX'))'; %Zero-phase filtering + Matrix transpose
% filtWristAngleY = (filtfilt(B1,A1,scaledWristAngleY'))'; %Zero-phase filtering + Matrix transpose
% filtWristAngleZ = (filtfilt(B1,A1,scaledWristAngleZ'))'; %Zero-phase filtering + Matrix transpose

%% Data resizing

%Acceleration
resizedThumbAccX = scaledThumbAccX(1:featureLength,1);
resizedThumbAccY = scaledThumbAccY(1:featureLength,1);
resizedThumbAccZ = scaledThumbAccZ(1:featureLength,1);
resizedIndexAccX = scaledIndexAccX(1:featureLength,1);
resizedIndexAccY = scaledIndexAccY(1:featureLength,1);
resizedIndexAccZ = scaledIndexAccZ(1:featureLength,1);
resizedWristAccX = scaledWristAccX(1:featureLength,1);
resizedWristAccY = scaledWristAccY(1:featureLength,1);
resizedWristAccZ = scaledWristAccZ(1:featureLength,1);

accelerations = [resizedThumbAccX resizedThumbAccY resizedThumbAccZ resizedIndexAccX ...
    resizedIndexAccY resizedIndexAccZ resizedWristAccX resizedWristAccY ...
    resizedWristAccZ];

%Velocity
resizedThumbVelX = scaledThumbVelX(1:featureLength,1);
resizedThumbVelY = scaledThumbVelY(1:featureLength,1);
resizedThumbVelZ = scaledThumbVelZ(1:featureLength,1);
resizedIndexVelX = scaledIndexVelX(1:featureLength,1);
resizedIndexVelY = scaledIndexVelY(1:featureLength,1);
resizedIndexVelZ = scaledIndexVelZ(1:featureLength,1);
resizedWristVelX = scaledWristVelX(1:featureLength,1);
resizedWristVelY = scaledWristVelY(1:featureLength,1);
resizedWristVelZ = scaledWristVelZ(1:featureLength,1);

velocities = [resizedThumbVelX resizedThumbVelY resizedThumbVelZ resizedIndexVelX ...
    resizedIndexVelY resizedIndexVelZ resizedWristVelX resizedWristVelY ...
    resizedWristVelZ];

%Euler angle
resizedThumbAngleX = scaledThumbAngleX(1:featureLength,1);
resizedIndexAngleX = scaledIndexAngleX(1:featureLength,1);
resizedHandAngleX = scaledHandAngleX(1:featureLength,1);
resizedHandAngleY = scaledHandAngleY(1:featureLength,1);
resizedHandAngleZ = scaledHandAngleZ(1:featureLength,1);
resizedWristAngleX = scaledWristAngleX(1:featureLength,1);
resizedWristAngleY = scaledWristAngleY(1:featureLength,1);
resizedWristAngleZ = scaledWristAngleZ(1:featureLength,1);

angles = [resizedThumbAngleX resizedIndexAngleX ...
    resizedHandAngleX resizedHandAngleY resizedHandAngleZ ...
    resizedWristAngleX resizedWristAngleY resizedWristAngleZ];

%% Create feature columns (feature_length x 11)

featuresData = [accelerations velocities angles]';

features= [features; {featuresData}];

end

%features(1:length(features));

%% Create a label column (number of files, so each roz of the cell array x 1) 

labels = ones(m, 1);
labels = (AGE * labels) / 10;

end