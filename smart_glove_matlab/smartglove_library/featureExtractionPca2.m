function [features] = featureExtractionPca(AGE, fileCount)
%% Feature extraction

handData = dlmread(sprintf('Hand_IMU_%d_%d.txt', AGE, fileCount)); %Read hand data
wristData = dlmread(sprintf('Wrist_IMU_%d_%d.txt', AGE, fileCount)); %Read wrist

%Feature matrix length definition
if length(handData) >= length(wristData), featureLength = length(wristData);
elseif length(handData) < length(wristData), featureLength = length(handData);
end

%Acceleration
thumbAcc = handData(:,7:9); %Thumb acceleration (X, Y, Z)
indexAcc = handData(:,10:12); %Index acceleration (X, Y, Z)
wristAcc = wristData(:,4:6); %Wrist acceleration (X, Y, Z)

%Velocity
thumbVel = cumtrapz(thumbAcc); %Thumb velocity (X, Y, Z)
indexVel = cumtrapz(indexAcc); %Index velocity (X, Y, Z)
wristVel = cumtrapz(wristAcc); %Wrist velocity (X, Y, Z)

%Finger Euler angle
thumbAngleX = handData(:,4); %Thumb x-axis Euler angle
indexAngleX = handData(:,5); %Index x-axis Euler angle

%Hand Euler angle
handAngleX = handData(:,1); %Hand x-axis Euler angle
handAngleY = handData(:,2); %Hand y-axis Euler angle
handAngleZ = handData(:,3); %Hand z-axis Euler angle

%Wrist Euler angle
wristAngleX = wristData(:,1); %Wrist x-axis Euler angle
wristAngleY = wristData(:,2); %Wrist y-axis Euler angle
wristAngleZ = wristData(:,3); %Wrist z-axis Euler angle

%% Feature scaling

%Acceleration
scaledThumbAccX = thumbAcc(:,1);
scaledThumbAccY = thumbAcc(:,2);
scaledThumbAccZ = thumbAcc(:,3);
scaledIndexAccX = indexAcc(:,1);
scaledIndexAccY = indexAcc(:,2);
scaledIndexAccZ = indexAcc(:,3);
scaledWristAccX = wristAcc(:,1);
scaledWristAccY = wristAcc(:,2);
scaledWristAccZ = wristAcc(:,3);

%Velocity
scaledThumbVelX = thumbVel(:,1);
scaledThumbVelY = thumbVel(:,2);
scaledThumbVelZ = thumbVel(:,3);
scaledIndexVelX = indexVel(:,1);
scaledIndexVelY = indexVel(:,2);
scaledIndexVelZ = indexVel(:,3);
scaledWristVelX = wristVel(:,1);
scaledWristVelY = wristVel(:,2);
scaledWristVelZ = wristVel(:,3);

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

scaledThumbAngleX = deg2rad(thumbAngleX);
scaledIndexAngleX = deg2rad(indexAngleX);
scaledHandAngleX = deg2rad(handAngleX);
scaledHandAngleY = deg2rad(handAngleY);
scaledHandAngleZ = deg2rad(handAngleZ);
scaledWristAngleX = deg2rad(wristAngleX);
scaledWristAngleY = deg2rad(wristAngleY);
scaledWristAngleZ = deg2rad(wristAngleZ);

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

features = [accelerations velocities angles];

end