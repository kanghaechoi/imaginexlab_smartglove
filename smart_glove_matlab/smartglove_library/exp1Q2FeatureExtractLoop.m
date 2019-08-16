function [features, labels] = exp1Q2FeatureExtractLoop(B1, A1, AGE, fileCount, GENDER, subjectNum, numberOfFeatures)
%% Feature extraction

%Read .txt file
handData = dlmread(sprintf('Hand_IMU_%d_%d_%d_0%d.txt', AGE, GENDER, subjectNum, fileCount)); %Read hand data
wristData = dlmread(sprintf('Wrist_IMU_%d_%d_%d_0%d.txt', AGE, GENDER, subjectNum, fileCount)); %Read wrist
    
%Feature matrix length definition
if length(handData) >= length(wristData), featureLength = length(wristData);
elseif length(handData) < length(wristData), featureLength = length(handData);
end
    
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

%Euler angle
resizedThumbAngleX = scaledThumbAngleX(1:featureLength,1);
resizedIndexAngleX = scaledIndexAngleX(1:featureLength,1);
resizedHandAngleX = scaledHandAngleX(1:featureLength,1);
resizedHandAngleY = scaledHandAngleY(1:featureLength,1);
resizedHandAngleZ = scaledHandAngleZ(1:featureLength,1);
resizedWristAngleX = scaledWristAngleX(1:featureLength,1);
resizedWristAngleY = scaledWristAngleY(1:featureLength,1);
resizedWristAngleZ = scaledWristAngleZ(1:featureLength,1);
%1-7
thumbFin = [resizedThumbAccX, resizedThumbAccY, resizedThumbAccZ, ...
    resizedThumbVelX, resizedThumbVelY, resizedThumbVelZ, resizedThumbAngleX];
%8-14
indexFin = [resizedIndexAccX, resizedIndexAccY, resizedIndexAccZ, ...
    resizedIndexVelX, resizedIndexVelY, resizedIndexVelZ, resizedIndexAngleX];
%15-17
hand = [resizedHandAngleX, resizedHandAngleY, resizedHandAngleZ];
%18-26
wrist = [resizedWristAccX, resizedWristAccY, resizedWristAccZ, ...
    resizedWristVelX, resizedWristVelY, resizedWristVelZ, ...
    resizedWristAngleX, resizedWristAngleY, resizedWristAngleZ];

%% Create feature columns (feature_length x 11)
identifyFeatures = [resizedWristAngleZ resizedHandAngleZ resizedThumbAccX resizedWristAccY resizedThumbVelX ...
            resizedThumbAccZ resizedIndexAccX resizedIndexAngleX resizedIndexVelZ resizedWristVelY ...
            resizedWristAngleX resizedThumbVelZ resizedThumbAngleX resizedWristAccX resizedWristAccZ ...
            resizedThumbAccY resizedWristVelZ resizedIndexVelY resizedWristAngleY resizedHandAngleY ...
            resizedIndexAccZ resizedIndexAccY resizedIndexVelX resizedThumbVelY resizedHandAngleX ... 
            resizedWristVelX];
        
features = identifyFeatures(:, 1:numberOfFeatures)';

%% Create a label column (number of files, so each roz of the cell array x 1) 

%Labels for identification
labels = ones(1, 1);
labels = ((GENDER * 1000) + (AGE * 10) + subjectNum) * labels;

end