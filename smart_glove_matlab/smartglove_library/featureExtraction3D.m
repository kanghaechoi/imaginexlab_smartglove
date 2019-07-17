function [features, labels] = featureExtraction3D(B1, A1, AGE, fileCount)
%% Feature extraction

features = zeros(10000, 11, fileCount);
labels = ones(10000, 1, fileCount);

for n = 1 : fileCount
    
    handData = dlmread(sprintf('Hand_IMU_%d_%d.txt', AGE, n)); %Read hand data
    wristData = dlmread(sprintf('Wrist_IMU_%d_%d.txt', AGE, n)); %Read wrist
    
    %feature20_n = handData;
    %wristData_n= wristData;


    %Feature matrix length definition
%     if length(handData) >= length(wristData), featureLength = length(wristData);
%     elseif length(handData) < length(wristData), featureLength = length(handData);
%     end

    %Acceleration
    thumbAcc = handData(:,7:9); %Thumb acceleration (X, Y, Z)
    indexAcc = handData(:,10:12); %Index acceleration (X, Y, Z)
    wristAcc = wristData(:,4:6); %Wrist acceleration (X, Y, Z)

    thumbAccMag = sum((thumbAcc .* thumbAcc),2); %Thumb acc magnitude
    indexAccMag = sum((indexAcc .* indexAcc),2); %Index acc magnitude
    wristAccMag = sum((wristAcc .* wristAcc),2); %Wrist acc magnitude

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
    scaledThumbAccMag = [featureScale(thumbAccMag); zeros(10000 - length(thumbAccMag),1)];
    scaledIndexAccMag = [featureScale(indexAccMag); zeros(10000 - length(indexAccMag),1)];
    scaledWristAccMag = [featureScale(wristAccMag); zeros(10000 - length(wristAccMag),1)];

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

    scaledThumbAngleX = [featureScale(deg2rad(thumbAngleX)); zeros(10000 - length(thumbAngleX),1)];
    scaledIndexAngleX = [featureScale(deg2rad(indexAngleX)); zeros(10000 - length(indexAngleX),1)];
    scaledHandAngleX = [featureScale(deg2rad(handAngleX)); zeros(10000 - length(handAngleX),1)];
    scaledHandAngleY = [featureScale(deg2rad(handAngleY)); zeros(10000 - length(handAngleY),1)];
    scaledHandAngleZ = [featureScale(deg2rad(handAngleZ)); zeros(10000 - length(handAngleZ),1)];
    scaledWristAngleX = [featureScale(deg2rad(wristAngleX)); zeros(10000 - length(wristAngleX),1)];
    scaledWristAngleY = [featureScale(deg2rad(wristAngleY)); zeros(10000 - length(wristAngleY),1)];
    scaledWristAngleZ = [featureScale(deg2rad(wristAngleZ)); zeros(10000 - length(wristAngleZ),1)];

    %% Data filtering

%     %Acceleration
%     filtThumbAccMag = (filtfilt(B1,A1,scaledThumbAccMag'))'; %Zero-phase filtering + Matrix transpose
%     filtIndexAccMag = (filtfilt(B1,A1,scaledIndexAccMag'))'; %Zero-phase filtering + Matrix transpose
%     filtWristAccMag = (filtfilt(B1,A1,scaledWristAccMag'))'; %Zero-phase filtering + Matrix transpose
% 
%     %Euler angle
%     filtThumbAngleX = (filtfilt(B1,A1,scaledThumbAngleX'))'; %Zero-phase filtering + Matrix transpose
%     filtIndexAngleX = (filtfilt(B1,A1,scaledIndexAngleX'))'; %Zero-phase filtering + Matrix transpose
%     filtHandAngleX = (filtfilt(B1,A1,scaledHandAngleX'))'; %Zero-phase filtering + Matrix transpose
%     filtHandAngleY = (filtfilt(B1,A1,scaledHandAngleY'))'; %Zero-phase filtering + Matrix transpose
%     filtHandAngleZ = (filtfilt(B1,A1,scaledHandAngleZ'))';%Zero-phase filtering + Matrix transpose
%     filtWristAngleX = (filtfilt(B1,A1,scaledWristAngleX'))'; %Zero-phase filtering + Matrix transpose
%     filtWristAngleY = (filtfilt(B1,A1,scaledWristAngleY'))'; %Zero-phase filtering + Matrix transpose
%     filtWristAngleZ = (filtfilt(B1,A1,scaledWristAngleZ'))'; %Zero-phase filtering + Matrix transpose

    %% Data resizing

%     %Acceleration
%     resizedThumbAccMag = filtThumbAccMag(1:featureLength,1);
%     resizedIndexAccMag = filtIndexAccMag(1:featureLength,1);
%     resizedWristAccMag = filtWristAccMag(1:featureLength,1);
% 
%     %Euler angle
%     resizedThumbAngleX = filtThumbAngleX(1:featureLength,1);
%     resizedIndexAngleX = filtIndexAngleX(1:featureLength,1);
%     resizedHandAngleX = filtHandAngleX(1:featureLength,1);
%     resizedHandAngleY = filtHandAngleY(1:featureLength,1);
%     resizedHandAngleZ = filtHandAngleZ(1:featureLength,1);
%     resizedWristAngleX = filtWristAngleX(1:featureLength,1);
%     resizedWristAngleY = filtWristAngleY(1:featureLength,1);
%     resizedWristAngleZ = filtWristAngleZ(1:featureLength,1);

    %% Create feature columns (feature_length x 11)
    
    features(:,:,n) = [scaledThumbAccMag, scaledIndexAccMag, scaledWristAccMag, scaledThumbAngleX, scaledIndexAngleX, ...
        scaledHandAngleX, scaledHandAngleY, scaledHandAngleZ, scaledWristAngleX, scaledWristAngleY, scaledWristAngleZ];

%% Create a label column (feature_length x 1) 

    labels(:,:,n) = (AGE .* ones(10000,1)) / 10;

end

end