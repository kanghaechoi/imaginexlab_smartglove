function [features, labels] = featureExtraction(B1, A1, AGE, fileCount)
%% Feature extraction

for n = 1:fileCount
    if n == 1
        handData = ones(1, 12);
        wristData = ones(1, 6);
        %continue;
    end
        
    handRead = dlmread(sprintf('Hand_IMU_%d_%d.txt', AGE, n)); %Read hand data
    wristRead = dlmread(sprintf('Wrist_IMU_%d_%d.txt', AGE, n)); %Read wrist
    
    handData = [handData; handRead];
    wristData = [wristData; wristRead];
end 

%Feature matrix length definition
if length(handData) >= length(wristData), featureLength = length(wristData);
elseif length(handData) < length(wristData), featureLength = length(handData);
end

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

%% Data filtering

%Acceleration
zplThumbAccMag = (filtfilt(B1,A1,thumbAccMag'))'; %Zero-phase filtering + Matrix transpose
zplIndexAccMag = (filtfilt(B1,A1,indexAccMag'))'; %Zero-phase filtering + Matrix transpose
zplWristAccMag = (filtfilt(B1,A1,wristAccMag'))'; %Zero-phase filtering + Matrix transpose

%Euler angle
zplThumbAngleX = (filtfilt(B1,A1,thumbAngleX'))'; %Zero-phase filtering + Matrix transpose
zplIndexAngleX = (filtfilt(B1,A1,indexAngleX'))'; %Zero-phase filtering + Matrix transpose
zplHandAngleX = (filtfilt(B1,A1,handAngleX'))'; %Zero-phase filtering + Matrix transpose
zplHandAngleY = (filtfilt(B1,A1,handAngleY'))'; %Zero-phase filtering + Matrix transpose
zplHandAngleZ = (filtfilt(B1,A1,handAngleZ'))';%Zero-phase filtering + Matrix transpose
zplWristAngleX = (filtfilt(B1,A1,wristAngleX'))'; %Zero-phase filtering + Matrix transpose
zplWristAngleY = (filtfilt(B1,A1,wristAngleY'))'; %Zero-phase filtering + Matrix transpose
zplWristAngleZ = (filtfilt(B1,A1,wristAngleZ'))'; %Zero-phase filtering + Matrix transpose

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

%% Feature scaling
%Acceleration
scaledThumbAccMag = normalize(resizedThumbAccMag, 'norm');
scaledIndexAccMag = normalize(resizedIndexAccMag, 'norm');
scaledWristAccMag = normalize(resizedWristAccMag, 'norm');

%Euler angle
scaledThumbAngleX = normalize(resizedThumbAngleX, 'norm');
scaledIndexAngleX = normalize(resizedIndexAngleX, 'norm');
scaledHandAngleX = normalize(resizedHandAngleX, 'norm');
scaledHandAngleY = normalize(resizedHandAngleY, 'norm');
scaledHandAngleZ = normalize(resizedHandAngleZ, 'norm');
scaledWristAngleX = normalize(resizedWristAngleX, 'norm');
scaledWristAngleY = normalize(resizedWristAngleY, 'norm');
scaledWristAngleZ = normalize(resizedWristAngleZ, 'norm');

%% Create a label column (feature_length x 1) 

labels = ones(featureLength, 1);
labels = (AGE * labels) / 10;

%% Create feature columns (feature_length x 11)

features = [scaledThumbAccMag, scaledIndexAccMag, scaledWristAccMag, scaledThumbAngleX, scaledIndexAngleX, ...
    scaledHandAngleX, scaledHandAngleY, scaledHandAngleZ, scaledWristAngleX, scaledWristAngleY, scaledWristAngleZ];

end