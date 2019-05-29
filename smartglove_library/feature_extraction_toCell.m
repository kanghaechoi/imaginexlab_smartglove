function [features, labels] = feature_extraction(b1, a1, age, file_count)
%% Feature extraction

for n = 1:file_count

    if n == 1
        %hand_data = ones(1, 12);
        %wrist_data = ones(1, 6);
         
         hand_read = dlmread(sprintf('Hand_IMU_%d_%d.txt', age, n)); %Read hand data
         wrist_read = dlmread(sprintf('Wrist_IMU_%d_%d.txt', age, n)); %Read wrist
         hand_data = {hand_read};
         wrist_data = {wrist_read};
        continue;
    else  
        hand_read = dlmread(sprintf('Hand_IMU_%d_%d.txt', age, n)); %Read hand data
        wrist_read = dlmread(sprintf('Wrist_IMU_%d_%d.txt', age, n)); %Read wrist
        hand_data= [hand_data; {hand_read}];
        wrist_data= [wrist_data; {wrist_read}];
    end
end 
hand_data(1:length(hand_data));
wrist_data(1:length(wrist_data));



[m, n] = size(hand_data);
features = {};




for k = 1:m
    
%Feature matrix length definition
if length(hand_data{k}) >= length(wrist_data{k}), feature_length = length(wrist_data{k});
elseif length(hand_data{k}) < length(wrist_data{k}), feature_length = length(hand_data{k});
end
    
thumb_acc = hand_data{k}(:,7:9); %Thumb acceleration (X, Y, Z)
index_acc = hand_data{k}(:,10:12); %Index acceleration (X, Y, Z)
wrist_acc = wrist_data{k}(:,4:6); %Wrist acceleration (X, Y, Z)

thumb_acc_mag = sum((thumb_acc .* thumb_acc),2) /1e+6; %Thumb acc magnitude + Downsizing
index_acc_mag = sum((index_acc .* index_acc),2) /1e+6; %Index acc magnitude + Downsizing
wrist_acc_mag = sum((wrist_acc .* wrist_acc),2) /1e+6; %Wrist acc magnitude + Downsizing

%Finger Euler angle
thumb_x_angle = hand_data{k}(:,4); %Thumb x-axis Euler angle
index_x_angle = hand_data{k}(:,5); %Index x-axis Euler angle

%Hand Euler angle
hand_x_angle = hand_data{k}(:,1); %Hand x-axis Euler angle
hand_y_angle = hand_data{k}(:,2); %Hand y-axis Euler angle
hand_z_angle = hand_data{k}(:,3); %Hand z-axis Euler angle

%Wrist Euler angle
wrist_x_angle = wrist_data{k}(:,1); %Wrist x-axis Euler angle
wrist_y_angle = wrist_data{k}(:,2); %Wrist y-axis Euler angle
wrist_z_angle = wrist_data{k}(:,3); %Wrist z-axis Euler angle

%% Data filtering

%Acceleration
zpl_thumb_acc_mag = (filtfilt(b1,a1,thumb_acc_mag'))'; %Zero-phase filtering + Matrix transpose
zpl_index_acc_mag = (filtfilt(b1,a1,index_acc_mag'))'; %Zero-phase filtering + Matrix transpose
zpl_wrist_acc_mag = (filtfilt(b1,a1,wrist_acc_mag'))'; %Zero-phase filtering + Matrix transpose

%Euler angle
zpl_thumb_x_angle = (filtfilt(b1,a1,thumb_x_angle'))'; %Zero-phase filtering + Matrix transpose
zpl_index_x_angle = (filtfilt(b1,a1,index_x_angle'))'; %Zero-phase filtering + Matrix transpose
zpl_hand_x_angle = (filtfilt(b1,a1,hand_x_angle'))'; %Zero-phase filtering + Matrix transpose
zpl_hand_y_angle = (filtfilt(b1,a1,hand_y_angle'))'; %Zero-phase filtering + Matrix transpose
zpl_hand_z_angle = (filtfilt(b1,a1,hand_z_angle'))';%Zero-phase filtering + Matrix transpose
zpl_wrist_x_angle = (filtfilt(b1,a1,wrist_x_angle'))'; %Zero-phase filtering + Matrix transpose
zpl_wrist_y_angle = (filtfilt(b1,a1,wrist_y_angle'))'; %Zero-phase filtering + Matrix transpose
zpl_wrist_z_angle = (filtfilt(b1,a1,wrist_z_angle'))'; %Zero-phase filtering + Matrix transpose

%% Data resizing

%Acceleration
resized_thumb_acc_mag = zpl_thumb_acc_mag(1:feature_length,1);
resized_index_acc_mag = zpl_index_acc_mag(1:feature_length,1);
resized_wrist_acc_mag = zpl_wrist_acc_mag(1:feature_length,1);

%Euler angle
resized_thumb_x_angle = zpl_thumb_x_angle(1:feature_length,1);
resized_index_x_angle = zpl_index_x_angle(1:feature_length,1);
resized_hand_x_angle = zpl_hand_x_angle(1:feature_length,1);
resized_hand_y_angle = zpl_hand_y_angle(1:feature_length,1);
resized_hand_z_angle = zpl_hand_z_angle(1:feature_length,1);
resized_wrist_x_angle = zpl_wrist_x_angle(1:feature_length,1);
resized_wrist_y_angle = zpl_wrist_y_angle(1:feature_length,1);
resized_wrist_z_angle = zpl_wrist_z_angle(1:feature_length,1);

%% Feature scaling
%Acceleration
scaled_thumb_acc_mag = normalize(resized_thumb_acc_mag, 'norm');
scaled_index_acc_mag = normalize(resized_index_acc_mag, 'norm');
scaled_wrist_acc_mag = normalize(resized_wrist_acc_mag, 'norm');

%Euler angle
scaled_thumb_x_angle = normalize(resized_thumb_x_angle, 'norm');
scaled_index_x_angle = normalize(resized_index_x_angle, 'norm');
scaled_hand_x_angle = normalize(resized_hand_x_angle, 'norm');
scaled_hand_y_angle = normalize(resized_hand_y_angle, 'norm');
scaled_hand_z_angle = normalize(resized_hand_z_angle, 'norm');
scaled_wrist_x_angle = normalize(resized_wrist_x_angle, 'norm');
scaled_wrist_y_angle = normalize(resized_wrist_y_angle, 'norm');
scaled_wrist_z_angle = normalize(resized_wrist_z_angle, 'norm');

%% Create feature columns (feature_length x 11)


features_data = [scaled_thumb_acc_mag, scaled_index_acc_mag, scaled_wrist_acc_mag, scaled_thumb_x_angle, scaled_index_x_angle, ...
    scaled_hand_x_angle, scaled_hand_y_angle, scaled_hand_z_angle, scaled_wrist_x_angle, scaled_wrist_y_angle, scaled_wrist_z_angle];

features= [features; {features_data}];
end

features(1:length(features));





%% Create a label column (number of files, so each roz of the cell array x 1) 

labels = ones(m, 1);
labels = (age * labels) / 10;





end