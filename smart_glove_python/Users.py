from sklearn import preprocessing
import numpy as np
class User:
    def __init__(self,hand_data, wrist_data):
        self.hand_data = hand_data
        self.wrist_data = wrist_data
        self.features = feature_extraction(hand_data,wrist_data)

# thumb_acc = hand_data{k}(:,7:9); %Thumb acceleration (X, Y, Z)
# index_acc = hand_data{k}(:,10:12); %Index acceleration (X, Y, Z)
# wrist_acc = wrist_data{k}(:,4:6); %Wrist acceleration (X, Y, Z)


def feature_extraction(hand_data, wrist_data):
    featuresLenght = len(hand_data)
    if len(hand_data)>len(wrist_data):
        featuresLenght = len(wrist_data)
    print(hand_data[0][6:9])
    for i in range(featuresLenght):
        # thumb_acc = hand_data[i][6:9]
        # index_acc = hand_data[i][9:12]
        # wrist_acc = wrist_data[i][3:6]
        thumb_acc = hand_data[:, [6, 7, 8]]
        index_acc = hand_data[:, [9, 10, 11]]
        wrist_acc = wrist_data[:, [3, 4, 5]]

        thumb_acc_mag = sum(np.multiply(thumb_acc, thumb_acc), 2) / 1e+6;
        index_acc_mag = sum((index_acc * index_acc), 2) / 1e+6;
        wrist_acc_mag = sum((wrist_acc * wrist_acc), 2) / 1e+6;

        # Finger Euler angle
        thumb_x_angle = hand_data[i][3]
        index_x_angle = hand_data[i][4]

        # Hand Euler angle
        hand_x_angle = hand_data[i][0]
        hand_y_angle = hand_data[i][1]
        hand_z_angle = hand_data[i][2]

        # Wrist Euler angle
        wrist_x_angle = wrist_data[i][0]
        wrist_y_angle = wrist_data[i][1]
        wrist_z_angle = wrist_data[i][2]

        #======= Features Scaling ========
        # Acceleration
        scaled_thumb_acc_mag = preprocessing.normalize(thumb_acc_mag);
        scaled_index_acc_mag = preprocessing.normalize(index_acc_mag);
        scaled_wrist_acc_mag = preprocessing.normalize(wrist_acc_mag);

        # Euler Angle
        scaled_thumb_x_angle = preprocessing.normalize(thumb_x_angle);
        scaled_index_x_angle = preprocessing.normalize(index_x_angle);
        scaled_hand_x_angle = preprocessing.normalize(hand_x_angle);
        scaled_hand_y_angle = preprocessing.normalize(hand_y_angle);
        scaled_hand_z_angle = preprocessing.normalize(hand_z_angle);
        scaled_wrist_x_angle = preprocessing.normalize(wrist_x_angle);
        scaled_wrist_y_angle = preprocessing.normalize(wrist_y_angle);
        scaled_wrist_z_angle = preprocessing.normalize(wrist_z_angle);
        print(1)