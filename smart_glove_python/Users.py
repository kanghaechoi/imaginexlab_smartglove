from scipy.signal import filtfilt
from sklearn import preprocessing
import numpy as np
from scipy import signal


class User:
    def __init__(self, hand_data, wrist_data, age):
        self.hand_data = hand_data
        self.wrist_data = wrist_data
        self.features = feature_extraction(hand_data, wrist_data)
        self.labels = age/10


# thumb_acc = hand_data{k}(:,7:9); %Thumb acceleration (X, Y, Z)
# index_acc = hand_data{k}(:,10:12); %Index acceleration (X, Y, Z)
# wrist_acc = wrist_data{k}(:,4:6); %Wrist acceleration (X, Y, Z)


def feature_extraction(hand_data, wrist_data):
    feature_length = len(hand_data)
    if len(hand_data) > len(wrist_data):
        feature_length = len(wrist_data)
    # for i in range(featuresLenght):
    # thumb_acc = hand_data[i][6:9]
    # index_acc = hand_data[i][9:12]
    # wrist_acc = wrist_data[i][3:6]
    thumb_acc = hand_data[:, [6, 7, 8]]
    index_acc = hand_data[:, [9, 10, 11]]
    wrist_acc = wrist_data[:, [3, 4, 5]]

    thumb_acc_mag = np.sum(np.multiply(thumb_acc, thumb_acc), 1) / 1e+6;
    index_acc_mag = np.sum(np.multiply(index_acc, index_acc), 1) / 1e+6;
    wrist_acc_mag = np.sum(np.multiply(wrist_acc, wrist_acc), 1) / 1e+6;

    # Finger Euler angle
    thumb_x_angle = hand_data[:, 3]
    index_x_angle = hand_data[:, 4]

    # Hand Euler angle
    hand_x_angle = hand_data[:, 0]
    hand_y_angle = hand_data[:, 1]
    hand_z_angle = hand_data[:, 2]

    # Wrist Euler angle
    wrist_x_angle = wrist_data[:, 0]
    wrist_y_angle = wrist_data[:, 1]
    wrist_z_angle = wrist_data[:, 2]

    # Data filtering
    [b1, a1] = butterworth_5hz(100)
    # Acceleration

    zpl_thumb_acc_mag = (filtfilt(b1, a1, thumb_acc_mag))
    zpl_index_acc_mag = (filtfilt(b1, a1, index_acc_mag))
    zpl_wrist_acc_mag = (filtfilt(b1, a1, wrist_acc_mag))

    # Euler   angle
    zpl_thumb_x_angle = (filtfilt(b1, a1, thumb_x_angle))
    zpl_index_x_angle = (filtfilt(b1, a1, index_x_angle))
    zpl_hand_x_angle = (filtfilt(b1, a1, hand_x_angle))
    zpl_hand_y_angle = (filtfilt(b1, a1, hand_y_angle))
    zpl_hand_z_angle = (filtfilt(b1, a1, hand_z_angle))
    zpl_wrist_x_angle = (filtfilt(b1, a1, wrist_x_angle))
    zpl_wrist_y_angle = (filtfilt(b1, a1, wrist_y_angle))
    zpl_wrist_z_angle = (filtfilt(b1, a1, wrist_z_angle))

    # ======= Features Scaling ========
    # Acceleration
    resized_thumb_acc_mag = zpl_thumb_acc_mag.reshape(1, -1)
    resized_index_acc_mag = zpl_index_acc_mag.reshape(1, -1)
    resized_wrist_acc_mag = zpl_wrist_acc_mag.reshape(1, -1)
    resized_thumb_x_angle = zpl_thumb_x_angle.reshape(1, -1)
    resized_index_x_angle = zpl_index_x_angle.reshape(1, -1)
    resized_hand_x_angle = zpl_hand_x_angle.reshape(1, -1)
    resized_hand_y_angle = zpl_hand_y_angle.reshape(1, -1)
    resized_hand_z_angle = zpl_hand_z_angle.reshape(1, -1)
    resized_wrist_x_angle = zpl_wrist_x_angle.reshape(1, -1)
    resized_wrist_y_angle = zpl_wrist_y_angle.reshape(1, -1)
    resized_wrist_z_angle = zpl_wrist_z_angle.reshape(1, -1)

    scaled_thumb_acc_mag = preprocessing.normalize(resized_thumb_acc_mag, norm='l2');
    scaled_index_acc_mag = preprocessing.normalize(resized_index_acc_mag, norm='l2');
    scaled_wrist_acc_mag = preprocessing.normalize(resized_wrist_acc_mag, norm='l2');

    # Euler Angle
    scaled_thumb_x_angle = preprocessing.normalize(resized_thumb_x_angle, norm='l2');
    scaled_index_x_angle = preprocessing.normalize(resized_index_x_angle, norm='l2');
    scaled_hand_x_angle = preprocessing.normalize(resized_hand_x_angle, norm='l2');
    scaled_hand_y_angle = preprocessing.normalize(resized_hand_y_angle, norm='l2');
    scaled_hand_z_angle = preprocessing.normalize(resized_hand_z_angle, norm='l2');
    scaled_wrist_x_angle = preprocessing.normalize(resized_wrist_x_angle, norm='l2');
    scaled_wrist_y_angle = preprocessing.normalize(resized_wrist_y_angle, norm='l2');
    scaled_wrist_z_angle = preprocessing.normalize(resized_wrist_z_angle, norm='l2');


    return [scaled_thumb_acc_mag, scaled_index_acc_mag, scaled_wrist_acc_mag, scaled_thumb_x_angle, scaled_index_x_angle,
            scaled_hand_x_angle
            ,scaled_hand_y_angle,scaled_hand_z_angle,scaled_wrist_x_angle,scaled_wrist_y_angle,scaled_wrist_z_angle]
    # Data resizing
    # resized_thumb_acc_mag = zpl_thumb_acc_mag(1:feature_length, 1)


def butterworth_5hz(fs):

    fmax = fs/2
    LPF_cutoff = 5/fmax
    order = 5;
    [b1, a1] = signal.butter(order, LPF_cutoff, 'low')
    return b1, a1