from scipy.signal import filtfilt
from sklearn import preprocessing
import numpy as np
from scipy import signal
import pandas as pd

class User:
    def __init__(self, hand_data, wrist_data, age):
        self.hand_data = hand_data
        self.wrist_data = wrist_data
        self.features, self.length = feature_extraction(hand_data, wrist_data)
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
    thumb_acc = hand_data[:, (6, 7, 8)]
    index_acc = hand_data[:, (9, 10, 11)]
    wrist_acc = wrist_data[:, (3, 4, 5)]

    thumb_acc_mag = np.sum(np.multiply(thumb_acc, thumb_acc), 1);
    index_acc_mag = np.sum(np.multiply(index_acc, index_acc), 1);
    wrist_acc_mag = np.sum(np.multiply(wrist_acc, wrist_acc), 1);
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

    thumb_acc_mag = np.reshape(thumb_acc_mag, (1, -1))
    index_acc_mag = np.reshape(index_acc_mag, (1, -1))
    wrist_acc_mag = np.reshape(wrist_acc_mag, (1, -1))

    scaled_thumb_acc_mag = preprocessing.normalize(thumb_acc_mag, norm='l2');
    scaled_index_acc_mag = preprocessing.normalize(index_acc_mag, norm='l2');
    scaled_wrist_acc_mag = preprocessing.normalize(wrist_acc_mag, norm='l2');




    scaledThumbAngleX = np.deg2rad(thumb_x_angle);
    scaledIndexAngleX = np.deg2rad(index_x_angle);
    scaledHandAngleX = np.deg2rad(hand_x_angle);
    scaledHandAngleY = np.deg2rad(hand_y_angle);
    scaledHandAngleZ = np.deg2rad(hand_z_angle);
    scaledWristAngleX = np.deg2rad(wrist_x_angle);
    scaledWristAngleY = np.deg2rad(wrist_y_angle);
    scaledWristAngleZ = np.deg2rad(wrist_z_angle);

    # Data filtering
    [b1, a1] = butterworth_5hz(100)
    # Acceleration

    zpl_thumb_acc_mag = (filtfilt(b1, a1, scaled_thumb_acc_mag))
    zpl_index_acc_mag = (filtfilt(b1, a1, scaled_index_acc_mag))
    zpl_wrist_acc_mag = (filtfilt(b1, a1, scaled_wrist_acc_mag))

    # Euler   angle
    zpl_thumb_x_angle = (filtfilt(b1, a1, scaledThumbAngleX))
    zpl_index_x_angle = (filtfilt(b1, a1, scaledIndexAngleX))
    zpl_hand_x_angle = (filtfilt(b1, a1, scaledHandAngleX))
    zpl_hand_y_angle = (filtfilt(b1, a1, scaledHandAngleY))
    zpl_hand_z_angle = (filtfilt(b1, a1, scaledHandAngleZ))
    zpl_wrist_x_angle = (filtfilt(b1, a1, scaledWristAngleX))
    zpl_wrist_y_angle = (filtfilt(b1, a1, scaledWristAngleY))
    zpl_wrist_z_angle = (filtfilt(b1, a1, scaledWristAngleZ))

    zpl_thumb_acc_mag = np.rot90(zpl_thumb_acc_mag)
    zpl_index_acc_mag = np.rot90(zpl_index_acc_mag)
    zpl_wrist_acc_mag = np.rot90(zpl_wrist_acc_mag)

    zpl_thumb_x_angle = np.reshape(zpl_thumb_x_angle, (-1, 1))
    zpl_index_x_angle = np.reshape(zpl_index_x_angle, (-1, 1))
    zpl_hand_x_angle = np.reshape(zpl_hand_x_angle, (-1, 1))
    zpl_hand_y_angle = np.reshape(zpl_hand_y_angle, (-1, 1))
    zpl_hand_z_angle = np.reshape(zpl_hand_z_angle, (-1, 1))
    zpl_wrist_x_angle = np.reshape(zpl_wrist_x_angle, (-1, 1))
    zpl_wrist_y_angle = np.reshape(zpl_wrist_y_angle, (-1, 1))
    zpl_wrist_z_angle = np.reshape(zpl_wrist_z_angle, (-1, 1))

    # ======= Features Scaling ========
    # Acceleration

    resized_thumb_acc_mag = zpl_thumb_acc_mag[0:feature_length,:]
    resized_index_acc_mag = zpl_index_acc_mag[0:feature_length,:]
    resized_wrist_acc_mag = zpl_wrist_acc_mag[0:feature_length,:]
    resized_thumb_x_angle = zpl_thumb_x_angle[0:feature_length,:]
    resized_index_x_angle = zpl_index_x_angle[0:feature_length,:]
    resized_hand_x_angle = zpl_hand_x_angle[0:feature_length,:]
    resized_hand_y_angle = zpl_hand_y_angle[0:feature_length,:]
    resized_hand_z_angle = zpl_hand_z_angle[0:feature_length,:]
    resized_wrist_x_angle = zpl_wrist_x_angle[0:feature_length,:]
    resized_wrist_y_angle = zpl_wrist_y_angle[0:feature_length,:]
    resized_wrist_z_angle = zpl_wrist_z_angle[0:feature_length,:]

    resized_thumb_acc_mag = np.reshape(resized_thumb_acc_mag, (1, -1))
    resized_index_acc_mag = np.reshape(resized_index_acc_mag, (1, -1))
    resized_wrist_acc_mag = np.reshape(resized_wrist_acc_mag, (1, -1))
    resized_thumb_x_angle = np.reshape(resized_thumb_x_angle, (1, -1))
    resized_index_x_angle = np.reshape(resized_index_x_angle, (1, -1))
    resized_hand_x_angle = np.reshape(resized_hand_x_angle, (1, -1))
    resized_hand_y_angle = np.reshape(resized_hand_y_angle, (1, -1))
    resized_hand_z_angle = np.reshape(resized_hand_z_angle, (1, -1))
    resized_wrist_x_angle = np.reshape(resized_wrist_x_angle, (1, -1))
    resized_wrist_y_angle = np.reshape(resized_wrist_y_angle, (1, -1))
    resized_wrist_z_angle = np.reshape(resized_wrist_z_angle, (1, -1))

    '''
    print(resized_thumb_acc_mag.shape)
    print(resized_index_acc_mag.shape)
    print(resized_wrist_acc_mag.shape)
    print(resized_thumb_x_angle.shape)
    print(resized_index_x_angle.shape)
    print(resized_hand_x_angle.shape)
    print(resized_hand_y_angle.shape)
    print(resized_hand_z_angle.shape)
    print(resized_wrist_x_angle.shape)
    print(resized_wrist_y_angle.shape)
    print(resized_wrist_z_angle.shape)
    '''

    return [resized_thumb_acc_mag, resized_index_acc_mag, resized_wrist_acc_mag, resized_thumb_x_angle, resized_index_x_angle,
            resized_hand_x_angle
            ,resized_hand_y_angle,resized_hand_z_angle,resized_wrist_x_angle,resized_wrist_y_angle,resized_wrist_z_angle],feature_length
    # Data resizing
    # resized_thumb_acc_mag = zpl_thumb_acc_mag(1:feature_length, 1)


class UserDataframe:
    def __init__(self, hand_data, wrist_data, age):
        self.hand_data = hand_data
        self.wrist_data = wrist_data
        print(hand_data.head(3))
        self.features, self.length = feature_extraction(hand_data, wrist_data)
        self.labels = age / 10


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
    thumb_acc = hand_data[['thumb_acc_x','thumb_acc_y','thumb_acc_z']]
    index_acc = hand_data[['index_acc_x','index_acc_y','index_acc_z']]
    wrist_acc = wrist_data[['wrist_acc_x','wrist_acc_y','wrist_acc_z']]
    # thumb_acc = hand_data[:, (6, 7, 8)]
    # index_acc = hand_data[:, (9, 10, 11)]
    # wrist_acc = wrist_data[:, (3, 4, 5)]

    thumb_acc_mag = np.sum(np.multiply(thumb_acc, thumb_acc), 1) / 1e+6;
    index_acc_mag = np.sum(np.multiply(index_acc, index_acc), 1) / 1e+6;
    wrist_acc_mag = np.sum(np.multiply(wrist_acc, wrist_acc), 1) / 1e+6;
    # Finger Euler angle

    thumb_x_angle = hand_data[['thumb_x_angle']]
    index_x_angle = hand_data[['index_x_angle']]

    # Hand Euler angle
    hand_x_angle = hand_data[['hand_x_angle']]
    hand_y_angle = hand_data[['hand_y_angle']]
    hand_z_angle = hand_data[['hand_z_angle']]

    # Wrist Euler angle
    wrist_x_angle = wrist_data[['wrist_x_angle']]
    wrist_y_angle = wrist_data[['wrist_y_angle']]
    wrist_z_angle = wrist_data[['wrist_z_angle']]

    scaled_thumb_acc_mag=  ((thumb_acc_mag-thumb_acc_mag.min())/(thumb_acc_mag.max()-thumb_acc_mag.min()))
    scaled_index_acc_mag = ((index_acc_mag - index_acc_mag.min()) / (index_acc_mag.max() - index_acc_mag.min()))
    scaled_wrist_acc_mag = ((wrist_acc_mag - wrist_acc_mag.min()) / (wrist_acc_mag.max() - wrist_acc_mag.min()))

    # scaled_thumb_acc_mag = preprocessing.normalize(thumb_acc_mag, norm='l1');
    # scaled_index_acc_mag = preprocessing.normalize(index_acc_mag, norm='l2');
    # scaled_wrist_acc_mag = preprocessing.normalize(wrist_acc_mag, norm='l2');

    scaledThumbAngleX = np.deg2rad(thumb_x_angle);
    scaledIndexAngleX = np.deg2rad(index_x_angle);
    scaledHandAngleX = np.deg2rad(hand_x_angle);
    scaledHandAngleY = np.deg2rad(hand_y_angle);
    scaledHandAngleZ = np.deg2rad(hand_z_angle);
    scaledWristAngleX = np.deg2rad(wrist_x_angle);
    scaledWristAngleY = np.deg2rad(wrist_y_angle);
    scaledWristAngleZ = np.deg2rad(wrist_z_angle);

    # Data filtering
    [b1, a1] = butterworth_5hz(100)
    # Acceleration

    zpl_thumb_acc_mag = (filtfilt(b1, a1, scaled_thumb_acc_mag))
    zpl_index_acc_mag = (filtfilt(b1, a1, scaled_index_acc_mag))
    zpl_wrist_acc_mag = (filtfilt(b1, a1, scaled_wrist_acc_mag))

    # Euler   angle
    zpl_thumb_x_angle = (filtfilt(b1, a1, scaledThumbAngleX))
    zpl_index_x_angle = (filtfilt(b1, a1, scaledIndexAngleX))
    zpl_hand_x_angle = (filtfilt(b1, a1, scaledHandAngleX))
    zpl_hand_y_angle = (filtfilt(b1, a1, scaledHandAngleY))
    zpl_hand_z_angle = (filtfilt(b1, a1, scaledHandAngleZ))
    zpl_wrist_x_angle = (filtfilt(b1, a1, scaledWristAngleX))
    zpl_wrist_y_angle = (filtfilt(b1, a1, scaledWristAngleY))
    zpl_wrist_z_angle = (filtfilt(b1, a1, scaledWristAngleZ))

    zpl_thumb_acc_mag = np.rot90(zpl_thumb_acc_mag)
    zpl_index_acc_mag = np.rot90(zpl_index_acc_mag)
    zpl_wrist_acc_mag = np.rot90(zpl_wrist_acc_mag)

    zpl_thumb_x_angle = np.reshape(zpl_thumb_x_angle, (-1, 1))
    zpl_index_x_angle = np.reshape(zpl_index_x_angle, (-1, 1))
    zpl_hand_x_angle = np.reshape(zpl_hand_x_angle, (-1, 1))
    zpl_hand_y_angle = np.reshape(zpl_hand_y_angle, (-1, 1))
    zpl_hand_z_angle = np.reshape(zpl_hand_z_angle, (-1, 1))
    zpl_wrist_x_angle = np.reshape(zpl_wrist_x_angle, (-1, 1))
    zpl_wrist_y_angle = np.reshape(zpl_wrist_y_angle, (-1, 1))
    zpl_wrist_z_angle = np.reshape(zpl_wrist_z_angle, (-1, 1))

    # ======= Features Scaling ========
    # Acceleration

    resized_thumb_acc_mag = zpl_thumb_acc_mag[0:feature_length, :]
    resized_index_acc_mag = zpl_index_acc_mag[0:feature_length, :]
    resized_wrist_acc_mag = zpl_wrist_acc_mag[0:feature_length, :]
    resized_thumb_x_angle = zpl_thumb_x_angle[0:feature_length, :]
    resized_index_x_angle = zpl_index_x_angle[0:feature_length, :]
    resized_hand_x_angle = zpl_hand_x_angle[0:feature_length, :]
    resized_hand_y_angle = zpl_hand_y_angle[0:feature_length, :]
    resized_hand_z_angle = zpl_hand_z_angle[0:feature_length, :]
    resized_wrist_x_angle = zpl_wrist_x_angle[0:feature_length, :]
    resized_wrist_y_angle = zpl_wrist_y_angle[0:feature_length, :]
    resized_wrist_z_angle = zpl_wrist_z_angle[0:feature_length, :]

    resized_thumb_acc_mag = np.reshape(resized_thumb_acc_mag, (1, -1))
    resized_index_acc_mag = np.reshape(resized_index_acc_mag, (1, -1))
    resized_wrist_acc_mag = np.reshape(resized_wrist_acc_mag, (1, -1))
    resized_thumb_x_angle = np.reshape(resized_thumb_x_angle, (1, -1))
    resized_index_x_angle = np.reshape(resized_index_x_angle, (1, -1))
    resized_hand_x_angle = np.reshape(resized_hand_x_angle, (1, -1))
    resized_hand_y_angle = np.reshape(resized_hand_y_angle, (1, -1))
    resized_hand_z_angle = np.reshape(resized_hand_z_angle, (1, -1))
    resized_wrist_x_angle = np.reshape(resized_wrist_x_angle, (1, -1))
    resized_wrist_y_angle = np.reshape(resized_wrist_y_angle, (1, -1))
    resized_wrist_z_angle = np.reshape(resized_wrist_z_angle, (1, -1))

    '''
    print(resized_thumb_acc_mag.shape)
    print(resized_index_acc_mag.shape)
    print(resized_wrist_acc_mag.shape)
    print(resized_thumb_x_angle.shape)
    print(resized_index_x_angle.shape)
    print(resized_hand_x_angle.shape)
    print(resized_hand_y_angle.shape)
    print(resized_hand_z_angle.shape)
    print(resized_wrist_x_angle.shape)
    print(resized_wrist_y_angle.shape)
    print(resized_wrist_z_angle.shape)
    '''

    return [resized_thumb_acc_mag, resized_index_acc_mag, resized_wrist_acc_mag, resized_thumb_x_angle,
            resized_index_x_angle,
            resized_hand_x_angle
               , resized_hand_y_angle, resized_hand_z_angle, resized_wrist_x_angle, resized_wrist_y_angle,
            resized_wrist_z_angle], feature_length
    # Data resizing
    # resized_thumb_acc_mag = zpl_thumb_acc_mag(1:feature_length, 1)

def butterworth_5hz(fs):

    fmax = fs/2
    LPF_cutoff = 5/fmax
    order = 5;
    [b1, a1] = signal.butter(order, LPF_cutoff, 'low')
    return b1, a1

'''
return [zpl_thumb_acc_mag, zpl_index_acc_mag, zpl_wrist_acc_mag, zpl_thumb_x_angle, zpl_index_x_angle,
            zpl_hand_x_angle
            ,zpl_hand_y_angle,zpl_hand_z_angle,zpl_wrist_x_angle,zpl_wrist_y_angle,zpl_wrist_z_angle]
            '''