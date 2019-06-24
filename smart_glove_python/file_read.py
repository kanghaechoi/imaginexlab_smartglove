import csv
from numpy import genfromtxt
import numpy as np
import time
from Users import User
from Users import UserDataframe
import glob
import pandas as pd


path = 'data/'
hand_path = 'Hand_IMU_'
wrist_path = 'Wrist_IMU_'
format_file = '.txt'
hand_data = []
wrist_data = []
UserList = []


def read_files(array_age, number_files):
    size_array = len(array_age)
    size_nb_files = len(number_files)
    if size_array != size_nb_files:
        raise Exception('There is a problem with the size of one of the data')
    for i in range(size_array):
        for j in range(1, number_files[i]+1):
            with open(path + hand_path + str(array_age[i]) + '_' + str(j) + format_file, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=' ')
                hand_read = list(csv_reader)
                hand_data.append(hand_read)
            with open(path + wrist_path + str(array_age[i]) + '_' + str(j) + format_file, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=' ')
                wrist_read = list(csv_reader)
                wrist_data.append(wrist_read)

    # print(len(hand_data));
    # print(len(wrist_data));
    return hand_data, wrist_data


def read_files_numpy(array_age, number_files):
    start = time.time()
    size_array = len(array_age)
    size_nb_files = len(number_files)

    hand_data_np = np.array([])
    wrist_data_np = np.array([])

    if size_array != size_nb_files:
        raise Exception('There is a problem with the size of one of the data')
    for i in range(size_array):
        for j in range(1, number_files[i]+1):
            hand_read = genfromtxt(path + hand_path + str(array_age[i]) + '_' + str(j) + format_file, delimiter=' ')
            wrist_read = genfromtxt(path + wrist_path + str(array_age[i]) + '_' + str(j) + format_file, delimiter=' ')
            hand_data_np = np.vstack((hand_data_np, hand_read))
            wrist_data_np = np.vstack(wrist_data_np, [wrist_read])
    elapsed = time.time() - start
    print(elapsed)
    return [hand_data_np, wrist_data_np]



def read_files_User(array_age):
    size_array = len(array_age)
    list_file_hand = []
    list_file_wrist = []
    for i in range(size_array):
        list_file_hand = glob.glob(path + hand_path+str(array_age[i])+"*"+format_file)
        list_file_wrist = glob.glob(path + wrist_path + str(array_age[i]) + "*" + format_file)
        list_file_hand.sort()
        list_file_wrist.sort()
        for j in range(0,len(list_file_hand)):
            print(list_file_hand[j])
            print(list_file_wrist[j])
            hand_read = genfromtxt(list_file_hand[j], delimiter=' ')
            wrist_read = genfromtxt(list_file_wrist[j], delimiter=' ')
            # print(path + hand_path + str(array_age[i]) + '_' + str(j) + format_file)
            usr = User(hand_read, wrist_read, array_age[i])
            UserList.append(usr)
    return UserList


def read_files_dataframe(array_age):
    size_array = len(array_age)
    with open(path + "DataFrameFileInfoHand" + format_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        columnsHands = list(csv_reader)
    with open(path + "DataFrameFileInfoWrist" + format_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        columnsWrist = list(csv_reader)
    for i in range(size_array):
        list_file_hand = glob.glob(path + hand_path+str(array_age[i])+"*"+format_file)
        list_file_wrist = glob.glob(path + wrist_path + str(array_age[i]) + "*" + format_file)
        list_file_hand.sort()
        list_file_wrist.sort()

        for j in range(0,len(list_file_hand)):
            print(list_file_hand[j])
            hand_read = pd.read_csv(list_file_hand[j], sep=' ', header=None)
            hand_read = hand_read.dropna(axis='columns')
            hand_read.columns = columnsHands[0]
            wrist_read = pd.read_csv(list_file_wrist[j],sep=' ', header=None)
            wrist_read = wrist_read.dropna(axis='columns')
            wrist_read.columns = columnsWrist[0]
            # print(path + hand_path + str(array_age[i]) + '_' + str(j) + format_file)
            usr = UserDataframe(hand_read, wrist_read, array_age[i])
            UserList.append(usr)
    return UserList

