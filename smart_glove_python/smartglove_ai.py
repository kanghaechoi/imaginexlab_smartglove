from file_read import *
from feature_extraction import *
from Users import User
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# read the file

ages = [20, 60]
number_files = [15, 1]
number_files = [15, 1]
##====== Classic read =======
#[hand_data, wrist_data] = read_files(ages, number_files)
#feature_extraction(hand_data, wrist_data)
##====== Numpy read =======
#[hand_data_np, wrist_data_np] = read_files_numpy(ages, number_files)
# feature_extraction(hand_data_np, wrist_data_np)

##====== Object read =======
[UserList] = read_files_User(ages, number_files)
UserList
print(1)# feature_extraction(hand_data_np, wrist_data_np)