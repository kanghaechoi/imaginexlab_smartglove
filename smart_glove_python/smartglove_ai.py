from file_read import *
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# read the file

ages = [20,60]
number_files = [90,3]
[hand_data,wrist_path] = read_files(ages,number_files);
