from dataset_generation import *
from file_read import read_files_User
from lstm_train import lstm_train
import time
# read the file

ages = [20, 60]
number_files = [15, 3]





##====== Object read =======
start = time.time()
UserList = read_files_User(ages, number_files)
data_train, data_test, labels_train, labels_test = dataset_generation(UserList)
lstm_train(data_train, labels_train, data_test, labels_test)
elapsed = time.time() - start
print(elapsed)

