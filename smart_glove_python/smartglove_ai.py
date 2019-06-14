from dataset_generation import *
from file_read import read_files_User
from file_read import read_files
from lstm_train import lstm_train
import time
import matplotlib.pyplot as plt
# read the file

ages = [20, 60]

number_files = [24, 3]
number_test_by_people = 3
frequency  = 100


##====== Object read for graph =======
Test_subject = []
hand_data, wrist_data = read_files(ages, number_files)
for i in range(0,len(hand_data), 3):
    print("============")
    time = len(hand_data[i])/frequency
    time2 = len(hand_data[i+1])/frequency
    time3 = len(hand_data[i+2]) / frequency
    print(len(hand_data[i]))
    print(len(hand_data[i+1]))
    print(len(hand_data[i+2]))
    Test_subject.append([time,time2,time3])
for impacts in Test_subject:
    timefilteredForce = plt.plot(impacts)
    timefilteredForce = plt.xlabel('points')
    timefilteredForce = plt.ylabel('Force')
plt.show()
'''
##====== Object read =======
start = time.time()
UserList = read_files_User(ages)
data_train, data_test, labels_train, labels_test = dataset_generation(UserList)
lstm_train(data_train, labels_train, data_test, labels_test)
elapsed = time.time() - start
print(elapsed)

'''