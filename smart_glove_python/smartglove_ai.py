from dataset_generation import *
from file_read import *
from lstm_train import lstm_train
import time
from plotting import plotting

ages = [20, 60] # Differents class of ages, used for the supervised learning process
number_files = [51, 3]  # Number of files according the the ages variable
number_test_by_people = 3 # Actual number of tests by people
frequency = 100  # Number of line per seconds in the txt file
# plotting(ages,number_files,number_test_by_people,frequency) # plotting

'''
##====== Object read =======
start = time.time() # Mesure of much time the process take
UserList = read_files_User(ages) # read and store the features into a list of users
data_train, data_test, labels_train, labels_test = dataset_generation(UserList)
lstm_train(data_train, labels_train, data_test, labels_test) # Launch the network
print(time.time() - start) # Print of much time the process took
'''

UserList = read_files_dataframe(ages) # read and store the features into a list of users