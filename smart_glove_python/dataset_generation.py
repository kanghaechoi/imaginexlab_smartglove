import random
from sklearn.model_selection import train_test_split
from Users import User
import numpy as np
import pandas as pd
def dataset_generation(userlist):
    ratio_test = 0.8
    shufle = False
    data_user = []
    label_user = []
    userlist.sort(key=lambda x: x.length, reverse=True)
    if(shufle):
        print(1)
        random.shuffle(userlist)
        print(1)
    for i in range(len(userlist)):
        data_user.append(userlist[i].features)
        label_user.append(userlist[i].labels)
    data_train, data_test, labels_train, labels_test = train_test_split(data_user, label_user, test_size=0.20)
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    data_user = np.array(data_user)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    data_train = np.transpose(data_train)
    data_test = np.transpose(data_test)
    print(data_train[0][0])
    print('data_train = ')
    print(np.shape(data_train))
    print('data_test = ')
    print(np.shape(data_test))
    print('labels_train = ')
    print(np.shape(labels_train))
    print('labels_test = ')
    print(np.shape(labels_test))
    return data_train, data_test, labels_train, labels_test
    print((1))
