import random
from sklearn.model_selection import train_test_split
from Users import User

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
    return data_train, data_test, labels_train, labels_test
    print((1))