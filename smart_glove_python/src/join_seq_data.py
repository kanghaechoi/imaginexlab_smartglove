import numpy as np
import pickle
import sys

from sklearn import preprocessing

def read_feature(path_1, path_2, path_3):
    with open(path_1, 'rb') as f:
        data_1 = pickle.load(f)

    with open(path_2, 'rb') as f:
        data_2 = pickle.load(f)

    with open(path_3, 'rb') as f:
        data_3 = pickle.load(f)

    all_data = np.dstack((data_1, data_2, data_3))

    return all_data


def read_label(path_1, path_2, path_3):
    with open(path_1, 'rb') as f:
        data_1 = pickle.load(f)

    with open(path_2, 'rb') as f:
        data_2 = pickle.load(f)

    with open(path_3, 'rb') as f:
        data_3 = pickle.load(f)

    all_data = np.concatenate((data_1, data_2, data_3))

    return all_data


def zero_to_one(array):
    min_max_scaler = preprocessing.MinMaxScaler()
    array_transformed = min_max_scaler.fit_transform(array)

    return array_transformed

if __name__ == '__main__':
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    OS = argument[1]

    # RESEARCH_QUESTION = 'q1'
    # OS = 'unix'

    if(OS == str('unix')):
        FEATURE_20 = './pickle/' + RESEARCH_QUESTION + '/20_feature_seq.pickle'
        FEATURE_50 = './pickle/' + RESEARCH_QUESTION + '/50_feature_seq.pickle'
        FEATURE_70 = './pickle/' + RESEARCH_QUESTION + '/70_feature_seq.pickle'

        LABEL_20 = './pickle/' + RESEARCH_QUESTION + '/20_label_seq.pickle'
        LABEL_50 = './pickle/' + RESEARCH_QUESTION + '/50_label_seq.pickle'
        LABEL_70 = './pickle/' + RESEARCH_QUESTION + '/70_label_seq.pickle'

        FEATURE_ALL = './pickle/' + RESEARCH_QUESTION + '/all_feature_seq.pickle'
        LABEL_ALL = './pickle/' + RESEARCH_QUESTION + '/all_label_seq.pickle'

    if (OS == str('windows')):
        FEATURE_20 = '../pickle/' + RESEARCH_QUESTION + '/20_feature_seq.pickle'
        FEATURE_50 = '../pickle/' + RESEARCH_QUESTION + '/50_feature_seq.pickle'
        FEATURE_70 = '../pickle/' + RESEARCH_QUESTION + '/70_feature_seq.pickle'

        LABEL_20 = '../pickle/' + RESEARCH_QUESTION + '/20_label_seq.pickle'
        LABEL_50 = '../pickle/' + RESEARCH_QUESTION + '/50_label_seq.pickle'
        LABEL_70 = '../pickle/' + RESEARCH_QUESTION + '/70_label_seq.pickle'

        FEATURE_ALL = '../pickle/' + RESEARCH_QUESTION + '/all_feature_seq.pickle'
        LABEL_ALL = '../pickle/' + RESEARCH_QUESTION + '/all_label_seq.pickle'

    all_feature = read_feature(FEATURE_20, FEATURE_50, FEATURE_70)
    # all_feature_norm = zero_to_one(all_feature)

    # print(all_feature_norm)

    all_label = read_label(LABEL_20, LABEL_50, LABEL_70)

    # print(all_label)

    with open(FEATURE_ALL, 'wb') as f:
        pickle.dump(all_feature, f, pickle.HIGHEST_PROTOCOL)

    with open(LABEL_ALL, 'wb') as f:
        pickle.dump(all_label, f, pickle.HIGHEST_PROTOCOL)

    print('Sequential data is joined together...')