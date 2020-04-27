import pickle
import numpy as np
import math
import random
import sys


def write_train_test(train_path, train_data, test_path, test_data):
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == '__main__':
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    MODE = argument[1]

    if(MODE == 'svm'):
        ALL_FEATURE = './pickle/' + RESEARCH_QUESTION + '/all_feature_svm.pickle'
        ALL_LABEL = './pickle/' + RESEARCH_QUESTION + '/all_label_svm.pickle'

        TRAIN_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/train_feature_svm.pickle'
        TEST_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/test_feature_svm.pickle'

        TRAIN_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/train_label_svm.pickle'
        TEST_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/test_label_svm.pickle'

        SHAPE_IDX = 0

    elif(MODE == 'seq'):
        ALL_FEATURE = './pickle/' + RESEARCH_QUESTION + '/all_feature_seq.pickle'
        ALL_LABEL = './pickle/' + RESEARCH_QUESTION + '/all_label_seq.pickle'

        TRAIN_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/train_feature_seq.pickle'
        TEST_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/test_feature_seq.pickle'

        TRAIN_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/train_label_seq.pickle'
        TEST_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/test_label_seq.pickle'

        SHAPE_IDX = 2

    with open(ALL_FEATURE, 'rb') as f:
        all_feature = pickle.load(f)

    with open(ALL_LABEL, 'rb') as f:
        all_label = pickle.load(f)

    # print(all_feature.shape[0])
    # print(all_label.shape[0])

    all_idx = np.linspace(0, (all_feature.shape[SHAPE_IDX] - 1), num=all_feature.shape[SHAPE_IDX], dtype=int)

    # print(all_idx)
    # print(len(all_idx))

    test_len = math.floor(all_feature.shape[SHAPE_IDX] * 0.15)

    # print(test_len)

    test_idx = random.sample(all_idx.tolist(), test_len)

    # print(max(test_idx))
    # print(test_idx)

    train_idx = np.delete(all_idx.reshape((1, -1)), test_idx).tolist()
    # print(max(train_idx))
    #
    # print(len(train_idx))

    # print(test_len + len(train_idx))
    if(MODE == 'svm'):
        train_feature = all_feature[train_idx, :]
        test_feature = all_feature[test_idx, :]
    elif(MODE == 'seq'):
        train_feature = all_feature[:, :, train_idx]
        test_feature = all_feature[:, :, test_idx]

    write_train_test(TRAIN_FEATURE_PATH, train_feature, TEST_FEATURE_PATH, test_feature)

    # print(train_feature.shape[0] + test_feature.shape[0])

    train_label = all_label[train_idx, :]
    test_label = all_label[test_idx, :]

    write_train_test(TRAIN_LABEL_PATH, train_label, TEST_LABEL_PATH, test_label)

    # print(train_label.shape[0] + test_label.shape[0])

    print('Divided train dataset and test dataset...')