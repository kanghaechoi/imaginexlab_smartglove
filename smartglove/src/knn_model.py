from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pickle
import numpy as np
import sys

def load_data(feature_path, label_path):
    with open(feature_path, 'rb') as f:
        feature = pickle.load(f)

    with open(label_path, 'rb') as f:
        label = pickle.load(f)

    return feature, label

if __name__ == '__main__':
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    IS_DEBUG = argument[1]
    N_NEIGH = argument[2]

    if(IS_DEBUG == 'n'):
        TRAIN_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/train_feature_norm.pickle'
        TEST_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/test_feature_norm.pickle'

        TRAIN_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/train_label_norm.pickle'
        TEST_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/test_label_norm.pickle'

    if (IS_DEBUG == 'y'):
        TRAIN_FEATURE_PATH = '../pickle/' + RESEARCH_QUESTION + '/train_feature_norm.pickle'
        TEST_FEATURE_PATH = '../pickle/' + RESEARCH_QUESTION + '/test_feature_norm.pickle'

        TRAIN_LABEL_PATH = '../pickle/' + RESEARCH_QUESTION + '/train_label_norm.pickle'
        TEST_LABEL_PATH = '../pickle/' + RESEARCH_QUESTION + '/test_label_norm.pickle'

    train_feature, train_label = load_data(TRAIN_FEATURE_PATH, TRAIN_LABEL_PATH)
    test_feature, test_label = load_data(TEST_FEATURE_PATH, TEST_LABEL_PATH)

    knn_model = KNeighborsClassifier(n_neighbors=int(N_NEIGH))
    knn_model.fit(train_feature, np.squeeze(train_label))

    predicted_label = knn_model.predict(test_feature).reshape((-1, 1))
    test_len = len(predicted_label)

    err_array = np.subtract(predicted_label, test_label)
    err_idx = np.where(err_array != 0)[1]

    err = round(((len(err_idx)/ test_len) * 100), 2)
    acc = 100 - err

    f1 = f1_score(test_label, predicted_label, average='macro')
    f1 = round((f1 * 100), 2)

    print('KNN model\'s accuracy is ', acc, '%')
    print('KNN model\'s F1 score is ', f1, '%')
