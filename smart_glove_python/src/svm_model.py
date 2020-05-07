from sklearn.svm import LinearSVC
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


def get_f1(y_pred, y_true): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val


if __name__ == '__main__':
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    IS_DEBUG = argument[1]

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

    svm_model = LinearSVC(tol=1e-5)
    svm_model.fit(train_feature, np.squeeze(train_label))

    predicted_label = svm_model.predict(test_feature).reshape((-1, 1))
    test_len = len(predicted_label)

    err_array = np.subtract(predicted_label, test_label)
    err_idx = np.where(err_array != 0)[1]

    err = round(((len(err_idx)/ test_len) * 100), 2)
    acc = 100 - err

    f1 = f1_score(test_label, predicted_label, average='macro')
    f1 = round((f1 * 100), 2)

    print('SVM model\'s accuracy is ', acc, '%')
    print('SVM model\'s F1 score is ', f1, '%')
