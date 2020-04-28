import os

def run_svm(q_num):
    print('Initiating ' + q_num.upper() + ' SVM classification process...')

    # Extract features to train SVM model
    os.system('python ../src/feature_extraction.py ' + q_num + ' 20')
    os.system('python ../src/feature_extraction.py ' + q_num + ' 50')
    os.system('python ../src/feature_extraction.py ' + q_num + ' 70')

    # Normalize dataset
    os.system('python ../src/data_normalize.py ' + q_num)

    # Divide training and test dataset
    os.system('python ../src/divide_dataset.py ' + q_num + ' svm')

    # Train SVM model and validate it
    os.system('python ../src/svm_model.py ' + q_num)

    print('Completed ' + q_num.upper() + ' SVM classification process...')

    return 0


def run_rnn(q_num):
    print('Initiating ' + q_num.upper() + ' RNN classification process...')

    # Extract features to train RNN model
    os.system('python ../src/feature_extraction_seq.py ' + q_num + ' 20')
    os.system('python ../src/feature_extraction_seq.py ' + q_num + ' 50')
    os.system('python ../src/feature_extraction_seq.py ' + q_num + ' 70')

    # Combine dataset
    os.system('python ../src/join_seq_data.py ' + q_num)

    # Divide training and test dataset
    os.system('python ../src/divide_dataset.py ' + q_num + ' seq')

    # Train RNN model and validate it
    os.system('python ../src/lstm_model.py ' + q_num)

    print('Completed ' + q_num.upper() + ' RNN classification process...')

    return 0


def run_cnn(q_num):
    print('Initiating ' + q_num.upper() + ' CNN classification process...')

    # Extract features to train RNN model
    os.system('python ..src/feature_extraction_seq.py ' + q_num + ' 20')
    os.system('python ..src/feature_extraction_seq.py ' + q_num + ' 50')
    os.system('python ..src/feature_extraction_seq.py ' + q_num + ' 70')

    # Combine dataset
    os.system('python ..src/join_seq_data.py ' + q_num)

    # Divide training and test dataset
    os.system('python ..src/divide_dataset.py ' + q_num + ' seq')

    # Train CNN model and validate it
    os.system('python ..src/cnn_model.py ' + q_num)

    print('Completed ' + q_num.upper() + ' CNN classification process...')

    return 0


def research_q1(q_num):
    selected_classifier = input('Q2-1. Which classifier do you want to use? (Options: svm, rnn, or cnn)')

    if(selected_classifier == 'svm'):
        run_svm(q_num, 'SVM')

    if(selected_classifier == 'rnn'):
        run_rnn(q_num, 'RNN')

    if(selected_classifier == 'cnn'):
        run_rnn(q_num, 'CNN')

    return 0


def research_q2(q_num):
    selected_classifier = input('Q2-2. Which classifier do you want to use? (Options: svm, rnn, or cnn)')

    if (selected_classifier == 'svm'):
        run_svm(q_num, 'SVM')

    if (selected_classifier == 'rnn'):
        run_rnn(q_num, 'RNN')

    if (selected_classifier == 'cnn'):
        run_rnn(q_num, 'CNN')

    return 0


def research_q3(q_num):
    selected_classifier = input('Q2-3. Which classifier do you want to use? (Options: svm, rnn, or cnn)')

    if (selected_classifier == 'svm'):
        run_svm(q_num, 'SVM')

    if (selected_classifier == 'rnn'):
        run_rnn(q_num, 'RNN')

    if (selected_classifier == 'cnn'):
        run_rnn(q_num, 'CNN')

    return 0


if __name__ == '__main__':
    selected_question = input('Q1. Which research question do you want to confirm? (Options: q1, q2, or q3)')

    if(selected_question == 'q1'):
        research_q1(selected_question)

    if(selected_question == 'q2'):
        research_q2

    if(selected_question == 'q3'):
        research_q3

