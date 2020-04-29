#! /bin/bash


run_svm() {
  QUESTION=$1
  OS=$2

  echo "Initiating $QUESTION SVM classification process on $OS..."

  # Extract features to train SVM model
  python ./src/feature_extraction.py $QUESTION 20 $OS
  python ./src/feature_extraction.py $QUESTION 50 $OS
  python ./src/feature_extraction.py $QUESTION 70 $OS

  # Normalize dataset
  python ./src/data_normalize.py $QUESTION $OS

  # Divide training and test dataset
  python ./src/divide_dataset.py $QUESTION svm $OS

  # Train SVM model and validate it
  python ./src/svm_model.py $QUESTION $OS

  echo "Completed $QUESTION SVM classification process on $OS..."
}


run_rnn() {
  QUESTION=$1
  OS=$2

  echo "Initiating $QUESTION RNN classification process on $OS..."

  # Extract features to train RNN model
  python ./src/feature_extraction_seq.py $QUESTION 20 $OS
  python ./src/feature_extraction_seq.py $QUESTION 50 $OS
  python ./src/feature_extraction_seq.py $QUESTION 70 $OS

  # Combine dataset
  python ./src/join_seq_data.py $QUESTION $OS

  # Divide training and test dataset
  python ./src/divide_dataset.py $QUESTION seq $OS

  # Train RNN model and validate it
  python ./src/lstm_model.py $QUESTION $OS

  echo "Completed $QUESTION RNN classification process on $OS..."
}


run_cnn() {
  QUESTION=$1
  OS=$2

  echo "Initiating $QUESTION CNN classification process on $OS..."

  # Extract features to train RNN model
  python .src/feature_extraction_seq.py $QUESTION 20 $OS
  python .src/feature_extraction_seq.py $QUESTION 50 $OS
  python .src/feature_extraction_seq.py $QUESTION 70 $OS

  # Combine dataset
  python .src/join_seq_data.py $QUESTION $OS

  # Divide training and test dataset
  python .src/divide_dataset.py $QUESTION seq $OS

  # Train CNN model and validate it
  python .src/cnn_model.py $QUESTION $OS

  echo "Completed $QUESTION CNN classification process on $OS..."
}


research_q1() {
  QUESTION=$1
  OS=$2

  echo "Q2-1. Which classifier do you want to use? (Options: svm, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm $QUESTION $OS
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn $QUESTION $OS
  elif [ $selected_classifier = "cnn" ]
  then
    run_rnn $QUESTION $OS
  fi
}


research_q2() {
  QUESTION=$1
  OS=$2

  echo "Q2-2. Which classifier do you want to use? (Options: svm, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm $QUESTION $OS
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn $QUESTION $OS
  elif [ $selected_classifier = "cnn" ]
  then
    run_cnn $QUESTION $OS
  fi
}


research_q3() {
  QUESTION=$1
  OS=$2

  echo "Q2-3. Which classifier do you want to use? (Options: svm, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm $QUESTION $OS
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn $QUESTION $OS
  elif [ $selected_classifier = "cnn" ]
  then
    run_cnn $QUESTION $OS
  fi
}

main() {
  echo "Q1. Which research QUESTION do you want to confirm? (Options: q1, q2, or q3)"
  read SELECTED_QUESTION

  UNIX="unix"

  if [ $SELECTED_QUESTION = "q1" ]
  then
    research_q1 $SELECTED_QUESTION $UNIX
  elif [ $SELECTED_QUESTION = "q2" ]
  then
    research_q2 $SELECTED_QUESTION $UNIX
  elif [ $SELECTED_QUESTION = "q3" ]
  then
    research_q3 $SELECTED_QUESTION $UNIX
  fi
}


main