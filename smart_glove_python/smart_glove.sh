#! /bin/bash


run_svm() {
  echo "Initiating $2 classification process..."

  # Extract features to train SVM model
  echo `python ./src/feature_extraction.py $1 20`
  echo `python ./src/feature_extraction.py $1 50`
  echo `python ./src/feature_extraction.py $1 70`

  # Normalize dataset
  echo `python ./src/data_normalize.py $1`

  # Divide training and test dataset
  echo `python ./src/divide_dataset.py $1 svm`

  # Train SVM model and validate it
  echo `python ./src/svm_model.py $1`

  echo "Completed $2 classification process..."
}


run_rnn() {
  echo "Initiating $2 classification process..."

  # Extract features to train RNN model
  echo `python ./src/feature_extraction_seq.py $1 20`
  echo `python ./src/feature_extraction_seq.py $1 50`
  echo `python ./src/feature_extraction_seq.py $1 70`

  # Combine dataset
  echo `python ./src/join_seq_data.py $1`

  # Divide training and test dataset
  echo `python ./src/divide_dataset.py $1 seq`

  # Train RNN model and validate it
  echo `python ./src/lstm_model.py $1`

  echo "Completed $2 classification process..."
}


run_cnn() {
  echo "Initiating $2 classification process..."

  # Extract features to train RNN model
  echo `python .src/feature_extraction_seq.py $1 20`
  echo `python .src/feature_extraction_seq.py $1 50`
  echo `python .src/feature_extraction_seq.py $1 70`

  # Combine dataset
  echo `python .src/join_seq_data.py $1`

  # Divide training and test dataset
  echo `python .src/divide_dataset.py $1 seq`

  # Train CNN model and validate it
  echo `python .src/cnn_model.py $1`

  echo "Completed $2 classification process..."
}


research_q1() {
  echo "Q2-1. Which classifier do you want to use? (Options: svm, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm $1 "Research Q1 SVM"
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn $1 "Research Q1 RNN"
  elif [ $selected_classifier = "cnn" ]
  then
    run_rnn $1 "Research Q1 CNN"
  fi
}


research_q2() {
  echo "Q2-2. Which classifier do you want to use? (Options: svm, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm "Research Q2 SVM"
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn "Research Q2 RNN"
  elif [ $selected_classifier = "cnn" ]
  then
    run_cnn "Research Q2 CNN"
  fi
}


research_q3() {
  echo "Q2-3. Which classifier do you want to use? (Options: svm, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm "Research Q3 SVM"
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn "Research Q3 RNN"
  elif [ $selected_classifier = "cnn" ]
  then
    run_cnn "Research Q3 CNN"
  fi
}

main() {
  echo "Q1. Which research question do you want to confirm? (Options: q1, q2, or q3)"
  read selected_question

  if [ $selected_question = "q1" ]
  then
    research_q1 $selected_question
  elif [ $selected_question = "q2" ]
  then
    research_q2
  elif [ $selected_question = "q3" ]
  then
    research_q3
  fi
}


main