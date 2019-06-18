from file_read import read_files
import matplotlib.pyplot as plt


def plotting(ages, number_files, number_test_by_people,frequency):
    # ==================Plot=======================
    # Plotting the time took by the subject for each tests
    # At the moment we still need the exact number of files
    # We are using a old function to read the files

    test_subject = []  # Array of Array[number_test_by_people]
    hand_data, wrist_data = read_files(ages, number_files)  # Read the files
    # For with a step of the number of tests
    # Eg. for 60 files and 3 test per people the for loop for 20 times
    # the actual number of users
    for i in range(0, len(hand_data), number_test_by_people):
        temp = []  # temporary variable
        # Then we store the data for each of the files within the number of tests
        for j in range(0, number_test_by_people):
            temp.append(len(hand_data[i + j]) / frequency)
        test_subject.append(temp)  # Add the temp to the full array
    for tests in test_subject:  # Used to separate the Arrays of Arrays
        timefilteredForce = plt.plot(tests)
        timefilteredForce = plt.xlabel('Tests')
        timefilteredForce = plt.ylabel('Time')
    temp = []
    # For used to create a proper X axis
    for i in range(0, number_test_by_people):
        temp.append(i)
    plt.xticks(temp)
    plt.show()
