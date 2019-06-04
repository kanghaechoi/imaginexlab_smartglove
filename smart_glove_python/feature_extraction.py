import numpy as np


def feature_extraction(hand_data, wrist_data):
    x = hand_data
    y = wrist_data
    # hand_data = np.array(hand_data)
    thumb_acc = [row[0] for row in x]
    for i in range(len(hand_data)):
        thumb_acc = hand_data[7:9][i]
        x=x
    return [hand_data, wrist_data]
