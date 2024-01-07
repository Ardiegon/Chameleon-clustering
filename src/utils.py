from math import log2
import numpy as np
from sklearn.metrics import confusion_matrix


def int_to_binary_cors(integer, max_dimensions):
    cors = []
    if integer != 0:
        for i in range(int(log2(integer)+1)):
            cors.append(integer%2)
            integer = (integer)//2
    for _ in range(max_dimensions-len(cors)):
        cors.append(0)
    return cors

def compute_confusion_matrix(original_labels, predicted_labels):
    confusion_mat = confusion_matrix(original_labels, predicted_labels)
    mapping = {}
    for i in range(len(confusion_mat)):
        original_label = np.argmax(confusion_mat[i])
        mapping[i] = original_label
    mapped_predicted_labels = np.array([mapping[label] for label in predicted_labels])
    return confusion_mat, mapped_predicted_labels

def align_prediction_keys(original_labels, predicted_labels):
    remaining_pred = list(set(predicted_labels))
    n_org_labels = [len(np.where(original_labels==u)[0]) for u in sorted(np.unique(original_labels))]
    mapping = {}
    for id, i in enumerate(sorted(n_org_labels, reverse=True)):
        label = n_org_labels[id:].index(i) + id
        label_answers =list(predicted_labels[np.where(original_labels==label)])
        score = {p: label_answers.count(p) for p in remaining_pred}
        max_key = max(score, key=lambda k: score[k])
        mapping[max_key] = label
        remaining_pred = [r for r in remaining_pred if r != max_key]
    predicted_labels = np.array([mapping[p] for p in predicted_labels])
    return predicted_labels

def sort_predictions(original_X, predicted):
    new_X, new_y = predicted
    
    sorted_ids = []
    for x in original_X:
        sorted_ids.append(np.where(new_X==x)[0][0])

    sorted_new_X = [new_X[i] for i in sorted_ids]
    sorted_new_y = [new_y[i] for i in sorted_ids]
    return [np.array(sorted_new_X), np.array(sorted_new_y)]
