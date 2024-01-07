from math import log2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def int_to_binary_cors(integer, max_dimensions):
    cors = []
    if integer != 0:
        for i in range(int(log2(integer)+1)):
            cors.append(integer%2)
            integer = (integer)//2
    for _ in range(max_dimensions-len(cors)):
        cors.append(0)
    return cors

def align_prediction_keys(original_labels, predicted_labels):
    remaining_pred = list(set(predicted_labels))
    # print(original_labels)

    n_org_labels = {}
    for i in sorted(np.unique(original_labels)):
        key = len(np.where(original_labels==i)[0])
        n_org_labels[i]= key

    print(n_org_labels)

    queue_dict = dict(sorted(n_org_labels.items(), key=lambda item: item[1], reverse=True))
    print(queue_dict)

    mapping = {}
    for label in queue_dict.keys():
        label_answers =list(predicted_labels[np.where(original_labels==label)])
        score = {p: label_answers.count(p) for p in remaining_pred}
        max_key = max(score, key=lambda k: score[k])
        mapping[max_key] = label
        remaining_pred = [r for r in remaining_pred if r != max_key]
    print(mapping)
    # print(predicted_labels)
    predicted_labels = np.array([mapping[p] for p in predicted_labels])
    return predicted_labels

def sort_predictions(original_X, predicted):
    pred_X, pred_y = predicted
    sorted_ids = []
    for org in original_X:
        for i, pred in enumerate(pred_X):
            if all(x == y for x, y in zip(org, pred)):
                sorted_ids.append(i)
    sorted_new_X = [pred_X[i] for i in sorted_ids]
    sorted_new_y = [pred_y[i] for i in sorted_ids]

    return [np.array(sorted_new_X), np.array(sorted_new_y)]

def compute_confusion_matrix_and_metrics(y_true, y_pred, save_path=None, class_labels=None):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    if class_labels is None:
        class_labels = sorted(np.unique(y_true))
    confusion_mat = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', labels=class_labels)
    recall = recall_score(y_true, y_pred, average='weighted', labels=class_labels)
    f1 = f1_score(y_true, y_pred, average='weighted', labels=class_labels)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=class_labels, yticklabels=class_labels)
    ax[0].set_title('Confusion Matrix')
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]
    sns.barplot(x=metrics_values, y=metrics_names, ax=ax[1])
    ax[1].set_title('Metrics')
    ax[1].set_xlim(0.0, 1.0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    plt.close()
    return confusion_mat, metrics