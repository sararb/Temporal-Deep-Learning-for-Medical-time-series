from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['flat_accuracy', 'recall_m', 'precision', 'f1_score_m', 'auc_score']


def flat_accuracy(preds, labels):
    preds = preds[0]
    preds = F.softmax(preds, dim=1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    acc = float(np.sum(pred_flat == labels_flat)) / len(labels_flat)
    return torch.tensor(acc)


def recall_m(preds, labels):
    preds = preds[0]
    preds = F.softmax(preds, dim=1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    recall = recall_score(y_true = labels_flat, y_pred = pred_flat, average='binary')
    return torch.tensor(recall)


def precision(preds, labels):
    preds = preds[0]
    preds = F.softmax(preds, dim=1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    recall = precision_score(y_true = labels_flat, y_pred = pred_flat, average='binary')
    return torch.tensor(recall)


def f1_score_m(preds, labels):
    preds = preds[0]
    preds = F.softmax(preds, dim=1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(preds, axis = 1)
    labels_flat = labels
    f1_score_ = f1_score(y_true = labels_flat, y_pred = pred_flat, average='binary')
    return torch.tensor(f1_score_)


def auc_score(preds, labels):
    preds = F.softmax(preds[0], dim=1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    labels_flat = labels
    proba_1 = preds[:, 1]
    auc = roc_auc_score(y_true = labels_flat, y_score = proba_1)
    return torch.tensor(auc)





