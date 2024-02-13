import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    
    # Calculate True Positive, False Positive, False Negative, True Negative
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    # Calculate Precision
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    
    # Calculate Recall
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    
    # Calculate F1-score
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
  
    # Calculate Accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    
    return precision, recall, f1, accuracy

    
def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    
    
    count = 0
    for pred, label in zip(y_pred, y_true):
        if pred == label:
            count += 1
    accuracy = count / len(y_true)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    
    
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - ss_res / ss_tot
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    
    
    mse = np.mean((y_pred - y_true)**2)
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.mean(np.abs(y_pred - y_true))
    return mae
    