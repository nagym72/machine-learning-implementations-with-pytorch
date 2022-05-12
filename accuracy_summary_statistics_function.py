import torch
import numpy as np
import math


def ex6(logits:torch.tensor, activation_function:callable, threshold:any, targets:torch.tensor, print_stats=False):
    """
    :param logits: torch.tensor which contains the output of the last NN layer (before applying the final activation function)
    :param activation_function: The function that is used to classify the category. Will take as input logits and return a value between 0 < threshold < 1.
    :param threshold: The threshold that is used to decide if a sample belongs to class A or B (binary classification
    :param targets: An array containing the labels of the classed which are to be predicted
    :return:
    """

    #check if input array is of correct type
    if torch.is_floating_point(logits) is not True:
        raise TypeError(f"supplied array is not of correct type but instead {type(logits)}")

    #check if threshold is correct type
    if torch.is_tensor(threshold) is not True:
        raise TypeError(f"supplied threshold are not of correct type but instead {type(threshold)}")

    #check if targets is a tensor step 1)
    if torch.is_tensor(targets) is not True:
        raise TypeError(f"supplied targets are not of correct type but instead {type(targets)}")

    #check if the targets are of boolean type step 2)
    if targets.dtype != torch.bool:
        raise TypeError(f"supplied targets are not of correct type but instead {type(targets)}")

    #check if shape of input corresponds to n_samples
    if (logits.shape[0] or targets.shape[0]) != n_samples:
        raise ValueError("Incorrect amount of samples supplied")

    #if targets is not of same shape as logits
    if logits.shape != targets.shape:
        raise ValueError(f"shape of input: {logits.shape}, shape of targets: {targets.shape}")

    #check if there is at least one 1 class and at least one 0 class.
    if sum(targets) == len(targets) or sum(targets) == 0:
        raise ValueError(f"your targets contain: {sum(targets)} TRUE or {len(targets)} FALSE")

    #first we convert our final output from the last layer of the NN into [0,1] range
    y_predicted_vals = activation_function(logits)
    #y_predictor will be applied to make the decision if we classify as class 1 or 0 based on submitted threshold
    y_predictor = lambda x: 0 if x < threshold else 1
    #applying the predictor filter function and retrieving an array that stores our predictions
    y_preds = torch.tensor(np.array(list(map(y_predictor, test_arr))))

    """
       CONFUSION MATRIX NOTATION:
       TP = class a and classified as class a
       FP = class b and classified as class a 
       TN = class b and classified as class b
       FN = class a and classified as class b 
    """

    """Strategy to solve TP, FP, FN, TN:
    First we will check for all correct predictions: Out of those filtered we will sum up all 1ns (= TP = all correct classified as 1)
    e.g we have 5 correct classified instances... 4 as 1 and 1 as 0. 
    If we sum up this [0,1,1,1,1] array we get 4 (=TP) and then we will subtract this number (4) from the total length of the correct hits = 5-4 = 1 = TN.
    These cover the TRUE NEG and TRUE POS cases.
    
    For FP and FN:
    We will filter out all targets that are classified as 1 and subtract the amount of true positives from that number... which will leave us with only the FP classified ones.
    Same procedure for 0 and False negatives.
    """
    #contains all hits

    targets = targets.to(torch.int)
    hits = targets[y_preds == targets]

    #all predicted 1 values summed up will be the amount of true positives = TP .
    True_positive = sum(hits)
    True_negative = len(hits) - True_positive
    #TP and TN will sum up to len(hits)

    #FP will be all instances classified as 1 minus those that are really 1 and also classified as 1
    False_positive = len(targets[y_preds == 1]) - True_positive
    False_negative = len(targets[y_preds == 0]) - True_negative

    #accuracy defined as correct preds / total labels
    accuracy = float((True_positive + True_negative)/ len(targets))

    #sensitivity = TP/(TP+FN)
    sensitivity = True_positive/(True_positive+False_negative)

    #specificity = TN/(TN+FP)
    specificity = True_negative/(True_negative+False_positive)

    #Balanced Accuracy = (((TP/(TP+FN)+(TN/(TN+FP))) / 2  = (sensitivity + specificity) / 2
    balanced_accuracy = float((specificity + sensitivity )/2)

    confusion_matrix = [[True_positive, False_negative],[False_positive, True_negative]]

    #precision = TP / (TP+FP)
    precision = True_positive / (True_positive+False_positive)

    #F1 score = F1 = 2 * (P * Sensitivity) / (P + Sensitivity)
    F1_score = float(2*precision*sensitivity/(precision + sensitivity))

    if print_stats == True:
        print(f"Correct as class A classified instances: {True_positive.item()}")
        print(f"Correct as class B classified instances: {True_negative.item()}")
        print(f"Incorrect as class A classified instances: {False_positive.item()}")
        print(f"Incorrect as class A classified instances: {False_negative.item()}")
        print("-----------------------------------------------")
        print("Confusion matrix:")
        print("  A B")
        print("A",True_positive.item(),False_negative.item(),"\nB",False_positive.item(),True_negative.item())
        print("-----------------------------------------------")
        print(f"Accuracy overall: {accuracy*100:.2f} %")
        print(f"Balanced accuracy: {balanced_accuracy*100:.2f} %")

    return (confusion_matrix, F1_score, accuracy, balanced_accuracy)


def _testactivation_sigmoid(x):
    """Sigmoidal activation function as test
       returned values will be in range [0,1]"""
    return (1/(1+math.e**(-x)))

def _testactivation_Relu(x):
    """Rectified linear unit function as test
       returned values will be in range [0,1]"""
    return (0 if x < 0 else x)



n_samples = 11   #just for testing purpose

test_arr = torch.tensor([0,1,2,3,4,5,6,7,8,9,10], dtype=torch.float64)
y_labels = torch.tensor([0,0,0,0,1,1,1,1,1,1,1], dtype=torch.bool)

ex6(logits=test_arr, activation_function=_testactivation_sigmoid, threshold=torch.tensor(0.9, dtype=torch.float32), targets=y_labels, print_stats=True)

