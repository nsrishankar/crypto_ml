# Function to help with model testing
import os
import glob
import math
import numpy as np
import scipy as sp
import time

from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from skmultilearn.problem_transform import BinaryRelevance

from utils import load_data,soft_labeling,sparse2array,argmax_custom,evaluate

def test_model(pickle_file,test_features,test_labels):
    trained_classifier=joblib.load(pickle_file) 
    print("Loaded trained Decision Tree Regression Model.")
    
    test_predictions_proba={}
    n_classes=5
    
    X_test=test_features[:,1:3]
    Y_test_hard,Y_test_soft=soft_labeling(test_features,test_labels,n_classes)
    print("Obtained testing features and labels.")
    
    for ind in range(len(test_features)):
        predictions=trained_classifier.predict(X_test[ind])
        dense_predictions=sparse2array(predictions)
        max_index,max_value=argmax_custom(dense_predictions)
        test_predictions_proba[ind]=[max_index+48,max_value]
    
    print("Obtained predictions")
    correct_count,classifier_accuracy,mse_error,mean_pred_probs,mean_true_probs=evaluate(evaluated_dict=test_predictions_proba,
                                                     ground_truth_labels=Y_test_hard,
                                                     ground_truth_probability=Y_test_soft)
    print(correct_count,classifier_accuracy,mse_error,mean_pred_probs,mean_true_probs)
    print("Classification Accuracy: {} % from 336 datapoints ".format(round(100*classifier_accuracy,3)))
    print("Mean Predicted Probability is {}%".format(round(100*mean_pred_probs,3)))
    
    return test_predictions_proba
   
    
if __name__=="__main__":
    saved_model='trained_DTR.pkl'
    #test_features=0
    #test_labels=0
    test_predictions_proba=test_model(saved_model,test_features,test_labels)