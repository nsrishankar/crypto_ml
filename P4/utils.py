# Utility functions for PrivacyLeakage_RegressionAnalysis_P4
import os
import glob
import math
import numpy as np
import scipy as sp

def load_data(data_path):
    loaded_data=open(data_path).read().split()
    
    features=[]
    labels=[]
    
    for line in loaded_data:
        temp=line.split(',')
        tstamp=float(temp[0])
        lat=float(temp[1])
        long=float(temp[2])
        accuracy=float(temp[3])
        label=int(temp[4])

        feature=np.hstack((tstamp,lat,long,accuracy))
        features.append(feature)
        labels.append(label)
    features=np.asarray(features)
    labels=np.asarray(labels)
    
    assert len(features)==len(labels),"Feature extraction failed." # Output error message
    return features,labels


def soft_labeling(x_features,x_labels,n_classes): 
# Using the previously loaded dataset where features=[timestep,latitude,longitude,accuracy] and label=[48-52 number]
# Will output a probabilistic label of shape [x_features,n_classes]
    labels=[]
    temp=np.zeros((len(x_features),n_classes))
    for ind in range(len(x_features)):
        label_value=x_labels[ind]
        accuracy_value=x_features[ind,3]
        temp[ind,0]=int(label_value)
        label_index=int(label_value-48)
        labels.append(int(label_value))
        for j in range(5):
            if j!=int(label_index):
                temp[ind,j]=(1-accuracy_value)/(n_classes-1)
            else:
                temp[ind,j]=accuracy_value
    return np.asarray(labels),temp

def sparse2array(raw_array): 
# Converts a sparse array to a dense 1*n array (can be done with Scipy's todense())
    array_length=raw_array.getnnz()
    temp=np.zeros((1,array_length))
    for i in range(array_length):
        temp[0,i]=raw_array[0,i]
    temp=temp/np.sum(temp)
    return temp

def argmax_custom(raw_array): 
# Returns the maximum index, and the maximum value of a 1*n array
    max_index=np.argmax(raw_array)
    max_value=raw_array[0,max_index]
    return max_index,max_value

def evaluate(evaluated_dict,ground_truth_labels,ground_truth_probability):
    # Takes in the dict created of [index]:predicted_label,probability and compares the predicted label
    # with ground truth label

    # Also finds the Mean-squared error of predicted probabilities vs. given probabilities  n*1 array
    assert len(evaluated_dict)==len(ground_truth_labels), "Array lengths should be of the same size"
    count=0
    mse_error=0.0
    sum_predicted_probability=0.0
    sum_true_probability=0.0
    for key,value in evaluated_dict.items():
        predicted_label=value[0]
        predicted_probability=value[1]
        true_label=ground_truth_labels[key]
        given_probability=ground_truth_probability[key]
        
        sum_predicted_probability+=predicted_probability
        sum_true_probability+=given_probability
        
        if predicted_label==true_label:
            # Check for each row if predicted label equals ground truth label (classifier measure)
            count+=1 
            # If labels are equal, check the probability differences (regressor measure)
            mse_error+=(predicted_probability-given_probability)**2
    classifier_accuracy=float(count/len(ground_truth_labels))
    mse_sum=mse_error/count # Finding the mean-square error for the labels that were guessed correctly
    mean_predicted_probability=sum_predicted_probability/len(ground_truth_labels)
    mean_true_probability=sum_true_probability/len(ground_truth_labels)
    
    return count,classifier_accuracy,mse_sum,mean_predicted_probability,mean_true_probability