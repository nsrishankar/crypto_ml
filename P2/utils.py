# Utility function to:
# Reshape images to be of the right format
# Find the top-k predictions for each image
# Normalize the image for display
# Load image, and predicted label for adversarial crafting
import math
import random
import cv2
import numpy as np
from numpy.random import randint, RandomState

import keras
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import preprocess_input,decode_predictions


def expand_dims(raw_image): # Create images of the form (1,raw_image.shape)
    return np.expand_dims(raw_image,axis=0)

def kpredictions(raw_image,model,k): # Run a sample image of dimension 3 through a model and return k-predictions
    model_input_image=expand_dims(raw_image)
    
    predictions=model.predict(model_input_image)
    top_k=decode_predictions(predictions,top=k)
    return ([top_k[0][i][1:3] for i in range(k)])

def load_image(file_name,model):
    loaded_image_bgr=cv2.resize(cv2.imread(file_name),(224,224)) # BGR order load
    resnet_image=np.expand_dims(loaded_image_bgr,axis=0)
    label=int(np.argmax(model.predict(resnet_image),axis=-1))
    return loaded_image_bgr,label

def normalize01(raw_image):
    im_max=np.max(raw_image)
    im_ptp=np.ptp(raw_image)
    return (raw_image-im_max)/-im_ptp