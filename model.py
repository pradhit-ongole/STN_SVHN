import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from sampler2 import BilinearInterpolation
from utils2 import get_initial_weights


def STN_Model(input_shape=(32, 32, 3), sampling_size=(32, 32,3), num_classes=10, reg=0.00, drop_rate=0.00):
    # Input
    STN_Input = keras.Input(shape=input_shape, name = 'STN_Input')

    # Layers for localization network
    locnet = layers.Conv2D(32, (5,5), activation = 'relu', padding = 'same')(STN_Input)
    locnet = layers.MaxPool2D(pool_size=(2, 2), padding = 'same')(locnet)
    locnet = layers.Conv2D(32, (5,5), activation = 'relu')(locnet)
    locnet = layers.MaxPool2D(pool_size=(2, 2), padding = 'same')(locnet)
    locnet = layers.Flatten()(locnet)
    locnet = layers.Dense(32)(locnet)
    locnet = layers.Activation('tanh')(locnet)
    
    locnet = layers.Dense(32)(locnet)

    locnet = layers.Activation('tanh')(locnet)
    locnet = layers.Dropout(drop_rate)(locnet)
    weights = get_initial_weights(32)
    locnet = layers.Dense(6, weights=weights)(locnet)

    # Grid generator and bilenear interpolator layer
    sampler = BilinearInterpolation(sampling_size)([STN_Input, locnet])

    # Classification layer
    classifier = layers.Conv2D(48, (5, 5), padding='same', activation = 'relu')(sampler)
    classifier = layers.MaxPool2D(pool_size=(2, 2), padding = 'same')(classifier)
    classifier = layers.Conv2D(64, (5, 5), activation = 'relu', padding = 'same')(classifier)
    classifier = layers.Conv2D(128, (5,5), activation = 'relu', padding = 'same')(classifier)
    classifier = layers.Flatten()(classifier)
    classifier = layers.Dense(256)(classifier)
    classifier = layers.Activation('relu')(classifier)
    classifier = layers.Dropout(drop_rate)(classifier)
    classifier = layers.Dense(num_classes)(classifier)
    classifier_output = layers.Activation('softmax')(classifier)

    model = keras.Model(inputs=STN_Input, outputs=classifier_output)
    
    return model