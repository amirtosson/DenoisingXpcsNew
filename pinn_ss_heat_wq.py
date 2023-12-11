#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:07:44 2023

@author: tosson
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
import time
import sys
import matplotlib.pyplot as plt



tf.random.set_seed(123)

x = tf.linspace(0.0, 1.0, 100)

bcs_x = [0.0,1.0] 
bcs_T = [0.0,0.0]
kappa = 0.5

bcs_x_tensor = tf.convert_to_tensor(bcs_x)[:,None] 
bcs_T_tensor = tf.convert_to_tensor(bcs_T)[:,None] 

optim = Adam(learning_rate=0.005)


def build_model(num_h_layers, num_nodes_per_layer):
    tf.keras.backend.set_floatx("float32")
    
    model = Sequential()
    model.add(Input(1))
    for _ in range(0, num_h_layers):
        model.add(Dense(num_nodes_per_layer, activation='tanh', kernel_initializer="glorot_uniform" ))
    model.add(Dense(1))
    return model

model = build_model(3, 32)


def boundary_loss(bcs_x_tensor, bcs_T_tensor):
    predicted_T = model(bcs_x_tensor)
    loss_bcs =  tf.reduce_mean(tf.square(predicted_T - bcs_T_tensor))
    return loss_bcs
    
def get_first_deriv(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        T = model(x)
    T_x = tape.gradient(T, x)
    return T_x

def get_second_deriv(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        T_x = get_first_deriv(x)
    T_xx = tape.gradient(T_x, x)
    return T_xx

source_func = lambda x: (15*x-2)/kappa

def physics_loss(x):
    predectied_Txx = get_second_deriv(x)
    loss_phy = tf.reduce_mean(tf.square(predectied_Txx + source_func(x)))
    return loss_phy

def loss_func(x, bcs_x_tensor, bcs_T_tensor):
    loss = boundary_loss(bcs_x_tensor, bcs_T_tensor) + physics_loss(x)
    return loss    


def get_grads():
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss_all = loss_func(x, bcs_x_tensor, bcs_T_tensor)
    g = tape.gradient(loss_all, model.trainable_variables)
    return loss_all, g
        
def training_step():
    loss, grads = get_grads()
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return loss


n_iterations = 5000 

for i in range(0, n_iterations+1):
    loss = training_step()
    if i % 100 == 0:
        print("Epoch {:05d}: loss= {:10.8e}".format(i, loss))

x_t = tf.linspace(0.0, 1.0, 200)
r = model.predict(x_t)
plt.plot(x_t.numpy() ,r)
plt.show()





































