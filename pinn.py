#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:25:00 2023

@author: tosson
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Dense, Dropout, Flatten

print(tf.config.list_physical_devices('GPU'))

def generate_data():
    print('Starting generating dataset:')
    dg = DataGenerator(64)
    return dg.generate_data_with_random_parameters(1, 100000, beta_max=1.2, beta_min=0.9, gamma_max=0.7, gamma_min=0.3,
                                                   constant_max=2.7, constant_min=0.0, save_data=False,
                                                   visualize_data=False)



# Define the 1D diffusion equation with an exponential source term
def physics_equation(x, u):
    # The 1D diffusion equation with an exponential source term
    diffusion_coefficient = 0.01
    source_term = tf.exp(-x)

    # Partial derivatives
    du_dx = tf.gradients(u, x)[0]
    du_dt = diffusion_coefficient * tf.gradients(du_dx, x)[0] - source_term

    return du_dt

# Define the neural network architecture
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense_output = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        model = Sequential(name=model_name)
        model.add(Dense(64, activation='sigmoid', input_shape=input_shape))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))
        return model
    
    
    
def loss_function(u_pred, x_bc, u_bc):
    # Physics-informed loss
    physics_loss = tf.reduce_mean(tf.square(physics_equation(x_bc, u_pred)))

    # Boundary condition loss
    bc_loss = tf.reduce_mean(tf.square(u_pred - u_bc))

    return physics_loss + bc_loss

def generate_training_data(num_points, x_min, x_max):
    x_bc = np.random.uniform(x_min, x_max, (num_points, 1))
    u_bc = np.exp(-x_bc)  # Exact solution for the exponential source term
    return x_bc, u_bc

# Training parameters
num_points = 1000
x_min, x_max = 0.0, 1.0

x_bc, u_bc = generate_training_data(num_points, x_min, x_max)



#u_bc, x_bc  = generate_data()

pinn_model = PINN()
pinn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_function)

history = pinn_model.fit(x_bc, u_bc, epochs=500, verbose=2)



plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Dense, Dropout, Flatten



class PINNModel(tf.keras.Model):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.conv1 = Conv1D(32, kernel_size=3, activation='relu', input_shape=(64, 1))
        self.flatten = Conv1D(32, kernel_size=3, padding='same', activation='relu')
        self.dense1 = Conv1DTranspose(32, kernel_size=3, padding='same', activation='relu')
        self.dense2 = Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


# Define the loss function for the PINN
def physics_loss(y_true, y_pred):
    # Add your physics-informed loss term here
    # For example, you can use an exponential function
    exp_loss = tf.reduce_mean(tf.exp(y_pred))
    return exp_loss


    loss = tf.reduce_mean(tf.square(u_pred - u_exact)) + tf.reduce_mean(tf.square(f_pred))
    return loss
def generate_data():
    print('Starting generating dataset:')
    dg = DataGenerator(64)
    return dg.generate_data_with_random_parameters(1, 100000, beta_max=1.2, beta_min=0.9, gamma_max=0.7, gamma_min=0.3,
                                                   constant_max=2.7, constant_min=0.0, save_data=False,
                                                   visualize_data=False)



output_data, input_data  = generate_data()

#x_train = np.sort(np.random.uniform(-1, 1, num_points)[:, None])
#u_exact = np.exp(-x_train**2)  # Example exponential function

# Convert data to TensorFlow tensors
#x_train_tf = tf.convert_to_tensor(input_data, dtype=tf.float32)
#u_exact_tf = tf.convert_to_tensor(output_data, dtype=tf.float32)

# Instantiate the PINN model´
#model = PhysicsInformedNN()

pinn_model = PINNModel()
#generator2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

pinn_model.compile(optimizer='adam', loss=physics_loss)
#pinn_model.summary() 

# Train the model with your data
pinn_model.fit(input_data, output_data, epochs=10, batch_size=32)

model = Sequential(name="model_name")
model.add(Conv1D(128, kernel_size=3,padding='same', input_shape=(64,1)))
#model.add(MaxPooling1D(pool_size=2, padding='same'))
#model.add(Dropout(0.50))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))
model.summary()

model.compile(optimizer='adam', loss=physics_loss)
model.fit(input_data, output_data, epochs=10, batch_size=32)
# Define optimizer

# Training loop

# Plot the results
x_test = x_train[10]
u_pred = model(tf.convert_to_tensor(x_test, dtype=tf.float32)).numpy()

plt.plot(x_train, u_exact, 'r-', label='Exact Solution')
plt.plot(x_test, u_pred, 'b--', label='PINN Prediction')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()


#%%
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# Define the Physics-Informed Neural Network (PINN) model
def PhysicsInformedNN():
    model = tf.keras.Sequential()
    # Encoder (Autoencoder)

    model.add( layers.Conv1D(16, kernel_size=3, activation='relu', padding='same', input_shape=(64,1)))
    model.add(  layers.MaxPooling1D(pool_size=2, padding='same'))
    model.add(   layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(   layers.MaxPooling1D(pool_size=2, padding='same'))
    model.add(   layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(   layers.MaxPooling1D(pool_size=2, padding='same'))
    
    model.add(   layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(   layers.UpSampling1D(size=2))
    model.add(  layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(  layers.UpSampling1D(size=2))
    model.add(  layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'))
    model.add( layers.UpSampling1D(size=2))
    model.add(  layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))
    
    # Physics-informed constraint layer (Exponential Function)
    #model.add(  layers.Lambda(lambda x: tf.math.exp(-x)))
    model.summary()          
    #model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
            
def physics_loss(y_true, y_pred):
    # Add your physics-informed loss term here
    # For example, you can use an exponential function
    exp_loss = tf.reduce_mean(tf.exp(y_true- y_pred))
    return exp_loss
          



# Create an instance of the PINN model
pinn_model = PhysicsInformedNN()
# Compile the model (you may need to adjust the loss function and optimizer based on your problem)
pinn_model.compile(optimizer='adam', loss=physics_loss)
# Print the model summary
pinn_model.fit(input_data, output_data, epochs=2, batch_size=32)


#%%

def ode_system(t, net):
    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    t_0 = tf.zeros((1,1))
    one = tf.ones((1,1))

    with tf.GradientTape() as tape:
        tape.watch(t)

        u = net(t)
        u_t = tape.gradient(u, t)

    ode_loss = u_t - tf.math.cos(2*np.pi*t)
    IC_loss = net(t_0) - one

    square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    total_loss = tf.reduce_mean(square_loss)

    return total_loss

NN = PhysicsInformedNN()
optm = tf.keras.optimizers.Adam(learning_rate = 0.001)
train_t = (np.array([0., 0.025, 0.475, 0.5, 0.525, 0.9, 0.95, 1., 1.05, 1.1, 1.4, 1.45, 1.5, 1.55, 1.6, 1.95, 2.])).reshape(-1, 1)
train_loss_record = []


for itr in range(6000):
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN)
        train_loss_record.append(train_loss)

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 == 0:
        print(train_loss.numpy())
        
        
        
#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random


NN = tf.keras.models.Sequential([
    #tf.keras.layers.Input((64)),
    tf.keras.layers.Dense(units = 64, activation = 'sigmoid', input_shape= (64,1)),
    tf.keras.layers.Dense(units = 32, activation = 'sigmoid'),
    tf.keras.layers.Dense(units = 32, activation = 'sigmoid'),
    tf.keras.layers.Dense(units = 1)
])

NN.summary()
optm = tf.keras.optimizers.Adam(learning_rate = 0.01)
def ode_system(t, n,  net):
    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    n = n.reshape(-1,1)
    n = tf.constant(n, dtype = tf.float32)
    t_0 = tf.zeros((1,1))
    one = tf.ones((1,1))

    with tf.GradientTape() as tape:
        tape.watch(n)

        u = net(n)
        u_t = tape.gradient(u, net.trainable_variables)
        #print(u_t)

    ode_loss = u_t - tf.math.exp(n)
    IC_loss = net(t_0) - one

    square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    total_loss = tf.reduce_mean(square_loss)

    return total_loss

TIME_DATA = np.logspace(-2, 2, 64)
train_t = 1 + 0.5 * np.exp(-(0.5 * TIME_DATA))


plt.plot(train_t)
plt.show()
#train_t = (np.array([0., 0.025, 0.475, 0.5, 0.525, 0.9, 0.95, 1., 1.05, 1.1, 1.4, 1.45, 1.5, 1.55, 1.6, 1.95, 2.])).reshape(-1, 1)
train_loss_record = []
number_images = np.linspace(100, 10, 1)
noise = np.random.normal(0, 1, train_t.shape)
g2_noisy = train_t + 1 * noise/(number_images + 0)
g2_noisy = g2_noisy.reshape((64,1))
plt.plot(g2_noisy)
plt.show()

for itr in range(1000):
    with tf.GradientTape() as tape:

        train_loss = ode_system(train_t, g2_noisy, NN)
        train_loss_record.append(train_loss)

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 == 0:
        print(train_loss.numpy())

plt.figure(figsize = (10,8))
plt.plot(train_loss_record)
plt.show()


test_t = 1 + 0.5 * np.exp(-(0.4 * TIME_DATA)).reshape((64,1))
number_images = np.linspace(100, 10, 1)
noise = np.random.normal(0, 1, test_t.shape)
g2_noisy_t = test_t + 5 * noise/(number_images + 0) 
plt.plot(g2_noisy_t)
plt.show()
#train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
#true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1
pred_u = NN.predict(g2_noisy_t.reshape((64,1))).ravel()

plt.figure(figsize = (10,8))
#plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(g2_noisy,label = 'N')
plt.plot(test_t, '-k',label = 'True')
plt.plot(pred_u, '--r', label = 'Prediction')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.show()