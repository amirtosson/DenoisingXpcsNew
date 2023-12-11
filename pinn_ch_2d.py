import numpy as np
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
import matplotlib.pyplot as plt

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


def get_first_deriv(u):
    with tf.GradientTape() as tape:
        tape.watch(u)
        T = model(u)
    T_x = tape.gradient(T, u)
    return T_x

def get_second_deriv(u):
    with tf.GradientTape() as tape:
        tape.watch(u)
        T_x = get_first_deriv(u)
    T_xx = tape.gradient(T_x, x)
    return T_xx
 


# Define the Cahn-Hilliard equation
def cahn_hilliard_equation(u, epsilon):
    with tf.GradientTape() as tape1:
        tape1.watch(u)
        c = 1 - u**2
        dc_du = tape1.gradient(c, u)
    grad_u = tape1.gradient(u, x)
    
    with tf.GradientTape() as tape2:
        tape2.watch(grad_u)
        laplacian_u = tf.reduce_sum(tf.gradients(grad_u, x, grad_ys=dc_du), axis=1)
    
    return u - epsilon * laplacian_u

# Define the loss function
def loss(model, x, epsilon, boundary_x, boundary_y):
    u = model(x)
    residual = cahn_hilliard_equation(u, epsilon)
    boundary_loss = tf.reduce_mean(tf.square(model(boundary_x) - boundary_y))
    return tf.reduce_mean(tf.square(residual)) + boundary_loss

# Generate synthetic data (you should replace this with your actual data)
x_data = np.random.rand(100, 2)  # Random points in 2D
boundary_x_data = np.random.rand(50, 2)  # Random boundary points
boundary_y_data = np.zeros((50, 1))  # Boundary condition u=0

# Convert data to TensorFlow tensors
x = tf.constant(x_data, dtype=tf.float32)
boundary_x = tf.constant(boundary_x_data, dtype=tf.float32)
boundary_y = tf.constant(boundary_y_data, dtype=tf.float32)

# Create the PINN model
model = PINN()

# Define optimization settings
optimizer = tf.optimizers.Adam(learning_rate=0.001)
epsilon = 0.01
epochs = 10000

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, epsilon, boundary_x, boundary_y)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")

# Plot the predicted solution
x_test = np.linspace(0, 1, 100)
y_test = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_test, y_test)
xy_test = np.column_stack((X.flatten(), Y.flatten()))
u_pred = model(xy_test).numpy().reshape(X.shape)

plt.contourf(X, Y, u_pred, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Predicted Solution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



#%%

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
x_data = tf.random.uniform([100,2])

boundary_x_data = np.random.rand(50, 2)  # Random boundary points
boundary_y_data = np.zeros((50, 1))  # Boundary condition u=0
bcx_tensor = tf.convert_to_tensor(boundary_x_data,  dtype=tf.float32)[:,:] 
bcy_tensor = tf.convert_to_tensor(boundary_y_data,  dtype=tf.float32)[:,:] 

optim = Adam(learning_rate=0.005)

def build_model(num_h_layers, num_nodes_per_layer):
    tf.keras.backend.set_floatx("float32")
    
    model = Sequential()
    model.add(Input((2)))
    for _ in range(0, num_h_layers):
        model.add(Dense(num_nodes_per_layer, activation='tanh', kernel_initializer="glorot_uniform" ))
    model.add(Dense(1))
    return model

model = build_model(3, 32)
epsilon = 0.01

def boundary_loss(bcs_x_tensor, bcs_y_tensor):
    predicted_Y = model(bcs_x_tensor)
    #print(predicted_Y.shape)
    #print(bcs_y_tensor.shape)
    loss_bcs =  tf.reduce_mean(tf.square(predicted_Y - bcs_y_tensor))
    return loss_bcs
 
def get_first_deriv(u_pred):
    with tf.GradientTape() as tape:
        tape.watch(u_pred)
        #print(u_pred)
        c = 1 - u_pred**2
        #print(c)
    c_x = tape.gradient(c, u_pred)
    #print(c_x)
    return c_x

def get_second_deriv(u_pred):
    with tf.GradientTape() as tape:
        tape.watch(u_pred)
        c_x = get_first_deriv(u_pred)
    c_xx = tape.gradient(c_x, u_pred)
    return c_xx, c_x


def physics_loss(u_pred):
    predectied_uxx, predectied_ux = get_second_deriv(u_pred)
    with tf.GradientTape() as tape:
        tape.watch(x_data)
        mo_all = predectied_uxx + x_data+ predectied_ux
        #print(mo_all)
    mo_all_laplacian = tape.gradient(mo_all, x_data)
    #print(mo_all_laplacian)

    loss_phy = tf.reduce_sum(mo_all_laplacian, axis=1)
    return u_pred - epsilon * loss_phy

def loss_func(x, bcs_x_tensor, bcy_tensor):
    u_pred = model(x)
    loss = boundary_loss(bcs_x_tensor, bcy_tensor) + physics_loss(u_pred)
    return loss   


def get_grads():
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss_all = loss_func(x_data, bcx_tensor, bcy_tensor)
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
        print( np.max(loss))

x_t = np.random.rand(100, 2)  # Random points in 2D

r = model.predict(x_t)
plt.plot(x_t ,r)
plt.show()




#%%
import tensorflow as tf
import numpy as np

# Define the Cahn-Hilliard equation
def cahn_hilliard(u, x, y):

    with tf.GradientTape() as tape:
        #tape.watch(u)
        tape.watch(x)
        u = u
    u_x = tape.gradient(u, x)  # Calculate the gradient with respect to x
      
    with tf.GradientTape() as tape2:
        #tape.watch(u)
        tape2.watch(y)
        u= u
    u_y = tape2.gradient(u, y)
    

    with tf.GradientTape() as tape3:
        #tape.watch(u)
        tape3.watch(x)
        u_x = u_x
    u_xx = tape3.gradient(u_x, x)  # Calculate the gradient with respect to x
        
    with tf.GradientTape() as tape4:
        #tape.watch(u)
        tape4.watch(y)
        u_yy = u_yy
    u_yy = tape4.gradient(u_y, y)    

 # Second derivative with respect to y
    
    du_dt = u_xx + u_yy - u + u**3 - u
    return du_dt

# Define the neural network architecture
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='tanh')  # Output u

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        u = self.dense3(x)
        return u

# Define loss function for the PINN
def PINN_loss(u_predicted, u_exact, x, y):
    du_dt_predicted = cahn_hilliard(u_predicted, x, y)
    loss = tf.reduce_mean(tf.square(du_dt_predicted - du_dt_exact))  # Use mean squared error
    return loss

# Generate some synthetic data (you should replace this with your actual data)
num_samples = 1000
x_data = np.random.rand(num_samples, 2) *10  # 2D input
u_exact_data = np.sin(x_data[:, 0]) * np.cos(x_data[:, 1])  # Example exact solution

# Extract x and y from the input data
x_values = x_data[:, 0]
y_values = x_data[:, 1]

# Create TensorFlow datasets
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((x_data, u_exact_data, x_values, y_values))
dataset = dataset.shuffle(buffer_size=num_samples).batch(batch_size)

# Instantiate the PINN model
model = PINN()

# Define optimization method
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for x_batch, u_exact_batch, x_values_batch, y_values_batch in dataset:
        with tf.GradientTape() as tape:
            u_predicted_batch = model(x_batch)
            loss = PINN_loss(u_predicted_batch, u_exact_batch, x_values_batch, y_values_batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}')

# After training, you can use the model to make predictions on new data
new_x = np.random.rand(10, 2)  # Example new data
predicted_u = model(new_x)
print("Predicted u:", predicted_u.numpy())






