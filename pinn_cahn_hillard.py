import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# Generate training data
N = 100
x = np.linspace(0, 1, N).reshape(-1, 1)
u_initial = np.sin(np.pi * x)
u_initial = tf.convert_to_tensor(u_initial)[:,None] 
x = tf.constant(x, dtype=tf.float32)


# Define the Cahn–Hilliard equation
def cahn_hilliard(u):
    with tf.GradientTape() as tape:
        tape.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
        u_x = tape2.gradient(u, x)
    laplacian = tape.gradient(u_x, x)
    du_dt = tf.reduce_sum(tf.square(laplacian) - u + tf.square(u) - tf.pow(u, 3))
    return du_dt

# Create the neural network for PINN
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)



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


# Define loss function
def loss_fn(u, x):
    u_pred = model(u)
    du_dt_pred = cahn_hilliard(u_pred)
    return tf.reduce_mean(tf.square(du_dt_pred))



# Create the PINN model
model = PINN()

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 10000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = loss_fn(u_initial, x)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Evaluate the trained model
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
x_test = tf.constant(x_test, dtype=tf.float32)
u_pred = model(x_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_test, u_pred, label='Predicted')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Cahn–Hilliard Equation - PINN')
plt.legend()
plt.show()


#%%

import tensorflow as tf
import numpy as np

# Define the Cahn-Hilliard equation
def cahn_hilliard(u):
    grad_u = tf.gradients(u, x)[0]
    grad2_u = tf.gradients(grad_u, x)[0]
    return grad2_u - u + u**3

# Create a neural network model
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# Define the loss function for the PINN
def loss_fn(model, x, u):
    with tf.GradientTape() as tape:
        tape.watch(x)
        predicted_u = model(x)
        residual = cahn_hilliard(predicted_u)
        loss = tf.reduce_mean(tf.square(residual))
    return loss

# Generate some training data (you can replace this with your data)
x_train = np.linspace(0, 1, 100).reshape(-1, 1)
u_train = np.random.rand(100, 1)

# Create the PINN model and optimizer
model = create_model(input_dim=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
for epoch in range(1000):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_train, u_train)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Evaluate the model
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
u_pred = model(x_test)

# Plot the results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_test, u_pred, label='Predicted')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()



