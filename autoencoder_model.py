"""
Created on 11.05.22
Author: Amir Tosson
Email: amir.tosson@uni-siegen.de  
Project: DenoisingXPCS
File : autoencoder_model.py
Class: autoencoder_model
"""
import tensorflow.keras as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Dense
from tensorflow.keras.optimizers import SGD
from datetime import datetime
import os


class Autoencoder:

    def __init__(self):
        super().__init__

    def build_model(self, input_shape=(0, 0), show_model_summary=True):
        if input_shape == (0, 0):
            return
        print("building model")
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


        # Create the model
        model = Sequential()
        model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, padding='same'))

        model.add(Conv1DTranspose(8, kernel_size=3, padding='same', activation='relu'))
        model.add(UpSampling1D(size=2))
        model.add(Conv1DTranspose(32, kernel_size=3, padding='same', activation='relu'))
        model.add(UpSampling1D(size=2))
        model.add(Conv1DTranspose(128, kernel_size=3, padding='same', activation='relu'))
        model.add(UpSampling1D(size=2))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dense(128, activation='relu'))
        model.add(Conv1D(2, kernel_size=3, activation='relu', padding='same'))
        if show_model_summary:
            model.summary()
        return model

    def train_model(self, model, input_size, input_data, output_data, epochs, batch_size, validation_split, save_model=True):
        input_data = input_data.reshape(len(input_data), input_size, 1)
        output_data = output_data.reshape(len(output_data), input_size, 1)

        lr_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=100000,
                                                               decay_rate=0.9)
        opt = SGD(learning_rate=lr_schedule)
        opt2 = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.8, beta_2=0.999, epsilon=1e-07, name="Adadelta")

        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
        hist = model.fit(input_data, output_data,
                         epochs=epochs,
                         shuffle=True,
                         batch_size=batch_size,
                         validation_split=validation_split)
        if save_model:
            now = datetime.now()
            model_version = str(now.day) + "_" + str(now.month) + "_" + str(now.hour)
            model.save('./Models/model_time_' + str(input_size) + "_v" + str(model_version))
        return hist, model_version

    def test_model(self,
                   model_version='',
                   input_size=0,
                   input_data=[]
                   ):
        l_model = tf.models.load_model('./Models/model_time_' + str(input_size) + "_v" + str(model_version))
        input_data_r = input_data.reshape(1, input_size, 1)
        reconstructed_image = l_model.predict(input_data_r)
        return reconstructed_image[0]
