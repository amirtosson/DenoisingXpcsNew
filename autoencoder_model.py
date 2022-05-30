"""
Created on 11.05.22
Author: Amir Tosson
Email: amir.tosson@uni-siegen.de  
Project: DenoisingXPCS
File : autoencoder_model.py
Class: autoencoder_model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
import tensorflow.keras as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from datetime import datetime
import glob

from tensorflow.keras.utils import plot_model


  
import tensorflow as tfs
    
    


class Autoencoder:

    def __init__(self):
        print("Num GPUs Available: ", len(tfs.config.experimental.list_physical_devices('GPU')))
        #os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        #tfs.debugging.set_log_device_placement(False)
        #tfs.get_logger().setLevel('ERROR')
        #print(tfs.test.is_built_with_cuda())
        super().__init__

    def build_model(self, input_shape=(0, 0), show_model_summary=True, model_name='sequential'):
        if input_shape == (0, 0):
            return
        print("building model")
        #os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


        # Create the model
        model = Sequential(name=model_name)
        model.add(Conv1D(128, kernel_size=3,padding='same', input_shape=input_shape))
        #model.add(MaxPooling1D(pool_size=2, padding='same'))
        #model.add(Dropout(0.50))
        model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2, padding='same'))
        #model.add(Dropout(0.50))
        #model.add(Conv1D(1, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2, padding='same'))
        #model.add(Conv1D(4, kernel_size=2, padding='same', activation='relu', kernel_initializer='he_uniform'))

        #model.add(Dropout(0.50))
        #model.add(Conv1DTranspose(1, kernel_size=2, padding='same', activation='relu'))
        #model.add(UpSampling1D(size=2))
        #model.add(Dropout(0.50))
        model.add(Conv1DTranspose(32, kernel_size=3, padding='same', activation='relu'))
        #model.add(UpSampling1D(size=2))
        #model.add(Dropout(0.50))
        #model.add(Conv1DTranspose(128, kernel_size=3, padding='same', activation='relu'))
        #model.add(UpSampling1D(size=2))
        #model.add(Dense(64, activation='sigmoid'))
        #model.add(Dense(32, activation='sigmoid'))
        #model.add(Dense(128, activation='relu'))
        model.add(Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))
        if show_model_summary:
            model.summary()
        return model
    
    
    def build_model_dense(self, input_shape=(0, 0), show_model_summary=True, model_name='sequential'):
        if input_shape == (0, 0):
            return
        print("building model")
        #os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


        # Create the model
        model = Sequential(name=model_name)
        model.add(Dense(64, activation='sigmoid', input_shape=input_shape))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))
        if show_model_summary:
            model.summary()
        return model
    
    

    def train_model(self, model, input_size, input_data, output_data, epochs, batch_size, validation_split, save_model=True):
        input_data = input_data.reshape(len(input_data), input_size, 1)
        output_data = output_data.reshape(len(output_data), input_size, 1)
        lr_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=100000,
                                                               decay_rate=0.9)
        opt = SGD(learning_rate=lr_schedule)
        opt2 = tf.optimizers.Adam(learning_rate=0.001)
        
        model.compile(optimizer='adam', loss='binary_crossentropy')
        #model.compile(optimizer=opt2, loss='mean_squared_error', metrics=['mse'])
        #model.compile(optimizer='sgd', loss=tf.losses.Huber())
        #model.compile(optimizer='adam', loss=tf.losses.Huber())

        
        hist = model.fit(input_data, output_data,
                         epochs=epochs,
                         shuffle=True,
                         batch_size=batch_size,
                         validation_split=validation_split)
        if save_model:
            now = datetime.now()
            model_version = str(now.day) + "_" + str(now.month) + "_" + str(now.hour)
            model_name= str(model.name) + "_" +str(input_size) + "_v" + str(model_version)
            model.save('./Models/model_' + model_name)
        return hist, model_name

    def test_model(self,
                   model_name='',
                   input_size=0,
                   input_data=[]
                   ):
        l_model = tf.models.load_model('./Models/' + model_name)
        plot_model(l_model, to_file='./Images/model_plot.png', show_shapes=True, show_layer_names=True)
        input_data_r = input_data.reshape(1, input_size, 1)
        reconstructed_image = l_model.predict(input_data_r)
        
        return reconstructed_image[0]
    
    
    def get_models_list(self, filters=''):
        all_models = []
        counter = 1
        for file in glob.glob("./Models/*"):
            if filters not in file: 
                continue
            all_models.append(file.replace('./Models/',''))
            print(str(counter)+') '+file.replace('./Models/',''))
            counter = counter + 1 
        return all_models
        
    def load_model_from_list(self, name_filters='', data_to_test=[]):
        a_m = self.get_models_list(name_filters)
        print('here')
        invalid_model_id = True
        while invalid_model_id:
            model_id = 10
            print('here2')
            if 0 < model_id <=  len(a_m):
                invalid_model_id = False
            else:
                invalid_model_id = True
        
        return self.test_model(model_name=a_m[model_id-1], input_size=64, input_data=data_to_test)
        
    
    def load_model_by_name(self, model_name='', data_to_test=[]):
        return self.test_model(model_name=model_name, input_size=64, input_data=data_to_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
