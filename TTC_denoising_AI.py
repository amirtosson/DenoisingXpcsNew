#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:42:40 2025

@author: tosson
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class TTCCNN:
    def __init__(self):
        super().__init__
        
        
        
    def build_model(self, input_shape=(0, 0), show_model_summary=True, model_name='sequential'):
        if input_shape == (0, 0):
            return
        print("building CNN")

        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.ELU(alpha=0.5)(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        outputs = layers.Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)
        cnn = models.Model(inputs, outputs)
        if show_model_summary:
            cnn.summary()
        return cnn
    
    
    def fit_model(self, 
                  model, 
                  input_data, 
                  output_data, 
                  epochs, 
                  batch_size, 
                  validation_split, 
                  save_model=True):
        def psnr(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)
        def ncc_loss(y_true, y_pred, reduce_batch=True):
            """
            Compute the Normalized Cross-Correlation (NCC) loss.
        
            Parameters:
                y_true (tf.Tensor): Ground truth data.
                y_pred (tf.Tensor): Predicted data.
                reduce_batch (bool): Whether to reduce the loss across the batch dimension.
        
            Returns:
                tf.Tensor: NCC loss value.
            """
            # Flatten the spatial dimensions (height x width)
            y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])  # Shape: (batch_size, height * width * channels)
            y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])  # Shape: (batch_size, height * width * channels)
        
            # Compute means
            mean_true = tf.reduce_mean(y_true, axis=1, keepdims=True)  # Shape: (batch_size, 1)
            mean_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)  # Shape: (batch_size, 1)
        
            # Subtract means
            y_true_centered = y_true - mean_true
            y_pred_centered = y_pred - mean_pred
        
            # Compute numerator: sum of element-wise product
            numerator = tf.reduce_sum(y_true_centered * y_pred_centered, axis=1)  # Shape: (batch_size,)
        
            # Compute denominator: product of norms
            norm_true = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered), axis=1))  # Shape: (batch_size,)
            norm_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered), axis=1))  # Shape: (batch_size,)
            denominator = norm_true * norm_pred  # Shape: (batch_size,)
        
            # Avoid division by zero
            epsilon = 1e-7
            denominator = tf.maximum(denominator, epsilon)
        
            # Compute NCC
            ncc = numerator / denominator  # Shape: (batch_size,)
        
            # Loss: Minimize (1 - NCC)
            loss = 1 - ncc  # Shape: (batch_size,)
        
            # Optionally reduce across the batch
            if reduce_batch:
                loss = tf.reduce_mean(loss)  # Scalar loss value
        
            return loss
        
        def ssim_loss(y_true, y_pred):
            ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
            return 1 - tf.reduce_mean(ssim_value)
        def pnsr_loss(y_true, y_pred, max_val=1.0):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            psnr = 10 * tf.math.log((max_val ** 2) / mse) / tf.math.log(10.0)
            return -psnr
        def poisson_loss(y_true, y_pred):
            y_pred = tf.maximum(y_pred, 1e-7)  # Avoid negative or zero predictions
            return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-10))
        # def combined_loss(y_true, y_pred, alpha=0.5):
        #     y_pred = tf.maximum(y_pred, 1e-7)
        #     #mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        #     poisson_loss_ = tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-10))
        #     ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        #     return alpha * poisson_loss_ + (1 - alpha) * ssim_loss
        def combined_loss(y_true, y_pred, alpha=0.5):
            dx = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            dy = y_pred[:, :, 1:] - y_pred[:, :, :-1]
            
            tv_loss = tf.reduce_sum(tf.abs(dx)) + tf.reduce_sum(tf.abs(dy))
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return 0.5 * mse_loss + 0.5 * ssim_loss + 0.2 * tv_loss      
        
        #input_data = input_data / np.max(input_data)
        #output_data = output_data / np.max(output_data)
        input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
        output_data = (output_data - np.min(output_data)) / (np.max(output_data) - np.min(output_data))
        
        #model.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.losses.Huber(),  metrics=['mae', 'mse'])
        model.compile(optimizer=Adam(learning_rate=1e-4),  loss=ncc_loss,   metrics=['mae', 'mse'])

        #model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        hist = model.fit(input_data, output_data,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=validation_split,
                 callbacks=[early_stopping])
        
        now = datetime.now()
        model_version = str(now.day) + str(now.month) + str(now.hour)+ str(now.minute)
        model_name= 'CNN_TTC_ks5325_sigmoid_ncc_loss_'+str(model_version)+ "_training_set_" +str(len(input_data)) 
        if save_model:
            model.save('./Models/TTC_models/model_' + model_name)
            
        return hist, model_name
    
    
    def test_model(self,
                   model_name='',
                   input_size=0,
                   input_data=[]
                   ):
        def psnr(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)
        def ncc_loss(y_true, y_pred, reduce_batch=True):
            """
            Compute the Normalized Cross-Correlation (NCC) loss.
        
            Parameters:
                y_true (tf.Tensor): Ground truth data.
                y_pred (tf.Tensor): Predicted data.
                reduce_batch (bool): Whether to reduce the loss across the batch dimension.
        
            Returns:
                tf.Tensor: NCC loss value.
            """
            # Flatten the spatial dimensions (height x width)
            y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])  # Shape: (batch_size, height * width * channels)
            y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])  # Shape: (batch_size, height * width * channels)
        
            # Compute means
            mean_true = tf.reduce_mean(y_true, axis=1, keepdims=True)  # Shape: (batch_size, 1)
            mean_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)  # Shape: (batch_size, 1)
        
            # Subtract means
            y_true_centered = y_true - mean_true
            y_pred_centered = y_pred - mean_pred
        
            # Compute numerator: sum of element-wise product
            numerator = tf.reduce_sum(y_true_centered * y_pred_centered, axis=1)  # Shape: (batch_size,)
        
            # Compute denominator: product of norms
            norm_true = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered), axis=1))  # Shape: (batch_size,)
            norm_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered), axis=1))  # Shape: (batch_size,)
            denominator = norm_true * norm_pred  # Shape: (batch_size,)
        
            # Avoid division by zero
            epsilon = 1e-7
            denominator = tf.maximum(denominator, epsilon)
        
            # Compute NCC
            ncc = numerator / denominator  # Shape: (batch_size,)
        
            # Loss: Minimize (1 - NCC)
            loss = 1 - ncc  # Shape: (batch_size,)
        
            # Optionally reduce across the batch
            if reduce_batch:
                loss = tf.reduce_mean(loss)  # Scalar loss value
        
            return loss
        
        
        def ssim_loss(y_true, y_pred):
            ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
            return 1 - tf.reduce_mean(ssim_value)
        def pnsr_loss(y_true, y_pred, max_val=1.0):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            psnr = 10 * tf.math.log((max_val ** 2) / mse) / tf.math.log(10.0)
            return -psnr
        
        def poisson_loss(y_true, y_pred):
            y_pred = tf.maximum(y_pred, 1e-7)  # Avoid negative or zero predictions
            return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-10))
        
        def combined_loss(y_true, y_pred, alpha=0.5):
            dx = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            dy = y_pred[:, :, 1:] - y_pred[:, :, :-1]
            
            tv_loss = tf.reduce_sum(tf.abs(dx)) + tf.reduce_sum(tf.abs(dy))
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return 0.5 * mse_loss + 0.5 * ssim_loss + 0.2 * tv_loss
        
        
        l_model = models.load_model('./Models/TTC_models/' + model_name, custom_objects={'ncc_loss': ncc_loss})
        #l_model = models.load_model('./Models/TTC_models/' + model_name)
        #l_model.summary()
        #input_data = input_data / np.max(input_data)
        #input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

        input_data_r = input_data.reshape(1, input_size, input_size, 1)
        reconstructed_image = l_model.predict(input_data_r)
        
        return reconstructed_image[0]
    






class TTCAutoencoder:
    def __init__(self):
        super().__init__
        
        
        
    def build_model(self, input_shape=(0, 0), show_model_summary=True, model_name='sequential'):
        if input_shape == (0, 0):
            return
        print("building autoencoder")
        inputs = layers.Input(shape=input_shape)

        # Encoder (down-sampling)
        # First convolution block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # Second convolution block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # Third convolution block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2),strides=1, padding='same')(x)

        # Fourth convolution block
        #x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        #x = layers.MaxPooling2D((2, 2),strides=1, padding='same')(x)

        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

        # Decoder (upsampling)
        # First deconvolution block
        #x = layers.Conv2DTranspose(512, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
        
        # Second deconvolution block
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
        
        # Third deconvolution block
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
        
        # Fourth deconvolution block
        x = layers.Conv2DTranspose(64, (3, 3),  activation='relu', padding='same')(x)
        
        # Output layer
        outputs = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
        autoencoder = models.Model(inputs, outputs)
        if show_model_summary:
            autoencoder.summary()
        return autoencoder
    
    
    def fit_model(self, 
                  model, 
                  input_data, 
                  output_data, 
                  epochs, 
                  batch_size, 
                  validation_split, 
                  save_model=True):
        def psnr(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)
        
        def ssim_loss(y_true, y_pred):
            ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
            return 1 - tf.reduce_mean(ssim_value)
        def pnsr_loss(y_true, y_pred, max_val=1.0):
            #mse = tf.reduce_mean(tf.square(y_true - y_pred))
            #y_true = (y_true-tf.reduce_min(y_true)) / (tf.reduce_max(y_true)-tf.reduce_min(y_true))
            #y_pred = (y_pred-tf.reduce_min(y_pred)) / (tf.reduce_max(y_pred)-tf.reduce_min(y_pred))

            errors = y_true - y_pred
            variance = tf.reduce_mean(tf.square(errors - tf.reduce_mean(errors)))
            #psnr = 10 * tf.math.log((max_val ** 2) / variance) / tf.math.log(10.0)
            return variance
        def poisson_loss(y_true, y_pred):
            y_pred = tf.maximum(y_pred, 1e-7)  # Avoid negative or zero predictions
            return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-10))
        # def combined_loss(y_true, y_pred, alpha=0.5):
        #     y_pred = tf.maximum(y_pred, 1e-7)
        #     #mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        #     poisson_loss_ = tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-10))
        #     ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        #     return alpha * poisson_loss_ + (1 - alpha) * ssim_loss
        def combined_loss(y_true, y_pred):
            dx = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            dy = y_pred[:, :, 1:] - y_pred[:, :, :-1]
            
            tv_loss = tf.reduce_sum(tf.abs(dx)) + tf.reduce_sum(tf.abs(dy))
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return 0.5 * mse_loss + 0.5 * ssim_loss + 0.2 * tv_loss      
        

        input_data = input_data / np.max(input_data)
        output_data = output_data / np.max(output_data)
        #input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
        #output_data = (output_data - np.min(output_data)) / (np.max(output_data) - np.min(output_data))
        
        #model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse',  metrics=['accuracy'])
        model.compile(optimizer=Adam(learning_rate=1e-4),  loss=combined_loss, metrics=['mae', 'mse'])

        #model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        hist = model.fit(input_data, output_data,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=validation_split,
                 callbacks=[early_stopping])
        
        now = datetime.now()
        model_version = str(now.day) + str(now.month) + str(now.hour)+ str(now.minute)
        model_name= 'Autoencoder_TTC_ks3_norm255_combi_loss_'+str(model_version)+ "_training_set_" +str(len(input_data)) 
        if save_model:
            model.save('./Models/TTC_models/model_' + model_name)
            
        return hist, model_name
    
    
    def test_model(self,
                   model_name='',
                   input_size=0,
                   input_data=[]
                   ):
        def psnr(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)
        
        def ssim_loss(y_true, y_pred):
            ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
            return 1 - tf.reduce_mean(ssim_value)
        def pnsr_loss(y_true, y_pred, max_val=1.0):
            #mse = tf.reduce_mean(tf.square(y_true - y_pred))
            #y_true = (y_true-tf.reduce_min(y_true)) / (tf.reduce_max(y_true)-tf.reduce_min(y_true))
            #y_pred = (y_pred-tf.reduce_min(y_pred)) / (tf.reduce_max(y_pred)-tf.reduce_min(y_pred))
            errors = y_true - y_pred
            variance = tf.reduce_mean(tf.square(errors - tf.reduce_mean(errors)))
            #psnr = 10 * tf.math.log((max_val ** 2) / variance) / tf.math.log(10.0)
            return variance
        
        def poisson_loss(y_true, y_pred):
            y_pred = tf.maximum(y_pred, 1e-7)  # Avoid negative or zero predictions
            return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-10))
        
        def combined_loss(y_true, y_pred, alpha=0.5):
            dx = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            dy = y_pred[:, :, 1:] - y_pred[:, :, :-1]
            
            tv_loss = tf.reduce_sum(tf.abs(dx)) + tf.reduce_sum(tf.abs(dy))
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return 0.5 * mse_loss + 0.5 * ssim_loss + 0.2 * tv_loss
        
        
        l_model = models.load_model('./Models/TTC_models/' + model_name, custom_objects={'combined_loss': combined_loss})
        #l_model = models.load_model('./Models/TTC_models/' + model_name)
        l_model.summary()

        input_data = input_data / np.max(input_data)
        #input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

        input_data_r = input_data.reshape(1, input_size, input_size, 1)
        reconstructed_image = l_model.predict(input_data_r)
        
        return reconstructed_image[0]
    



class TTCGANs:
    def __init__(self, input_shape, generator_filters=[64, 32], discriminator_filters=[32, 64]):
        """
        Initialize the GAN model.

        Parameters:
            input_shape (tuple): Shape of the input TTC matrix (height, width, channels).
            generator_filters (list): List of filter sizes for the generator.
            discriminator_filters (list): List of filter sizes for the discriminator.
        """
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.input_shape = input_shape
        self.generator = self.build_generator(generator_filters)
        self.discriminator = self.build_discriminator(discriminator_filters)
        self.gan = self.build_gan()
        
     

    def build_generator(self, filters):
        """
        Build the generator network.

        Parameters:
            filters (list): List of filter sizes for the generator.

        Returns:
            tf.keras.Model: Generator model.
        """
        
        print('building generator')
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        x = inputs
        skips = []
        for f in filters:
            x = layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
            #x = layers.MaxPooling2D((2, 2))(x)
            skips.append(x)

        # Bottleneck
        x = layers.Conv2D(filters[-1], (3, 3), activation='relu', padding='same')(x)

        # Decoder with skip connections
        for i, f in enumerate(reversed(filters)):
            x = layers.Conv2DTranspose(f, (3, 3),  activation='relu', padding='same')(x)
            
            x = layers.Concatenate()([x, skips[-(i + 1)]])
        
        # Output layer
        outputs = layers.Conv2D(1, (1, 1), activation='linear', padding='same')(x)

        return models.Model(inputs, outputs, name="generator")

    def build_discriminator(self, filters):
        """
        Build the discriminator network.

        Parameters:
            filters (list): List of filter sizes for the discriminator.

        Returns:
            tf.keras.Model: Discriminator model.
        """
        print('building discriminator')
        inputs = layers.Input(shape=self.input_shape)

        x = inputs
        for f in filters:
            x = layers.Conv2D(f, (3, 3),  activation='relu', padding='same')(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        return models.Model(inputs, outputs, name="discriminator")

    def build_gan(self):
        """
        Combine the generator and discriminator into a GAN.

        Returns:
            tf.keras.Model: GAN model.
        """
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=self.input_shape)
        generated_image = self.generator(gan_input)
        gan_output = self.discriminator(generated_image)

        return models.Model(gan_input, gan_output, name="gan")

    def model_compile(self):
        """
        Compile the GAN model.

        Parameters:
            generator_optimizer: Optimizer for the generator.
            discriminator_optimizer: Optimizer for the discriminator.
            loss_fn: Loss function (e.g., binary cross-entropy).
        """
        

        loss_fn=tf.keras.losses.BinaryCrossentropy()
        
        self.generator.compile(optimizer=self.generator_optimizer, loss=loss_fn)
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss=loss_fn, metrics=['accuracy'])
        self.gan.compile(optimizer=self.generator_optimizer, loss=loss_fn)
    def train_step(self, noisy_data, clean_data=None):
        """
        Perform a single training step.

        Parameters:
            noisy_data (numpy.ndarray): Noisy TTC matrices.
            clean_data (numpy.ndarray): Clean TTC matrices (optional for supervised training).
        """
        real_labels = tf.ones((noisy_data.shape[0], 1))
        fake_labels = tf.zeros((noisy_data.shape[0], 1))

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake data
            generated_data = self.generator(noisy_data, training=True)

            # Discriminator predictions
            d_real_output = self.discriminator(clean_data if clean_data is not None else noisy_data, training=True)
            d_fake_output = self.discriminator(generated_data, training=True)

            # Losses
            d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, d_real_output)
            d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, d_fake_output)
            d_loss = 0.5 * (tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake))

            g_loss = tf.keras.losses.binary_crossentropy(real_labels, d_fake_output)

        # Compute gradients and apply them
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        del tape  # Cleanup GradientTape

        return d_loss, g_loss

    def model_train(self, noisy_data, clean_data=None, epochs=50, batch_size=32):
        """
        Train the GAN.

        Parameters:
            noisy_data (numpy.ndarray): Noisy TTC matrices.
            clean_data (numpy.ndarray): Clean TTC matrices (optional for supervised training).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        noisy_data = noisy_data.astype('float32')
        clean_data = clean_data.astype('float32') if clean_data is not None else None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            indices = np.random.permutation(len(noisy_data))
            noisy_data = noisy_data[indices]
            clean_data_epoch = clean_data[indices] if clean_data is not None else None

            for i in range(0, len(noisy_data), batch_size):
                noisy_batch = noisy_data[i:i + batch_size]
                clean_batch = clean_data_epoch[i:i + batch_size] if clean_data_epoch is not None else None

                d_loss, g_loss = self.train_step(noisy_batch, clean_batch)
                
                #print(f"  [D loss: {d_loss}] [G loss: {g_loss}]")
        """
        Train the GAN.

        Parameters:
            noisy_data (numpy.ndarray): Noisy TTC matrices.
            clean_data (numpy.ndarray): Clean TTC matrices (optional for supervised training).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        noisy_data = noisy_data.astype('float32')
        clean_data = clean_data.astype('float32') if clean_data is not None else None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Shuffle data
            indices = np.random.permutation(len(noisy_data))
            noisy_data = noisy_data[indices]
            clean_data_epoch = clean_data[indices] if clean_data is not None else None

            for i in range(0, len(noisy_data), batch_size):
                # Get real and noisy batches
                noisy_batch = noisy_data[i:i + batch_size]
                clean_batch = clean_data_epoch[i:i + batch_size] if clean_data is not None else None

                # Generate fake data
                generated_batch = self.generator.predict(noisy_batch)

                # Train discriminator
                real_labels = np.ones((len(noisy_batch), 1))
                fake_labels = np.zeros((len(noisy_batch), 1))

                d_loss_real = self.discriminator.train_on_batch(clean_batch if clean_batch is not None else noisy_batch, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(generated_batch, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train generator
                g_loss = self.gan.train_on_batch(noisy_batch, real_labels)

                # Print progress
                print(f"  [D loss: {d_loss[0]:.3f}, acc: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.3f}]")
            now = datetime.now()
            model_version = str(now.day) + str(now.month) + str(now.hour)+ str(now.minute)
            model_name= 'GANS_TTC_'+str(model_version)+ "_training_set_" +str(len(noisy_data)) 
            if save_model:
                self.gan.save('./Models/TTC_models/model_' + model_name)
            return model_name
                
            

    def generate(self, noisy_data):
        """
        Generate denoised TTC matrices.

        Parameters:
            noisy_data (numpy.ndarray): Noisy TTC matrices.

        Returns:
            numpy.ndarray: Denoised TTC matrices.
        """
        return self.generator.predict(noisy_data)