"""
Created on 10.05.22
Author: Amir Tosson
Email: amir.tosson@uni-siegen.de  
Project: Denoising
File : main.py
"""
import numpy as np
from data_generator import DataGenerator
from autoencoder_model import Autoencoder
import matplotlib.pyplot as plt

import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def generate_data():
    print('Starting generating dataset:')
    dg = DataGenerator(64)
    return dg.generate_data_with_random_parameters(20, 100000, beta_max=1.2, beta_min=0.9, gamma_max=0.7, gamma_min=0.3,
                                                   constant_max=2.7, constant_min=0.0, save_data=True,
                                                   visualize_data=False)


if __name__ == '__main__':
    pure_sets, noisy_sets = generate_data()
    ae = Autoencoder()
    model = ae.build_model(input_shape=(64,1))
    model_history, model_name = ae.train_model(model,64,noisy_sets, pure_sets,20,64,0.8)
    dg = DataGenerator(64)
    pure_d, noise_d = dg.generate_test_dataset(1, 0.5, 0.4, 5)
    reconstructed_img = ae.test_model(model_name,64,noise_d)
    fig, axes = plt.subplots(1, 3)
  # Plot sample and reconstruciton
    axes[0].plot(noise_d)
    axes[0].set_title('Noisy waveform')
    axes[1].plot(pure_d)
    axes[1].set_title('Pure waveform')
    axes[2].plot(reconstructed_img)
    axes[2].set_title('Denoised waveform')
    plt.savefig("Results_test_dataset_nf_"+str(model_name)+'.png', format='png', dpi=600)
    plt.show()