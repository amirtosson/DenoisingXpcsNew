"""
Created on 10.05.22
Author: Amir Tosson
Email: amir.tosson@uni-siegen.de  
Project: DenoisingXPCS
File : data_generator.py
Class: data_generator
"""

import numpy as np
import tqdm as tq
from datetime import datetime
import matplotlib.pyplot as plt


class DataGenerator:
    TIME_POINTS = 64
    TIME_DATA = []

    def __init__(self, time_points):
        self.TIME_POINTS = time_points
        self.TIME_DATA = np.logspace(-2, 2, num=time_points)
        super().__init__

    def generate_data_with_random_parameters(self,
                                             num_random_sets=1,
                                             num_samples=1,
                                             beta_max=1,
                                             beta_min=0,
                                             gamma_max=1,
                                             gamma_min=0,
                                             constant_max=1,
                                             constant_min=0,
                                             noise_factor_max=2,
                                             noise_factor_min=6,
                                             save_data=False,
                                             file_name="data_set",
                                             visualize_data=True,
                                             num_samples_visualize=4
                                             ):
        beta_random = np.random.uniform(low=beta_min, high=beta_max, size=(num_random_sets,))
        gamma_random = np.random.uniform(low=gamma_min, high=gamma_max, size=(num_random_sets,))
        c_random = np.random.uniform(low=constant_min, high=constant_max, size=(num_random_sets,))
        noise_random = np.random.uniform(low=noise_factor_min, high=noise_factor_max, size=(num_random_sets,))
        all_noise = []
        all_pure = []
        for i in range(num_random_sets):
            print("\nDataset Nr: ", str(i + 1))
            with tq.tqdm(total=num_samples) as pbar:
                data_pure, data_noise = self.__generate_datasets(num_samples,
                                                                 beta_random[i],
                                                                 gamma_random[i],
                                                                 c_random[i],
                                                                 noise_random[i],
                                                                 pbar)
                for l in data_pure:
                    all_pure.append(l)
                for k in data_noise:
                    all_noise.append(k)
        all_noise, all_pure = np.array(all_noise), np.array(all_pure)

        if visualize_data:
            for i in range(0, num_samples_visualize):
                random_index = np.random.randint(0, len(all_noise) - 1)
                fig, axes = plt.subplots(1, 2)
                # Plot sample and reconstruciton
                axes[0].plot(all_noise[random_index])
                axes[0].set_title('Noisy waveform')
                axes[1].plot(all_pure[random_index])
                axes[1].set_title('Pure waveform')
                plt.show()
        if save_data:
            now = datetime.now()
            data_set_version = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "_" + str(now.minute)
            np.save('./Datasets/' + file_name + '_noise_' + str(self.TIME_POINTS) + '_' + str(num_random_sets) + '_'
                    + str(num_samples) + '_' + data_set_version, all_noise)
            np.save('./Datasets/' + file_name + '_pure_' + str(self.TIME_POINTS) + '_' + str(num_random_sets) + '_'
                    + str(num_samples) + '_' + data_set_version, all_pure)
        return all_noise, all_pure

    def __generate_datasets(self,
                            num_sample,
                            beta,
                            gamma,
                            c,
                            noise_factor,
                            pbar):
        pure_data = []
        noise_data = []
        for _ in range(num_sample):
            pbar.update(1)
            g2_pure = c + beta * np.exp(-(gamma * self.TIME_DATA))
            pure_data.append(g2_pure)
            number_images = np.linspace(150, 10, self.TIME_POINTS)
            noise = np.random.normal(0, 1, g2_pure.shape)
            g2_noisy = g2_pure + noise_factor * noise / (number_images + 0)
            noise_data.append(g2_noisy)
        return np.array(pure_data), np.array(noise_data)

    def generate_test_dataset(self, c, beta, gamma, noise_factor):
        g2_pure = c + beta * np.exp(-(gamma * self.TIME_DATA))
        number_images = np.linspace(150,10,self.TIME_POINTS)
        noise = np.random.normal(0, 1, g2_pure.shape)
        g2_noisy = g2_pure + noise_factor * noise/(number_images + 0)
        fig, axes = plt.subplots(1, 2)
        # Plot sample and reconstruciton
        axes[0].plot(g2_noisy)
        axes[0].set_title('Noisy waveform')
        axes[1].plot(g2_pure)
        axes[1].set_title('Pure waveform')
        plt.show()
        return np.array(g2_pure), np.array(g2_noisy)

    def plotting_simulation_data(self, noisy_data=[], original_data=[], reconstructed_data=[]):
        fig, axes = plt.subplots(1, 3)
        # Plot sample and reconstruciton
        axes[0].plot(noisy_data)
        axes[0].set_title('Noisy waveform')
        axes[1].plot(original_data)
        axes[1].set_title('Pure waveform')
        axes[2].plot(reconstructed_data)
        axes[3].set_title('Reconstructed waveform')
        plt.show()
