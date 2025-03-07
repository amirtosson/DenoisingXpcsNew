#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:25:30 2025

@author: tosson
"""
import numpy as np
from datetime import datetime
import random
def exponential_decay(t1, t2, tau):
    return np.exp(-np.abs(t1 - t2) / tau)

def aging_exponential(t1, t2, tau0, alpha):
    tau = tau0 * (1 + alpha * np.minimum(t1, t2))  # Aging time constant
    return np.exp(-np.abs(t1 - t2) / tau)


def calculate_transmitted_flux(I0, mu, d):
    #return delta_t * I0 * np.exp(-mu * d) * (0.03/ssd**2) * np.exp(-0.124 * sample_thickness) 
    return I0 * np.exp(-mu * d)

class TTCDataGenerator:
    def __init__(self):
        super().__init__
        
        
    def generate_random_set(self,number_of_images=10000,
                            dynamics_type='normal',  
                            number_of_time_points=100, 
                            beta = 0.5,
                            alpha = 5,
                            tau = 1.0, 
                            save_data=True):
        I_beam = 1e12
        mu_abs = 0.139
        absorber_thickness_cm = range(10,11,2)
        C = np.zeros((number_of_time_points, number_of_time_points))
        output_images_size = number_of_images*len(absorber_thickness_cm)
        
        pure_images = np.zeros((output_images_size,number_of_time_points,number_of_time_points ))
        noisy_images = np.zeros_like(pure_images)
        for img in range(0,output_images_size, len(absorber_thickness_cm)):
            t = np.linspace(0, random.randint(1, 10), number_of_time_points) 
            alpha = random.uniform(0.2, 0.3)
            alpha_2 = 0#random.uniform(0.5, 1.0)
            tau = random.uniform(60.5, 60.9)
            tau_2 = random.uniform(19.0, 20.9)

            beta = random.uniform(0.2, 0.3)
            beta_2 = random.uniform(0.5, 0.6)

            average_photons_per_frame = calculate_transmitted_flux(I_beam, mu_abs, 2)
            noise_mask = np.random.poisson(average_photons_per_frame, size= (number_of_time_points,number_of_time_points))
            noise_mask = (noise_mask - np.min(noise_mask)) / (np.max(noise_mask) - np.min(noise_mask))
            
            #MIRROR noise_mask! TODO: check
            noise_mask = np.tril(noise_mask) + np.tril(noise_mask, -1).T

            for i in range(number_of_time_points):
                for j in range(number_of_time_points):
                    if dynamics_type=='normal':
                        C[i, j] = 1 + beta * exponential_decay(t[i], t[j], tau)
                    elif dynamics_type=='aging':
                        C[i, j] = 1 + beta * aging_exponential(t[i], t[j], tau, alpha)
                    elif dynamics_type=='multi-systems':
                            s1= (0.5 *(  exponential_decay(t[i], t[j], 15.3)))
                            s2= (0.3*(exponential_decay(t[i], t[j], 60.5)))
                            
                            C[i, j] =  1+ np.maximum(s1,s2)

            for index, abs_thichness in enumerate(absorber_thickness_cm):
                C_noisy = C + abs_thichness*0.2*noise_mask
                pure_images[img+index] = C
                noisy_images[img+index] = C_noisy
        if save_data:
            now = datetime.now()
            data_set_version = str(now.day) + str(now.month) + str(now.hour) + str(now.minute)
            np.save('./Datasets/TTC/'+data_set_version+dynamics_type+'_TTC_noisy_mirrored_' +str(output_images_size), noisy_images)
            np.save('./Datasets/TTC/'+data_set_version+dynamics_type+'_TTC_pure_mirrored_' +str(output_images_size), pure_images)     
        return pure_images, noisy_images

    
    def generate_noise_mask(self, number_points= 100):
        noise_mask = np.random.poisson(4, size= (number_points,number_points))
        noise_mask = (noise_mask - np.min(noise_mask)) / (np.max(noise_mask) - np.min(noise_mask))
        
        #MIRROR noise_mask! TODO: check
        noise_mask = np.tril(noise_mask) + np.tril(noise_mask, -1).T
        return noise_mask

    def new_data_set_generator(self,number_of_images=10000,
                               number_of_noisy_img_per_pure = 4,
                               number_of_time_points=100,
                               save_data=True):
        
        
        
        
        output_size = number_of_images*number_of_noisy_img_per_pure
        
        pure_images = np.zeros((output_size,number_of_time_points,number_of_time_points ))
        noisy_images = np.zeros_like(pure_images)
        for img in range(0,output_size, number_of_noisy_img_per_pure):
            # the pure ttc 
            speckle_contrast = random.uniform(0.7, 0.98) # beta
            relaxation_time_initial = random.uniform(0.01, 3)  # gamma (fast)
            relaxation_time_final = random.uniform(3.1, 10) # gamma (slower)
            stretching_exponent_initial = random.uniform(0.9, 1.2)  # alpha
            stretching_exponent_final = random.uniform(0.001, 0.8)  # 
            t_max = random.uniform(6, 10) 
            t_points = number_of_time_points 
            # Time grid
            t = np.linspace(0, t_max, t_points)
            T1, T2 = np.meshgrid(t, t)  # Create a 2D time grid     
            relaxation_time_t = relaxation_time_initial + (relaxation_time_final - relaxation_time_initial) * (T1 + T2) / (2 * t_max)
            stretching_exponent_t = stretching_exponent_initial - (stretching_exponent_initial - stretching_exponent_final) * (T1 + T2) / (2 * t_max)
            stretching_exponent_t = np.clip(stretching_exponent_t, stretching_exponent_final, stretching_exponent_initial)
            ttc_pure = 1+  speckle_contrast * np.exp(-np.abs(T2 - T1) / relaxation_time_t) ** stretching_exponent_t 
            noise = self.generate_noise_mask()
            for idx in range(number_of_noisy_img_per_pure):
                ttc_noise = ttc_pure + idx*0.1*noise
                pure_images[img+idx] = ttc_pure
                noisy_images[img+idx] = ttc_noise
        if save_data:
            now = datetime.now()
            data_set_version = str(now.day) + str(now.month) + str(now.hour) + str(now.minute)
            np.save('./Datasets/TTC/High_Random_'+data_set_version+'_TTC_noisy_mirrored_' +str(output_size), noisy_images)
            np.save('./Datasets/TTC/High_Random_'+data_set_version+'_TTC_pure_mirrored_' +str(output_size), pure_images)
        return  pure_images, noisy_images
        