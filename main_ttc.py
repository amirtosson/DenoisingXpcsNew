#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:07:24 2025

@author: tosson
"""
from TTC_data_generator import TTCDataGenerator
from TTC_denoising_AI import TTCAutoencoder, TTCGANs, TTCCNN
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt

data_test = np.load('./TESTDATA/ttc_q1_series3.npy')

# Get the original shape of the data
original_shape = data_test.shape
if original_shape[0] > 100 or original_shape[1] > 100:
    print("Resizing data to 100x100")
    data_test_resized = data_test[0:100,0:100]
else:
    data_test_resized = data_test
# Create a new array with the desired shape, filled with zeros (or any other padding value)
#padded_data_test = np.zeros((100, 100))

# Copy the original data into the new array
#padded_data_test[:original_shape[0], :original_shape[1]] = data_test
#data_test = padded_data_test


plt.imshow(data_test,origin='lower')
plt.show()

plt.imshow(data_test_resized,origin='lower')
plt.show()
model_name = "CNN_TTC_ks5325_linear_huber_loss_1321115_training_set_125000"

ttc_cnn = TTCCNN()
C_denoised = ttc_cnn.test_model("model_"+model_name, 100, data_test_resized)

fig, ax = plt.subplots()
ax.imshow(C_denoised[0:50,0:50], origin='lower')
plt.show()

# Save the denoised image
fig.savefig('denoised_image_mora.png', dpi=300, bbox_inches='tight', transparent=True,format='png')


# # Parameters
# beta = 0.7  # Speckle contrast
# tau_initial = 0.1  # Initial relaxation time (fast)
# tau_final = 5.0  # Final relaxation time (slower as system ages)
# alpha_initial = 1  # Initial stretching exponent (pure exponential decay)
# alpha_final = 0.9 # Final stretching exponent (more stretched, glassy dynamics)
# t_max = 10  # Maximum time
# t_points = 100  # Time resolution
# # Parameters for Two Different Systems in the Sample
# beta1, beta2 = 0.7, 0.7  # Speckle contrast for system 1 and system 2
# tau1, alpha1 = .1, 0.01  # System 1: Fast relaxation, exponential decay
# tau2, alpha2 = 5.0, 0.5  # System 2: Slow relaxation, stretched exponential

# # Time grid
# t = np.linspace(0, t_max, t_points)
# T1, T2 = np.meshgrid(t, t)  # Create a 2D time grid

# # Define weighted contributions of the two systems
# weight1, weight2 = 1, 1  # Proportion of system 1 and system 2 in the sample

# # Compute TTC (Two-Time Correlation Function) for each system separately
# C_ttc_sys1 = 1 + beta * np.exp(-np.abs(T2 - T1) / tau1) ** alpha1
# C_ttc_sys2 = 1 + beta2 * np.exp(-np.abs(T2 - T1) / tau2) ** alpha2

# # Weighted combination of both systems
# C_ttc_mixed = weight1 * C_ttc_sys1 + weight2 * C_ttc_sys2
# # Time grid

# # Time-dependent relaxation time: increases as the system ages
# tau_t = tau_initial + (tau_final - tau_initial) * (T1 + T2) / (2 * t_max)

# # Time-dependent stretching exponent: decreases as the system slows down
# alpha_t = alpha_initial - (alpha_initial - alpha_final) * (T1 + T2) / (2 * t_max)
# alpha_t = np.clip(alpha_t, alpha_final, alpha_initial)  # Ensure within bounds

# # Compute TTC with time-dependent relaxation time
# C_ttc_time_dependent = 1+  beta * np.exp(-np.abs(T2 - T1) / tau_t) ** alpha_t #+C_ttc_mixed

# # Plot the TTC heatmap for time-dependent relaxation
# plt.figure(figsize=(7,6))
# plt.imshow(C_ttc_time_dependent, cmap='inferno', extent=[0, t_max, 0, t_max], origin='lower')
# plt.colorbar(label="C(q, t1, t2)")
# plt.xlabel("Time \( t_1 \)")
# plt.ylabel("Time \( t_2 \)")
# plt.title("TTC with Time-Dependent Relaxation Time and Stretching Exponent")
# plt.show()









   # beta = 0.3   # Speckle contrast
   # tau1, alpha1 = 0.4, 1.0  # Fast relaxation (short times)
   # tau2, alpha2 = 1.5, 0.5  # Slow relaxation (long times)
   # t_max = 10  # Maximum time
   # t_points = 100  # Time resolution
   # stretching_exponent_constant = 0.8
   # # Time grid
   # t = np.linspace(0, t_max, t_points)
   # T1, T2 = np.meshgrid(t, t)  # Create a 2D time grid
   
   # # Define piecewise alpha values
   # alpha = np.where(np.abs(T2 - T1) < tau1, alpha1, alpha2)
   # tau0 = np.where(np.abs(T2 - T1) < tau1 , tau1 , tau2)
   
   # TT_map = np.minimum(T2 , T1)
   # tau_map = tau0 * (1+alpha * TT_map)
   # #alpha_map = tau_map * alpha1
   
   # #tau = tau* (1+ stretching_exponent_constant)
   # #alpha_g = np.abs(T2 - T1)* stretching_exponent_constant
   # # Compute TTC (Two-Time Correlation Function)
   # C_ttc_1 =  beta * np.exp(-np.abs(T2 - T1)/ tau_map) ** alpha
   
   # C_ttc_2 =  .05 * np.exp(-np.abs(T2 - T1) / tau1) ** alpha1
   # C_ttc =  1+C_ttc_1

   # # Plot the TTC heatmap
   # plt.figure(figsize=(7,6))
   # plt.imshow(C_ttc, cmap='inferno', extent=[0, t_max, 0, t_max], origin='lower')
   # plt.colorbar(label="C(q, t1, t2)")
   # plt.xlabel("Time \( t_1 \)")
   # plt.ylabel("Time \( t_2 \)")
   # plt.title("Two-Time Correlation Function (TTC) with Two α Values")
   # plt.show()
   
   
   
   # t = np.linspace(0.1, 10, 100)
   
   # # Parameters for two different alpha values
   # tau1, alpha1 = 2.0, 1.0  # Fast relaxation (exponential decay)
   # tau2, alpha2 = 5.0, 0.5  # Slow relaxation (stretched decay)
   # beta = 0.7
   
   # # Piecewise g2(t): Fast decay at short times, slow decay at long times
   # g2_t = np.where(t < tau1, 
   #                 1 + beta * np.exp(-(t / tau1) ** alpha1),  # Short time behavior
   #                 1 + beta * np.exp(-(t / tau2) ** alpha2))  # Long time behavior
   
   # # Plot
   # plt.plot(t, g2_t, label=f"α1={alpha1}, α2={alpha2}")
   # plt.xlabel("Time t")
   # plt.ylabel("g2(t)")
   # plt.title("Two Different α in One Sample")
   # plt.legend()
   # plt.show()
   
   # TIME_DATA = np.linspace(0, 100, num=100)
   # g_2 = 0.9*np.exp(-pow(TIME_DATA/50.5,5))
   # g_2_2 = 0.5*np.exp(-TIME_DATA/60.5)
   # plt.plot(TIME_DATA, g_2)
   # #plt.plot(TIME_DATA, g_2_2)
   # #plt.plot(TIME_DATA,  np.maximum(g_2_2,g_2))

   # #plt.xscale('log')
   # plt.show()
   
   # np.maximum([1,2,3,4,5],[5,4,3,2,1])




if __name__ == '__main__':
    
    
    

 
    
    ttc_dg = TTCDataGenerator()
    
    
    ttc_output_data, ttc_input_data  = ttc_dg.new_data_set_generator(2,2)
    ttc_ae = TTCAutoencoder()
    ae_model = ttc_ae.build_model(input_shape=(100,100,1))
    history, model_name = ttc_ae.fit_model(ae_model, ttc_input_data, ttc_output_data, 5, 32, 0.7, save_model=True)
      
           
    idx = 159610
    test_data = ttc_input_data[idx]  
    
    #test_data = inp[1]  
    C_t_d = gaussian_filter(test_data, sigma=1.0)
    C_denoised = ttc_ae.test_model("model_"+model_name, 100, C_t_d)
    C_denoised_direct = ttc_ae.test_model("model_"+model_name, 100, test_data)

      

    ig, ax = plt.subplots(2, 2, figsize=(16, 16))
       
    ax[0][0].imshow( ttc_output_data[idx]  , cmap='plasma', origin='lower', 
             extent=[0, 10, 0, 10], aspect='auto')
    ax[0][0].set_title('Pure TTC')
    ax[0][0].set_xlabel('$t_1$')
    ax[0][0].set_ylabel('$t_2$')

    ax[0][1].imshow(test_data, cmap='plasma', origin='lower', 
             extent=[0, 10, 0, 10], aspect='auto')
    ax[0][1].set_title('Noisiy TTC')
    ax[0][1].set_xlabel('$t_1$')
    ax[0][1].set_ylabel('$t_2$')
       
    ax[1][0].imshow(C_t_d, cmap='plasma', origin='lower', aspect='auto')
    ax[1][0].set_title('Denoisied TTC [GS]')
    ax[1][0].set_xlabel('$t_1$')

    ax[1][1].imshow(C_denoised_direct, cmap='plasma', origin='lower', aspect='auto')
    ax[1][1].set_title('Denoisied TTC [CNN] only')
    ax[1][1].set_xlabel('$t_1$')   
    plt.tight_layout()
    plt.show()
    
    
    # for i in range(len(ttc_p)):
    #     plt.imshow(ttc_p[i], cmap='inferno', origin='lower', 
    #               extent=[0, 10, 0, 10], aspect='auto')
    #     plt.show()
    #     plt.imshow(ttc_n[i], cmap='inferno', origin='lower', 
    #               extent=[0, 10, 0, 10], aspect='auto')
    #     plt.show()
    
    # oo, inp = ttc_dg.generate_random_set(2, 'multi-systems', 100, 0.5,4 ,0.9,save_data=False)
    
    # for iii in oo: 
    #     plt.imshow(iii/np.max(iii), cmap='plasma', origin='lower', 
    #               extent=[0, 10, 0, 10], aspect='auto')
    #     plt.show()
        
    # x = np.linspace(0, 10, oo[0].shape[0])  # Corresponding to the first time dimension
    # y = np.linspace(0, 10, oo[0].shape[1]) 
    
    # X, Y = np.meshgrid(x,y)
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, oo[9], cmap='viridis', edgecolor='none')
    # ax.view_init(elev=10, azim=60)
    # # Add labels
    # ax.set_xlim([10,0])


    
    # ax.set_ylim([10,0])
    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')
    
    # # Show the plot
    # plt.show()
    
    
    #ttc_output_data, ttc_input_data = np.load('./Datasets/TTC/721124multi-systems_TTC_pure_mirrored_250000.npy') , np.load('./Datasets/TTC/721124multi-systems_TTC_noisy_mirrored_250000.npy')
    
    #ttc_output_data, ttc_input_data = np.load('./Datasets/TTC/521344aging_TTC_pure_160000.npy') , np.load('./Datasets/TTC/521344aging_TTC_noisy_160000.npy')

    #ttc_output_data_1, ttc_input_data_1 =np.load('./Datasets/TTC/321514normal_TTC_pure_160000.npy') , np.load('./Datasets/TTC/321514normal_TTC_noisy_160000.npy')  
    #ttc_output_data_2, ttc_input_data_2 =np.load('./Datasets/TTC/321317aging_TTC_pure_80000.npy') , np.load('./Datasets/TTC/321317aging_TTC_noisy_80000.npy')  
    
    #ttc_output_data_1 = ttc_output_data_1[0:40000]
    #ttc_input_data_1 = ttc_input_data_1[0:40000]
    #ttc_output_data = np.concatenate((ttc_output_data_1, ttc_output_data_2))
    #ttc_input_data = np.concatenate((ttc_input_data_1, ttc_input_data_2))
    

  
    #C_denoised = ttc_ae.test_model("model_"+model_name, 100, test_data)
      
    
    

    
    
    # C_denoised = ttc_ae.test_model("model_"+model_name, 100, test_data)
    # ig, ax = plt.subplots(1, 3, figsize=(16, 5))
       
    # ax[0].imshow(ttc_output_data[idx], cmap='plasma', origin='lower', 
    #          extent=[0, 10, 0, 10], aspect='auto')
    # ax[0].set_title('Pure TTC')
    # ax[0].set_xlabel('$t_1$')
    # ax[0].set_ylabel('$t_2$')
    
    # ax[1].imshow(test_data, cmap='plasma', origin='lower', 
    #          extent=[0, 10, 0, 10], aspect='auto')
    # ax[1].set_title('Noisiy TTC')
    # ax[1].set_xlabel('$t_1$')
    # ax[1].set_ylabel('$t_2$')
       
    # ax[2].imshow(C_denoised, cmap='plasma', origin='lower', aspect='auto')
    # ax[2].set_title('Denoisied TTC')
    # ax[2].set_xlabel('$t_1$')
       
    # plt.tight_layout()
    # plt.show()
    
    
#%%
from TTC_data_generator import TTCDataGenerator
from TTC_denoising_AI import TTCAutoencoder, TTCGANs, TTCCNN
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt    
from scipy.ndimage import gaussian_filter



    
    
ttc_dg = TTCDataGenerator()    
#ttc_output_data, ttc_input_data = np.load('./Datasets/TTC/721124multi-systems_TTC_pure_mirrored_250000.npy') , np.load('./Datasets/TTC/721124multi-systems_TTC_noisy_mirrored_250000.npy')
#ttc_output_data, ttc_input_data = ttc_dg.generate_random_set(10000, 'multi-systems', 100, 0.5,4 ,0.9,save_data=True)
ttc_output_data, ttc_input_data = ttc_dg.new_data_set_generator(6000,35)
ttc_cnn = TTCCNN()
cnn_model = ttc_cnn.build_model(input_shape=(100,100,1))
history, model_name = ttc_cnn.fit_model(cnn_model, ttc_input_data, ttc_output_data, 15, 32, 0.2, save_model=True)


idx = 1
test_data = ttc_input_data[idx]  

#test_data = inp[1]  
noise_mask = np.random.poisson(100, size= (100,100))
noise_mask = (noise_mask - np.min(noise_mask)) / (np.max(noise_mask) - np.min(noise_mask))    

#plt.imshow(noise_mask, cmap='plasma', origin='lower', 
          #extent=[0, 10, 0, 10], aspect='auto')
#plt.show()
C_denoised_direct = ttc_cnn.test_model("model_"+model_name, 100, test_data)
C_t_d = gaussian_filter(test_data, sigma=1.0)
#C_denoised = ttc_cnn.test_model("model_"+model_name, 100, C_t_d)

  

ig, ax = plt.subplots(2, 2, figsize=(16, 16))
   
ax[0][0].imshow( ttc_output_data[idx]  , cmap='plasma', origin='lower', 
         extent=[0, 10, 0, 10], aspect='auto')
ax[0][0].set_title('Pure TTC')
ax[0][0].set_xlabel('$t_1$')
ax[0][0].set_ylabel('$t_2$')

ax[0][1].imshow(test_data, cmap='plasma', origin='lower', 
         extent=[0, 10, 0, 10], aspect='auto')
ax[0][1].set_title('Noisy TTC')
ax[0][1].set_xlabel('$t_1$')
ax[0][1].set_ylabel('$t_2$')
   
ax[1][0].imshow(C_t_d, cmap='plasma', origin='lower', aspect='auto')
ax[1][0].set_title('Denoisied TTC [CNN+GS]')
ax[1][0].set_xlabel('$t_1$')

ax[1][1].imshow(C_denoised_direct, cmap='plasma', origin='lower', aspect='auto')
ax[1][1].set_title('Denoisied TTC [CNN] only')
ax[1][1].set_xlabel('$t_1$')   
plt.tight_layout()
plt.show()


#%% benchmarking 

from skimage.metrics import structural_similarity as ssim

def compute_similarity(reference, test, alpha=0.01):
    """
    Compute a combined similarity score using SSIM and NCC.
    
    Parameters:
    - reference: Ground truth image.
    - test: Image to compare.
    - alpha: Weight factor (0 to 1), default 0.5 (equal SSIM & NCC contribution).
    
    Returns:
    - SSIM value
    - NCC value
    - Combined similarity score
    """
    # Compute SSIM
    ssim_value = ssim(reference, test, data_range=test.max() - test.min())

    # Compute NCC
    mean_ref = np.mean(reference)
    mean_test = np.mean(test)
    numerator = np.sum((reference - mean_ref) * (test - mean_test))
    denominator = np.sqrt(np.sum((reference - mean_ref) ** 2) * np.sum((test - mean_test) ** 2))
    ncc_value = numerator / denominator if denominator != 0 else 0

    # Compute combined similarity score
    similarity_score = alpha * ssim_value + (1 - alpha) * ncc_value
    
    return ssim_value, ncc_value, similarity_score



def calculate_snr(pure_ttc, denoised_ttc, noisy_ttc):
    """
    Calculate the Signal-to-Noise Ratio (SNR) as a percentage between the pure and denoised TTC images.

    Parameters:
        pure_ttc (numpy array): Pure TTC image (ground truth).
        denoised_ttc (numpy array): Denoised TTC image.

    Returns:
        float: SNR as a percentage.
    """
    # Ensure the inputs are numpy arrays
    pure_ttc = pure_ttc.astype(np.float32)
    denoised_ttc = denoised_ttc.astype(np.float32)
    noisy_ttc = noisy_ttc.astype(np.float32)
    noise = abs(noisy_ttc - pure_ttc)
    # Calculate mean squared signal
    mean_squared_signal = np.mean(denoised_ttc)
    noise_std = np.std(noise)
    snr = 10 * np.log10(mean_squared_signal**2 / noise_std**2) if noise_std != 0 else float('inf')

    return snr

outo, ino =  ttc_dg.new_data_set_generator(1,40)

snr_gs = []

snr_cnn = []


for i in range(len(outo)):
    C_denoised_direct = ttc_cnn.test_model("model_"+model_name, 100, ino[i])
    C_t_d = gaussian_filter(ino[i], sigma=1.0)
    C_denoised_direct = C_denoised_direct.reshape((100,100))
    C_t_d_normalized =  C_t_d / np.max(C_t_d)
    ground_t_normalized = outo[i] / np.max(outo[i])
    noisy_normalized = ino[i] / np.max(ino[i])

    _,_,gs_ = compute_similarity(ground_t_normalized, C_t_d_normalized)
    _,_,cnn_ = compute_similarity(ground_t_normalized, C_denoised_direct)
    snr_gs.append(gs_)
    snr_cnn.append(cnn_)
    
    ig, ax = plt.subplots(2, 2, figsize=(16, 16))
       
    ax[0][0].imshow( ground_t_normalized  , cmap='plasma', origin='lower', 
              extent=[0, 10, 0, 10], aspect='auto')
    ax[0][0].set_title('Pure TTC')
    ax[0][0].set_xlabel('$t_1$')
    ax[0][0].set_ylabel('$t_2$')
    
    ax[0][1].imshow(noisy_normalized, cmap='plasma', origin='lower', 
              extent=[0, 10, 0, 10], aspect='auto')
    ax[0][1].set_title('Noisy TTC')
    ax[0][1].set_xlabel('$t_1$')
    ax[0][1].set_ylabel('$t_2$')
       
    ax[1][0].imshow(C_t_d_normalized, cmap='plasma', origin='lower', aspect='auto')
    ax[1][0].set_title('Denoisied TTC [GS]')
    ax[1][0].set_xlabel('$t_1$')
    
    ax[1][1].imshow(C_denoised_direct, cmap='plasma', origin='lower', aspect='auto')
    ax[1][1].set_title('Denoisied TTC [CNN]')
    ax[1][1].set_xlabel('$t_1$')   
    plt.tight_layout()
    plt.show()



plt.plot(range(len(snr_gs)), snr_gs, label= 'GS')
plt.plot(range(len(snr_gs)), snr_cnn, label= 'CNN')
plt.xlabel('noise factor')
plt.ylabel('NCC value')
plt.legend()
plt.show()









    