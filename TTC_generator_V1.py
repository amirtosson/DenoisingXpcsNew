#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 08:20:24 2025

@author: tosson
"""


import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

# Parameters
N = 100  # Number of time points
number_of_frames = 2000
delta_t = 0.002
t = np.linspace(0, delta_t*number_of_frames, N)  # Time array
tau = 1.0  # Decay time constant
beta = 0.5  # Contrast parameter (often < 1 for XPCS)
noise_level = 1  # Additive noise level
alpha = 2


I_beam = 1e12
mu_abs = 0.139
absorber_thickness_cm = range(0,16,2)
mu_sample = 0.124 
sample_thickness = 0.5
ssd = 400

P_theta = 0.1



def calculate_transmitted_flux(I0, mu, d):
    #return delta_t * I0 * np.exp(-mu * d) * (0.03/ssd**2) * np.exp(-0.124 * sample_thickness) 
    return I0 * np.exp(-mu * d)

def calculate_scattered_photons_per_frame(I0, mu, d):
    return delta_t *I0 * np.exp(-mu * d)/(1000*500)* np.exp(-0.124 * 0.5)*0.02


# Define the exponential decay core function
def exponential_decay(t1, t2, tau):
    return np.exp(-np.abs(t1 - t2) / tau)

def stretched_exponential(t1, t2, tau, beta):
    return np.exp(-(np.abs(t1 - t2) / tau)**beta)


def aging_exponential(t1, t2, tau0, alpha):
    tau = tau0 * (1 + alpha * np.minimum(t1, t2))  # Aging time constant
    return np.exp(-np.abs(t1 - t2) / tau)


def calculate_snr(pure_ttc, denoised_ttc):
    """
    Calculate the Signal-to-Noise Ratio (SNR) as a percentage between the pure and denoised TTC images.

    Parameters:
        pure_ttc (numpy array): Pure TTC image (ground truth).
        denoised_ttc (numpy array): Denoised TTC image.

    Returns:
        float: SNR as a percentage.
    """
    # Ensure the inputs are numpy arrays
    pure_ttc = np.asarray(pure_ttc)
    denoised_ttc = np.asarray(denoised_ttc)

    # Calculate mean squared signal
    mean_squared_signal = np.mean(pure_ttc**2)

    # Calculate mean squared noise
    noise = pure_ttc - denoised_ttc
    mean_squared_noise = np.mean(noise**2)

    # Avoid division by zero
    if mean_squared_signal == 0:
        return 0.0  # If there is no signal, SNR is 0%

    # Calculate SNR as a percentage
    snr_percentage = (1 - (mean_squared_noise - mean_squared_signal)) * 100
    return snr_percentage

def calculate_snr_AT(pure_ttc, denoised_ttc, noise_mask):
    snr_truth = np.mean(pure_ttc)/np.std(noise_mask)
    snr_generated = np.mean(denoised_ttc)/np.std(noise_mask)
    return 10*np.log10(snr_truth**2), 10*np.log10(snr_generated**2)

def calculate_psnr(pure_ttc, denoised_ttc, max_value=1.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the pure and denoised TTC images.

    Parameters:
        pure_ttc (numpy array): Pure TTC image (ground truth).
        denoised_ttc (numpy array): Denoised TTC image.
        max_value (float): Maximum possible pixel value in the image (default is 1.0 for normalized images).

    Returns:
        float: PSNR in decibels (dB).
    """
    pure_ttc = np.asarray(pure_ttc)
    denoised_ttc = np.asarray(denoised_ttc)
    pure_ttc = (pure_ttc - np.min(pure_ttc)) / (np.max(pure_ttc) - np.min(pure_ttc))
    denoised_ttc = (denoised_ttc - np.min(denoised_ttc)) / (np.max(denoised_ttc) - np.min(denoised_ttc))
    mse = np.mean((pure_ttc - denoised_ttc) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float('inf')  # Infinite PSNR if the images are identical

    # Calculate PSNR
    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr

def calculate_and_compare_histograms(pure_ttc, denoised_ttc):
    data_range = pure_ttc.max() - pure_ttc.min()
    similarity, _ =ssim(pure_ttc , denoised_ttc, data_range=data_range,full=True)
    return similarity

def wavelet_denoise_ttc(noisy_ttc, wavelet='db4', level=4, mode='soft'):
    """
    Denoise a noisy TTC function using wavelet denoising.

    Parameters:
        noisy_ttc (numpy array): Noisy TTC matrix.
        wavelet (str): Wavelet type (e.g., 'db4', 'sym4').
        level (int): Decomposition level.
        mode (str): Thresholding mode ('soft' or 'hard').

    Returns:
        numpy array: Denoised TTC matrix.
    """
    # Apply wavelet denoising to each row of the TTC matrix
    denoised_ttc = np.zeros_like(noisy_ttc)
    for i in range(noisy_ttc.shape[0]):
        # Decompose the signal
        coeffs = pywt.wavedec(noisy_ttc[i, :], wavelet, level=level)
        # Threshold the coefficients
        coeffs_thresh = [pywt.threshold(c, value=np.std(c) * np.sqrt(2 * np.log(len(c))), mode=mode) for c in coeffs]
        # Reconstruct the signal
        denoised_ttc[i, :] = pywt.waverec(coeffs_thresh, wavelet)
    return denoised_ttc

def gaussian_smooth_ttc(noisy_ttc, sigma=1.0):
    """
    Apply Gaussian smoothing to a noisy TTC matrix.

    Parameters:
        noisy_ttc (numpy array): Noisy TTC matrix.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        numpy array: Smoothed TTC matrix.
    """
    return gaussian_filter(noisy_ttc, sigma=sigma)


def pca_denoise_ttc(noisy_ttc, n_components=10):
    """
    Denoise a noisy TTC matrix using PCA.

    Parameters:
        noisy_ttc (numpy array): Noisy TTC matrix.
        n_components (int): Number of principal components to retain.

    Returns:
        numpy array: Denoised TTC matrix.
    """
    pca = PCA(n_components=n_components)
    denoised_ttc = pca.inverse_transform(pca.fit_transform(noisy_ttc))
    return denoised_ttc

def svd_denoising(ttc_data, rank=10):
    U, s, Vt = np.linalg.svd(ttc_data, full_matrices=False)
    # Retain only the top `rank` singular values
    s[rank:] = 0
    # Reconstruct the denoised matrix
    denoised_ttc = U @ np.diag(s) @ Vt
    return denoised_ttc

# Example usage

# Generate the TTC matrix
C = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        C[i, j] = 1 + beta * aging_exponential(t[i], t[j], tau, alpha)
        #C[i, j] = 1 + beta * exponential_decay(t[i], t[j], tau)


C = (C - np.min(C)) / (np.max(C) - np.min(C))

snr = []
snr_gs = []
snr_pca = []
snr_wl = []
snr_svd = []


sf_gs = []
sf_pca = []
sf_wl = []


average_photons_per_frame = calculate_transmitted_flux(I_beam, mu_abs, 2)
noise_mask = np.random.poisson(average_photons_per_frame, size= (100,100))
noise_mask = (noise_mask - np.min(noise_mask)) / (np.max(noise_mask) - np.min(noise_mask))

# Add noise to simulate experimental data
for abs_thichness in absorber_thickness_cm:
    
    #average_photons_per_frame = calculate_transmitted_flux(I_beam, mu_abs, abs_thichness)
    
    #intensity_scale = np.sum(np.random.poisson(average_photons_per_frame, number_of_frames))
    #print(average_photons_per_frame)
    #print(intensity_scale)
    
    #noise_mask = (np.random.poisson(intensity_scale, size= (100,100)) ) + (np.random.normal(average_photons_per_frame, 10, (100,100)))
    #noise_mask = np.random.poisson(average_photons_per_frame, size= (100,100))
    #noise_mask = (noise_mask - np.min(noise_mask)) / (np.max(noise_mask) - np.min(noise_mask))

    #C_noisy = (C_noisy - np.min(C_noisy)) / (np.max(C_noisy) - np.min(C_noisy))
    C_noisy = C + abs_thichness*0.1*noise_mask
    
    
    C_denoise_wavelet = wavelet_denoise_ttc(C_noisy)
    C_denoise_gaussian_smooth = gaussian_smooth_ttc(C_noisy, 1)
    C_denoise_PCA = pca_denoise_ttc(C_noisy, 3)
    C_denoise_svd = svd_denoising(C_noisy, 10)

    
    
    fig, ax = plt.subplots(2, 3, figsize=(16, 16))
    
    ax[0][0].imshow(C, cmap='plasma', origin='lower', 
                 extent=[0, 10, 0, 10], aspect='auto')
    ax[0][0].set_title('Pure TTC')
    ax[0][0].set_xlabel('$t_1$')
    ax[0][0].set_ylabel('$t_2$')
    
    ax[0][1].imshow(C_noisy, cmap='plasma', origin='lower', aspect='auto')
    ax[0][1].set_title('TTC with noise')
    ax[0][1].set_xlabel('$t_1$')
    
    ax[0][2].imshow(C_denoise_gaussian_smooth, cmap='plasma', origin='lower', aspect='auto')
    ax[0][2].set_title('TTC denoised (Gaussian Smoothing)')
    ax[0][2].set_xlabel('$t_1$')
    
    ax[1][0].imshow(C_denoise_wavelet, cmap='plasma', origin='lower', aspect='auto')
    ax[1][0].set_title('TTC denoised (Wavelet)')
    ax[1][0].set_xlabel('$t_1$')
    
    ax[1][1].imshow(C_denoise_PCA, cmap='plasma', origin='lower', aspect='auto')
    ax[1][1].set_title('TTC denoised (PCA)')
    ax[1][1].set_xlabel('$t_1$')
    
    ax[1][2].imshow(C_denoise_svd, cmap='plasma', origin='lower', aspect='auto')
    ax[1][2].set_title('TTC denoised (SVD)')
    ax[1][2].set_xlabel('$t_1$')
    
    plt.tight_layout()
    
    plt.savefig("./Images/TTC/TTC_Data_Simulated_abs_"+ str(abs_thichness)+'.png', format='png', dpi=600)

    plt.show()
    
    
    s_o = calculate_psnr(C, C_noisy)
    s_g = calculate_psnr(C, C_denoise_gaussian_smooth)
    s_wl = calculate_psnr(C, C_denoise_wavelet)
    s_pca = calculate_psnr(C, C_denoise_PCA)
    s_svd = calculate_psnr(C, C_denoise_svd)

    snr.append(s_o)
    snr_gs.append(s_g)
    snr_wl.append(s_wl)
    snr_pca.append(s_pca)
    snr_svd.append(s_svd)
    #sf_gs.append(calculate_and_compare_histograms(C, C_denoise_gaussian_smooth))
    #sf_pca.append(calculate_and_compare_histograms(C, C_denoise_PCA))
    #sf_wl.append(calculate_and_compare_histograms(C, C_denoise_wavelet))

plt.plot(absorber_thickness_cm, snr, label= 'Original SNR')
plt.plot(absorber_thickness_cm, snr_gs, label= 'SNR after denoising (GSF)')
plt.plot(absorber_thickness_cm, snr_wl, label= 'SNR after denoising (WL)')
plt.plot(absorber_thickness_cm, snr_pca, label= 'SNR after denoising (PCA)')
plt.plot(absorber_thickness_cm, snr_svd, label= 'SNR after denoising (SVD)')
plt.xlabel('Absorber thickness')
plt.ylabel('SNR')
#plt.plot(absorber_thickness_cm, snr_pca)
#plt.yscale("log")
plt.legend()
plt.savefig("./Images/TTC/TTC_SNR"+'.png', format='png', dpi=600)


plt.show()



#plt.plot(absorber_thickness_cm, sf_gs, label= 'Similarity (GSF)')
#plt.plot(absorber_thickness_cm, sf_wl, label= 'Similarity  (WL)')
#plt.plot(absorber_thickness_cm, sf_pca, label= 'Similarity  (PCA)')

#plt.plot(absorber_thickness_cm, snr_pca)
#plt.yscale("log")
#plt.legend()
#plt.show()



#%%
from TTC_data_generator import TTCDataGenerator

ttc_dg =  TTCDataGenerator()

ttc_input_data, ttc_output_data = ttc_dg.generate_random_set(1000)




#%%

noise_mask = np.random.poisson(C * intensity_scale)

C_noisy_test = C + noise_mask

fig, ax = plt.subplots(1, 4, figsize=(12, 5))

ax[0].imshow(C, cmap='viridis', origin='lower', 
             extent=[0, 10, 0, 10], aspect='auto')
ax[0].set_title('Pure TTC')
ax[0].set_xlabel('$t_1$')
ax[0].set_ylabel('$t_2$')

ax[1].imshow(noise_mask, cmap='viridis', origin='lower', aspect='auto')
ax[1].set_title('TTC with noise')
ax[1].set_xlabel('$t_1$')

ax[2].imshow(C_noisy_test, cmap='viridis', origin='lower', aspect='auto')
ax[2].set_title('TTC denoised (PCA)')
ax[2].set_xlabel('$t_1$')

ax[3].imshow(C_noisy_test - noise_mask, cmap='viridis', origin='lower', aspect='auto')
ax[3].set_title('TTC denoised (Gaussian Smoothing)')
ax[3].set_xlabel('$t_1$')

plt.tight_layout()
plt.show()


snr_test = calculate_snr(C, (C_noisy_test - noise_mask))

print(snr_test)

print(C_noisy_test - noise_mask-C)


#%%




import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz


def generate_TTC(stationary=False, num_points=128, noise_level=0.001, beta=0.5):
    """
    Generate a synthetic two-time correlation matrix C(t1, t2).
    
    Parameters:
        stationary (bool): If True, use stationary dynamics (constant relaxation rate).
                           If False, simulate aging (time-dependent relaxation rate).
        num_points (int): Number of time points (N x N matrix).
        noise_level (float): Amplitude of Gaussian noise added to intensities.
        beta (float): Stretching exponent for non-stationary dynamics (e.g., 0.5 for aging).
    
    Returns:
        C (np.ndarray): Two-time correlation matrix of shape (num_points, num_points).
    """
    # Time array
    t = np.linspace(-1, 1, num_points)
    
    # Simulate time-dependent relaxation rate Γ(t)
    if stationary:
        Gamma = 1.0 * np.ones_like(t)  # Constant relaxation rate
    else:
        Gamma = 1 + 0.5 * np.exp(-t / 5)  # Aging: relaxation slows down over time
    
    # Generate intensity fluctuations I(t)
    I = np.zeros(num_points)
    I[0] = 1.0  # Initial condition
    dt = t[1] - t[0]
    for i in range(1, num_points):
        # Langevin-like dynamics with time-dependent damping
        dI = -Gamma[i] * I[i-1] * dt + np.sqrt(dt) * np.random.randn()
        I[i] = I[i-1] + dI
    
    # Add noise to intensity (simulate photon counting)
    I += noise_level * np.random.randn(num_points)
    
    # Compute two-time correlation matrix C(t1, t2)
    C = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            C[i, j] = np.mean(I[:num_points - max(i, j)] * I[max(i, j) - min(i, j):num_points - min(i, j)])

    # Normalize by average intensities
    C /= (np.mean(I) ** 2)
    
    return C



def generate_TTC_with_HI(num_points=128, noise_level=0.1, gamma=1.0, eta=0.3):
    """
    Generate a TTC matrix with hydrodynamic interactions using a Generalized Langevin Equation (GLE).
    
    Parameters:
        num_points (int): Number of time points (N x N matrix).
        noise_level (float): Amplitude of noise.
        gamma (float): Friction coefficient (inverse relaxation time).
        eta (float): Hydrodynamic coupling strength.
    
    Returns:
        C (np.ndarray): Two-time correlation matrix of shape (num_points, num_points).
    """
    # Time array
    t = np.linspace(0, 10, num_points)
    dt = t[1] - t[0]
    
    # Hydrodynamic memory kernel (simplified Oseen tensor approximation)
    def memory_kernel(tau):
        return (1 + eta) * np.exp(-gamma * tau)  # Exponential decay with HI coupling
    
    # Construct memory kernel matrix
    kernel = np.array([memory_kernel(ti) for ti in t])
    K = toeplitz(kernel)  # Toeplitz matrix for convolution
    
    # Generate correlated noise with hydrodynamic interactions
    dW = np.random.randn(num_points)
    xi = np.dot(K, dW) * np.sqrt(dt)  # Convolve noise with memory kernel
    
    # Simulate particle displacement (GLE)
    x = np.zeros(num_points)
    for i in range(1, num_points):
        dx = -gamma * x[i-1] * dt + xi[i-1]
        x[i] = x[i-1] + dx
    
    # Intensity fluctuations (proportional to displacement squared)
    I = 1 + 0.5 * x**2 + noise_level * np.random.randn(num_points)
    
    # Compute two-time correlation matrix C(t1, t2)
    C = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            C[i, j] = np.correlate(I[i:], I[:num_points-i], mode='valid')[0] if i <= j else C[j, i]
    
    # Normalize
    C /= np.max(C)
    
    return C

# Generate TTC matrices with and without hydrodynamic interactions
C_without_HI = generate_TTC(stationary=False)  # From previous code (no HI)
C_with_HI = generate_TTC_with_HI(eta=0.5)     # With hydrodynamic coupling

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(C_without_HI, cmap='viridis', origin='lower', 
             extent=[0, 10, 0, 10], aspect='auto')
ax[0].set_title('Without Hydrodynamic Interactions')
ax[0].set_xlabel('$t_1$')
ax[0].set_ylabel('$t_2$')

ax[1].imshow(C_with_HI, cmap='viridis', origin='lower', 
             extent=[0, 10, 0, 10], aspect='auto')
ax[1].set_title('With Hydrodynamic Interactions ($\\eta=0.5$)')
ax[1].set_xlabel('$t_1$')

plt.tight_layout()
plt.show()