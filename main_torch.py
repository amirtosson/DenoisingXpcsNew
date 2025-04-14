"""
Created on Mon Feb  3 12:07:24 2025

@author: tosson
"""

from TTC_denoising_AI_torch import TTCCNN
from TTC_data_generator import TTCDataGenerator

if __name__ == '__main__':
    #ttc_dg = TTCDataGenerator()
    #ttc_output_data, ttc_input_data = ttc_dg.new_data_set_generator(600,35,save_data=False)
    #ttc_input_data = ttc_input_data.reshape(-1, 1, 100, 100)
    #ttc_output_data = ttc_output_data.reshape(-1, 1, 100, 100)
    ttc_cnn = TTCCNN()
    model = ttc_cnn.build_model()
    #ttc_cnn.fit_model(model, ttc_input_data, ttc_output_data, 1, 32, 0.2, save_model=False)