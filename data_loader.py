#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:51:21 2022
Author: Amir Tosson
Email: amir.tosson@uni-siegen.de  
Project: DenoisingXPCS
File : data_loader.py
Class: data_loader
"""

import glob
import csv
import numpy as np


class DataLoader:
    
    def __init__(self):
        super().__init__
        
    def load_exp_datasets(self, remove_last_element=True, data_structure_id='siegen'):
        all_data = []
        files_list = glob.glob(r"./ExpDatasets/m0*.csv")
        
        files_list.sort()
        if data_structure_id == 'siegen':
            for _file in files_list:
                print(_file)
                file = open(_file)
                csvreader = csv.reader(file)
                rows = []
                for row in csvreader:
                    row_h = []
                    for t in row:
                        if t == "nan":
                            t = -100.0
                        row_h.append(float(t))
                    rows.append(row_h)
                file.close()
                all_data.append(rows)
            all_data = np.array(all_data) 
            
            if remove_last_element:
                all_data = all_data[:,:, :-1]
        
        return all_data, all_data[0][0]