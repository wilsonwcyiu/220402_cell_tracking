# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:41:40 2021

@author: 13784
"""
import os
from os import listdir
from os.path import join, basename
import numpy as np
from skimage import measure, filters
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
import copy
import time
# import cv2
import random
from itertools import combinations
import pickle

def main():
    pkl_file_path: str = "G:/My Drive/leiden_university_course_materials/thesis/260101_thesis_followup/260106_Dicts/viterbi_results_dict_adj2.pkl"
    track_file_path = open_track_dictionary(pkl_file_path)
        
    print("finish")



'''opening a track dictionary from file'''
def open_track_dictionary(save_file):
    pickle_in = open(save_file,"rb")
    dictionary = pickle.load(pickle_in)
    return dictionary



if __name__ == '__main__':
    main()