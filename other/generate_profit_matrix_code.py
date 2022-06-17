
import itertools
import traceback
from datetime import datetime
import decimal
import enum
import os
from decimal import Decimal
from enum import Enum
from functools import cmp_to_key
from os import listdir
from os.path import join, basename
from pathlib import Path

import numpy as np
from skimage import measure, filters
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
import copy
import time
import cv2
#from google.colab.patches import cv2_imshow
import random
from itertools import combinations
import pickle
from collections import defaultdict


import time
from multiprocessing.pool import ThreadPool

def main():
    folder_path: str = 'D:/viterbi linkage/dataset/'

    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    output_folder = folder_path + 'output_unet_seg_finetune//'
    save_dir = folder_path + 'save_directory_enhancement/'

    input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                         'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']

    all_segmented_filename_list = listdir(segmentation_folder)
    all_segmented_filename_list.sort()

    series_frame_num_prof_matrix_dict_dict = {}

    print("series_frame_num_prof_matrix_dict_dict = defaultdict(dict)")
    for series in input_series_list:
        segmented_filename_list: list = derive_segmented_filename_list_by_series(series, all_segmented_filename_list)

        frame_num_prof_matrix_dict: dict = derive_frame_num_prof_matrix_dict(segmentation_folder, output_folder, series, segmented_filename_list)

        frame_num_list = sorted(frame_num_prof_matrix_dict.keys())
        for frame_num in frame_num_list:
            profit_mtx_arr = frame_num_prof_matrix_dict[frame_num]

            total_row = profit_mtx_arr.shape[0]
            total_col = profit_mtx_arr.shape[1]
            print(f"series_frame_num_prof_matrix_dict_dict[\"{series}\"][{frame_num}] = np.array([")
            for row_idx in range(total_row):
                print("[", end='')
                for col_idx in range(total_col):
                    print(profit_mtx_arr[row_idx][col_idx], end='')
                    if col_idx != (total_col-1):
                        print(", ", end='')

                if row_idx != (total_row-1):    print("], ")
                else:                           print("]")
            print("])")

            print()
            # exit()






def derive_segmented_filename_list_by_series(series: str, segmented_filename_list: list):
    result_segmented_filename_list: list = []

    for segmented_filename in segmented_filename_list:
        if series in segmented_filename:
            result_segmented_filename_list.append(segmented_filename)

    return result_segmented_filename_list




def derive_frame_num_prof_matrix_dict(segmentation_folder_path: str, output_folder_path: str, series: str, segmented_filename_list):
    frame_num_prof_matrix_dict: dict = {}

    #get the first image (frame 0) and label the cells:
    img = plt.imread(segmentation_folder_path + segmented_filename_list[0])

    label_img = measure.label(img, background=0, connectivity=1)
    cellnb_img = np.max(label_img)

    for frame_num in range(1, len(segmented_filename_list)):
        # get next frame and number of cells next frame
        img_next = plt.imread(segmentation_folder_path + '/' + segmented_filename_list[frame_num])

        label_img_next = measure.label(img_next, background=0, connectivity=1)
        cellnb_img_next = np.max(label_img_next)

        #create empty dataframe for element of profit matrix C
        prof_mat = np.zeros( (cellnb_img, cellnb_img_next), dtype=float)

        #loop through all combinations of cells in this and the next frame
        for cellnb_i in range(cellnb_img):
            #cellnb i + 1 because cellnumbering in output files starts from 1
            cell_i_filename = "mother_" + segmented_filename_list[frame_num][:-4] + "_Cell" + str(cellnb_i + 1).zfill(2) + ".png"
            cell_i = plt.imread(output_folder_path + series + '/' + cell_i_filename)
            #predictions are for each cell in curr img
            cell_i_props = measure.regionprops(label_img_next, intensity_image=cell_i) #label_img_next是二值图像为255，无intensity。需要与output中的预测的细胞一一对应，预测细胞有intensity
            for cellnb_j in range(cellnb_img_next):
                #calculate profit score from mean intensity neural network output in segmented cell area
                prof_mat[cellnb_i, cellnb_j] = cell_i_props[cellnb_j].mean_intensity         #得到填充矩阵size = max(cellnb_img, cellnb_img_next)：先用预测的每一个细胞的mean_intensity填满cellnb_img, cellnb_img_next行和列

        frame_num_prof_matrix_dict[frame_num] = prof_mat

        #make next frame current frame
        cellnb_img = cellnb_img_next
        label_img = label_img_next

    return frame_num_prof_matrix_dict


if __name__ == '__main__':
    main()