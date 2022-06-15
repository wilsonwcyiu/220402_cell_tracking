# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:31:51 2021

@author: 13784
"""
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

# from main.viterbi_adjust3e_refactoring import CellId




def main():

    folder_path: str = 'D:/viterbi linkage/dataset/'

    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    images_folder = folder_path + 'dataset//images//'
    output_folder = folder_path + 'output_unet_seg_finetune//'
    save_dir = folder_path + 'save_directory_enhancement/'


    is_use_thread: bool = False


    date_str: str = datetime.now().strftime("%Y%m%d-%H%M%S") + "/"



    start_time = time.perf_counter()

    input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                         'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    # input_series_list = ['S02']
    # input_series_list = ['S02', 'S03', 'S04']

    all_segmented_filename_list = listdir(segmentation_folder)
    all_segmented_filename_list.sort()

    existing_series_list = derive_existing_series_list(input_series_list, listdir(output_folder))

    series_frame_num_cell_coord_list_dict_dict = {}

    try:
        for series in existing_series_list:
            print(f"working on series: {series}. ")
            frame_num_cell_coord_list_dict = cell_tracking_core_flow(series, segmentation_folder, all_segmented_filename_list, output_folder)
            series_frame_num_cell_coord_list_dict_dict[series] = frame_num_cell_coord_list_dict

            save_cell_coord_to_excel(series_frame_num_cell_coord_list_dict_dict, save_dir)


    except Exception as e:
        time.sleep(2)
        traceback.print_exc()
        time.sleep(2)






    execution_time = time.perf_counter() - start_time

    print(f"Execution time: {np.round(execution_time, 4)} seconds")






def __________object_start_label():
    raise Exception("for labeling only")



class ROUTING_STRATEGY_ENUM(enum.Enum):
    ALL_LAYER = 1
    ONE_LAYER = 2


class CUT_STRATEGY_ENUM(enum.Enum):
    DURING_ROUTING = 1
    AFTER_ROUTING = 2


class BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM(enum.Enum):
    SHARE = 1


class HyperPara():
    def __init__(self, routing_strategy_enum: ROUTING_STRATEGY_ENUM, merge_threshold: float, minimum_track_length: int, cut_threshold: float, is_do_post_adjustment: bool,
                 cut_strategy_enum: CUT_STRATEGY_ENUM, both_cell_below_threshold_strategy_enum: BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM,
                 discount_rate_per_layer: [str, int]):
        self.routing_strategy_enum: ROUTING_STRATEGY_ENUM = routing_strategy_enum
        self.merge_threshold: float = merge_threshold
        self.minimum_track_length: int = minimum_track_length
        self.cut_threshold: float = cut_threshold
        self.is_do_post_adjustment: bool = is_do_post_adjustment
        self.cut_strategy_enum = cut_strategy_enum
        self.both_cell_below_threshold_strategy_enum = both_cell_below_threshold_strategy_enum
        self.discount_rate_per_layer = discount_rate_per_layer   # can be merge_threshold to set as merge_threshold value, or provide an exact value


    def __str__(self):
        return f"routing_strategy_enum: {self.routing_strategy_enum.name}; " \
               f"merge_threshold: {self.merge_threshold}; " \
               f"minimum_track_length: {self.minimum_track_length}; " \
               f"cut_threshold: {self.cut_threshold}; " \
               f"is_do_post_adjustment: {self.is_do_post_adjustment}; " \
               f"cut_strategy_enum: {self.cut_strategy_enum.name}; " \
               f"both_cell_below_threshold_strategy_enum: {self.both_cell_below_threshold_strategy_enum.name}; " \
               f"discount_rate_per_layer: {self.discount_rate_per_layer}; "


    def __str_newlines__(self):
        return f"routing_strategy_enum: {self.routing_strategy_enum.name}; \n" \
               f"merge_threshold: {self.merge_threshold}; \n" \
               f"minimum_track_length: {self.minimum_track_length}; \n" \
               f"cut_threshold: {self.cut_threshold}; \n" \
               f"is_do_post_adjustment: {self.is_do_post_adjustment}; \n" \
               f"cut_strategy_enum: {self.cut_strategy_enum.name}; \n" \
               f"both_cell_below_threshold_strategy_enum: {self.both_cell_below_threshold_strategy_enum.name}; \n" \
               f"discount_rate_per_layer: {self.discount_rate_per_layer}; "

    def __eq__(self, other):
        if self.routing_strategy_enum == other.routing_strategy_enum and \
                self.merge_threshold == other.merge_threshold and \
                self.minimum_track_length == other.minimum_track_length and \
                self.cut_threshold == other.cut_threshold:

            return True

        return False


    def __hash__(self):
        return hash((self.routing_strategy_enum, self.merge_threshold, self.minimum_track_length, self.cut_threshold))



class CellId():

    def __init__(self, start_frame_num: int, cell_idx: int):
        self.start_frame_num = start_frame_num
        self.cell_idx = cell_idx


    def str_short(self):
        return f"{self.start_frame_num}-{self.cell_idx};"


    def __str__(self):
        # return f"CellId(start_frame_num: {self.start_frame_num}; cell_idx: {self.cell_idx})"
        return f"CellId({self.start_frame_num}, {self.cell_idx})"


    def __eq__(self, other):
        if self.start_frame_num == other.start_frame_num and self.cell_idx == other.cell_idx:
            return True

        return False

    def __hash__(self):
        return hash((self.start_frame_num, self.cell_idx))


def compare_cell_id(cell_1, cell_2):
    if cell_1.start_frame_num == cell_2.start_frame_num:
        if cell_1.cell_idx < cell_2.cell_idx:           return -1
        elif cell_1.cell_idx > cell_2.cell_idx:         return 1
        else:                                           raise Exception("cell_1.node_idx == cell_2.node_idx")

    if cell_1.start_frame_num < cell_2.start_frame_num:            return -1
    elif cell_1.start_frame_num > cell_2.start_frame_num:          return 1







def __________flow_function_start_label():
    raise Exception("for labeling only")





def cell_tracking_core_flow(series: str, segmentation_folder: str, all_segmented_filename_list: list, output_folder: str):

    segmented_filename_list: list = derive_segmented_filename_list_by_series(series, all_segmented_filename_list)

    frame_num_cell_coord_list_dict = derive_frame_cell_label_coord(segmentation_folder, output_folder, series, segmented_filename_list)

    return frame_num_cell_coord_list_dict

def __________component_function_start_label():
    raise Exception("for labeling only")







def __________unit_function_start_label():
    raise Exception("for labeling only")


def save_cell_coord_to_excel(series_frame_num_cell_coord_list_dict_dict, excel_output_dir_path: str):
    import pandas as pd

    file_name: str = f"all_series_cell_coord.xlsx"
    filepath = excel_output_dir_path + file_name;
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter') #pip install xlsxwriter

    for series, frame_num_cell_coord_list_dict in series_frame_num_cell_coord_list_dict_dict.items():
        max_col_among_all_frame: int = 0
        for frame_num, cell_coord_list in frame_num_cell_coord_list_dict.items():
            if  len(cell_coord_list)> max_col_among_all_frame:
                max_col_among_all_frame = len(cell_coord_list)

        row_list = []
        for frame_num, cell_coord_list in frame_num_cell_coord_list_dict.items():
            col_list = []
            for cell_coord in cell_coord_list:
                col_list.append(str(cell_coord))

            col_list += [" " for _ in range(max_col_among_all_frame - len(col_list))]
            row_list.append(np.array(col_list))

        tmp_array: np.arrays = np.array(row_list)

        df = pd.DataFrame (tmp_array)
        df.index += 1
        df.to_excel(writer, sheet_name=series, index=True)

        writer.sheets[series].freeze_panes(1, 1)

        for column in df:
            column_length = 10
            col_idx = df.columns.get_loc(column)
            writer.sheets[series].set_column(col_idx, col_idx, column_length)

    writer.save()



def derive_existing_series_list(input_series_list: list, series_dir_list: list):
    existing_series_list:list = []
    for input_series in input_series_list:
        if input_series in series_dir_list:
            existing_series_list.append(input_series)

    return existing_series_list



def derive_segmented_filename_list_by_series(series: str, segmented_filename_list: list):
    result_segmented_filename_list: list = []

    for segmented_filename in segmented_filename_list:
        if series in segmented_filename:
            result_segmented_filename_list.append(segmented_filename)

    return result_segmented_filename_list


#[frame_num] -> idx coord list
def derive_frame_cell_label_coord(segmentation_folder_path: str, output_folder_path: str, series: str, segmented_filename_list):
    frame_num_cell_coord_list_dict: dict = {}

    # max_col: int = 0
    # for frame_idx in range(0, len(segmented_filename_list)):
    #     img = plt.imread(segmentation_folder_path + segmented_filename_list[frame_idx])
    #     label_img = measure.label(img, background=0, connectivity=1)
    #     cellnb_img = np.max(label_img)
    #
    #     max_col = np.max(cellnb_img, max_col)


    for frame_idx in range(0, len(segmented_filename_list)):
        frame_num = frame_idx + 1
        img = plt.imread(segmentation_folder_path + segmented_filename_list[frame_idx])
        label_img = measure.label(img, background=0, connectivity=1)
        cellnb_img = np.max(label_img)

        #loop through all combinations of cells in this and the next frame
        # print(f"--------frame_idx: {frame_idx}:")
        cell_coord_tuple_list = []
        for cellnb_i in range(cellnb_img):
            cell_i_filename = "mother_" + segmented_filename_list[frame_idx][:-4] + "_Cell" + str(cellnb_i + 1).zfill(2) + ".png"
            cell_i = plt.imread(segmentation_folder_path + segmented_filename_list[0])
            # cell_i = plt.imread(output_folder_path + series + '/' + cell_i_filename)
            cell_i_props = measure.regionprops(label_img, intensity_image=cell_i) #label_img_next是二值图像为255，无intensity。需要与output中的预测的细胞一一对应，预测细胞有intensity

            x, y = cell_i_props[cellnb_i].centroid
            x, y = int(x), int(y)

            cell_coord_tuple_list.append((x, y))

            # print(f"cell_idx: {cellnb_i}: x,y={x}, {y}.", end='')

            # if (cellnb_i+1) % 4 == 0:
            #     print()

        # print("\n")

        frame_num_cell_coord_list_dict[frame_num] = cell_coord_tuple_list


    # #make next frame current frame
        # cellnb_img = cellnb_img_next
        # label_img = label_img_next

    return frame_num_cell_coord_list_dict


def __________common_function_start_label():
    raise Exception("for labeling only")



def dev_print(*arg):
    print(*arg)



def save_track_dictionary(dictionary, save_file):
    if not os.path.exists(save_file):
        with open(save_file, 'w'):
            pass
    pickle_out = open(save_file, "wb")
    pickle.dump(dictionary, pickle_out)
    pickle_out.close()



def __________code_validation_function_start_label():
    raise Exception("for labeling only")


def code_validate_if_cellid_not_exist_in_occupation_data(frame_num_node_idx_cell_occupation_list_list_dict, check_cell_id):
    exist_record_tuple_list = []
    for frame_num in frame_num_node_idx_cell_occupation_list_list_dict.keys():
        node_idx_cell_occupation_list_list = frame_num_node_idx_cell_occupation_list_list_dict[frame_num]
        for node_idx, cell_occupation_list in enumerate(node_idx_cell_occupation_list_list):
            if check_cell_id in cell_occupation_list:
                exist_record_tuple_list.append((frame_num, node_idx))

    if len(exist_record_tuple_list) != 0:
        raise Exception("len(exist_record_tuple_list) != 0", check_cell_id.__str__(), exist_record_tuple_list)


def code_validate_track_list(track_tuple_list_list: list):

    for track_tuple_list in track_tuple_list_list:
        frame_idx = track_tuple_list[0][1]

        for track_tuple in track_tuple_list[1:]:
            next_frame_idx = track_tuple[1]

            if next_frame_idx != (frame_idx + 1):
                # print("stnsen", next_frame_idx, (frame_idx + 1))
                raise Exception(track_tuple_list[0], next_frame_idx, track_tuple_list)

            frame_idx = next_frame_idx



def code_validate_cell_probability(cell_1_prob: float, cell_2_prob: float):
    if cell_1_prob == 0: raise Exception("cell_1_prob == 0")
    if cell_2_prob == 0: raise Exception("cell_2_prob == 0")



def code_validate_track(cell_idx_track_list_dict):
    for cell_id, track_list in cell_idx_track_list_dict.items():
        cell_start_frame_num = cell_id.start_frame_num
        tmp_start_frame_num = track_list[0][1] + 1

        if cell_start_frame_num != tmp_start_frame_num:
            raise Exception(cell_id.cell_idx, cell_start_frame_num, tmp_start_frame_num)



def code_validate_track_length(cell_idx_track_list_dict):
    for cell_id, track_list in cell_idx_track_list_dict.items():
        cell_start_frame_num = cell_id.start_frame_num
        tmp_start_frame_num = track_list[0][1] + 1

        if cell_start_frame_num != tmp_start_frame_num:
            raise Exception(cell_id.cell_idx, cell_start_frame_num, tmp_start_frame_num)

        if len(track_list) < 5:
            raise Exception(cell_id.__str__(), tmp_start_frame_num)


def code_validate_track_key_order(cell_idx_track_list_dict):
    max_frame_num = 0
    max_cell_idx = 0
    for cell_id in cell_idx_track_list_dict.keys():
        # print(cell_id.__str__())
        # print(cell_id.start_frame_num < max_frame_num)
        # print(cell_id.start_frame_num == max_frame_num and cell_id.cell_idx < max_cell_idx)

        if cell_id.start_frame_num < max_frame_num:
            raise Exception("cell_id.start_frame_num < max_frame_num")
        if cell_id.start_frame_num == max_frame_num and cell_id.cell_idx < max_cell_idx:
            raise Exception("cell_id.start_frame_num = max_frame_num and cell_id.cell_idx < max_cell_idx")

        max_frame_num = cell_id.start_frame_num
        max_cell_idx = cell_id.cell_idx


if __name__ == '__main__':
    main()