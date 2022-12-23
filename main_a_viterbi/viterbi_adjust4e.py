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




def main():

    folder_path: str = 'D:/viterbi linkage/dataset/'

    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    output_folder = folder_path + 'output_unet_seg_finetune//'
    save_dir = folder_path + 'save_directory_enhancement/'


    is_use_thread: bool = True

    is_use_cell_dependency_feature: bool = False

    ## hyper parameter settings
    routing_strategy_enum_list: list = [ROUTING_STRATEGY_ENUM.ONE_LAYER]
    merge_threshold_list: list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # merge_threshold_list: list = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9]
    minimum_track_length_list: list = [5]
    cut_threshold_list: list = [0.45]
    is_do_post_adjustment_list: list = [False]
    # cut_strategy_enum_list: list = [CUT_STRATEGY_ENUM.AFTER_ROUTING, CUT_STRATEGY_ENUM.DURING_ROUTING]
    cut_strategy_enum_list: list = [CUT_STRATEGY_ENUM.DURING_ROUTING]
    both_cell_below_threshold_strategy_enum_list: list = [BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM.SHARE]
    discount_rate_per_layer: list = [0.8, 0.85, 0.9, 0.95, 0.98] # {"merge_threshold" or number}

    # routing_strategy_enum_list: list = [ROUTING_STRATEGY_ENUM.ALL_LAYER]
    # merge_threshold_list: list = [0.5]
    # minimum_track_length_list: list = [5]
    # cut_threshold_list: list = [0.01]
    # is_do_post_adjustment_list: list = [False]
    # cut_strategy_enum_list: list = [CUT_STRATEGY_ENUM.AFTER_ROUTING]
    # both_cell_below_threshold_strategy_enum_list: list = [BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM.SHARE]
    # discount_rate_per_layer: list = [0.5] #"merge_threshold",

    hyper_para_combination_list = list(itertools.product(routing_strategy_enum_list,
                                                         merge_threshold_list,
                                                         minimum_track_length_list,
                                                         cut_threshold_list,
                                                         is_do_post_adjustment_list,
                                                         cut_strategy_enum_list,
                                                         both_cell_below_threshold_strategy_enum_list,
                                                         discount_rate_per_layer
                                                         ))

    hyper_para_list: list = []
    for hyper_para_combination in hyper_para_combination_list:
        routing_strategy_enum: ROUTING_STRATEGY_ENUM = hyper_para_combination[0]
        merge_threshold: float = hyper_para_combination[1]
        minimum_track_length: int = hyper_para_combination[2]
        cut_threshold: float = hyper_para_combination[3]
        is_do_post_adjustment: bool = hyper_para_combination[4]
        cut_strategy_enum: CUT_STRATEGY_ENUM = hyper_para_combination[5]
        both_cell_below_threshold_strategy_enum: BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM = hyper_para_combination[6]
        discount_rate_per_layer: [str, float] = hyper_para_combination[7]

        hyper_para: HyperPara = HyperPara(routing_strategy_enum,
                                          merge_threshold,
                                          minimum_track_length,
                                          cut_threshold,
                                          is_do_post_adjustment,
                                          cut_strategy_enum,
                                          both_cell_below_threshold_strategy_enum,
                                          discount_rate_per_layer)

        hyper_para_list.append(hyper_para)


    date_str: str = datetime.now().strftime("%Y%m%d-%H%M%S")


    total_para_size: int = len(hyper_para_list)

    # for idx, hyper_para in enumerate(hyper_para_list):
    #     print(f"start. Parameter set: {idx+1}/ {total_para_size}; {hyper_para.__str__()}")
    # exit()


    for idx, hyper_para in enumerate(hyper_para_list):
        para_set_num: int = idx+1

        print(f"start. Parameter set: {para_set_num}/ {total_para_size}; {hyper_para.__str__()}")



        start_time = time.perf_counter()

        input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                             'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
        # input_series_list = ['S01']


        all_segmented_filename_list = listdir(segmentation_folder)
        all_segmented_filename_list.sort()

        existing_series_list = derive_existing_series_list(input_series_list, listdir(output_folder))

        viterbi_result_dict = {}

        try:
            if is_use_thread:
                for input_series in input_series_list:
                    viterbi_result_dict[input_series] = []

                pool = ThreadPool(processes=8)
                thread_list: list = []

                for series in existing_series_list:
                    print(f"working on series: {series}. ")

                    async_result = pool.apply_async(cell_tracking_core_flow, (series, segmentation_folder, all_segmented_filename_list, output_folder, hyper_para, is_use_cell_dependency_feature, )) # tuple of args for foo
                    thread_list.append(async_result)

                total_threads: int = len(thread_list)
                for thread_idx in range(total_threads):
                    return_series, final_result_list = thread_list[thread_idx].get()
                    viterbi_result_dict[return_series] = final_result_list
                    print(f"Thread {thread_idx + 1}/ {total_threads} completed")

            else:
                for series in existing_series_list:
                    print(f"working on series: {series}. ")
                    return_series, final_result_list = cell_tracking_core_flow(series, segmentation_folder, all_segmented_filename_list, output_folder, hyper_para, is_use_cell_dependency_feature)
                    viterbi_result_dict[series] = final_result_list

        except Exception as e:
            time.sleep(1)
            traceback.print_exc()
            print(f"series {series}. para {para_set_num}.  hyper_para: {hyper_para.__str__()}")
            continue



        result_file_name: str = Path(__file__).name.replace(".py", "")

        hyper_para_indicator: str = "R(" +  str(hyper_para.routing_strategy_enum.name)[0:3] + ")_" + \
                                    "M(" + str(merge_threshold) + ")_" + \
                                    "MIN(" + str(minimum_track_length) + ")_" + \
                                    "CT(" + str(cut_threshold) + ")_" + \
                                    "ADJ(" + ("YES" if is_do_post_adjustment else "NO") + ")_" + \
                                    "CS(" +  str(cut_strategy_enum.name)[0] + ")_" + \
                                    "BB(" + str(both_cell_below_threshold_strategy_enum.name)[0] + ")"

        result_dir = save_dir + date_str + "_" + result_file_name + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        abs_save_dir: str = result_dir + result_file_name + "_hp" + str(idx+1).zfill(3) + "__" + hyper_para_indicator
        save_track_dictionary(viterbi_result_dict, abs_save_dir + ".pkl")

        execution_time = time.perf_counter() - start_time
        with open(abs_save_dir + ".txt", 'w') as f:
            f.write(f"Execution time: {np.round(execution_time, 4)} seconds\n")
            f.write("hyper_para--- ID: " + str(idx+1) + "; \n" + hyper_para.__str_newlines__())
            f.write("\n")
            for series in existing_series_list:
                f.write("======================" + str(series) + "================================")
                f.write("\n")

                cell_track_list_list = sorted(viterbi_result_dict[series])
                for cell_track_list in cell_track_list_list:
                    # for cell_track_list in viterbi_result_dict[series]:
                    f.write(str(cell_track_list))
                    f.write("\n")

                f.write("\n\n")



        tmp_abs_save_dir: str = save_dir + result_file_name
        with open(tmp_abs_save_dir + ".txt", 'w') as f:
            f.write(f"Execution time: {np.round(execution_time, 4)} seconds\n")
            f.write("hyper_para--- ID: " + str(idx+1) + "; \n" + hyper_para.__str_newlines__())
            f.write("\n")
            for series in existing_series_list:
                f.write("======================" + str(series) + "================================")
                f.write("\n")

                cell_track_list_list = sorted(viterbi_result_dict[series])
                for cell_track_list in cell_track_list_list:
                    # for cell_track_list in viterbi_result_dict[series]:
                    f.write(str(cell_track_list))
                    f.write("\n")
                f.write("\n\n")



        print("save_track_dictionary: ", abs_save_dir)

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





def cell_tracking_core_flow(series: str, segmentation_folder: str, all_segmented_filename_list: list, output_folder: str, hyper_para: HyperPara, is_use_cell_dependency_feature: bool):

    segmented_filename_list: list = derive_segmented_filename_list_by_series(series, all_segmented_filename_list)

    frame_num_prof_matrix_dict: dict = derive_frame_num_prof_matrix_dict(segmentation_folder, output_folder, series, segmented_filename_list)

    frame_num_prof_matrix_dict = update_below_cut_threshold_value_to_zero(frame_num_prof_matrix_dict, hyper_para.cut_threshold)

    all_track_dict = execute_cell_tracking_task(frame_num_prof_matrix_dict, hyper_para, is_use_cell_dependency_feature)

    all_track_dict = filter_track_dict_by_length(all_track_dict, hyper_para.minimum_track_length)

    sorted_cell_id_key_list = sorted(list(all_track_dict.keys()), key=cmp_to_key(compare_cell_id))
    sorted_dict: dict = {}
    for sorted_key in sorted_cell_id_key_list:
        sorted_dict[sorted_key] = all_track_dict[sorted_key]
    all_track_dict = sorted_dict

    track_list_list: list = list(all_track_dict.values())

    code_validate_track_list(track_list_list)

    is_do_post_adjustment: bool = hyper_para.is_do_post_adjustment
    if is_do_post_adjustment:
        prof_mat_list: list = deprecate_derive_prof_matrix_list(segmentation_folder, output_folder, series, segmented_filename_list)
        final_track_list = post_adjustment_old(track_list_list, prof_mat_list)

        return series, final_track_list

    else:
        return series, track_list_list






def __________component_function_start_label():
    raise Exception("for labeling only")





def execute_cell_tracking_task(frame_num_prof_matrix_dict: dict, hyper_para, is_use_cell_dependency_feature: bool):
    all_cell_id_track_list_dict: dict = defaultdict(list)

    start_frame_num: int = 1
    first_frame_mtx: np.array = frame_num_prof_matrix_dict[start_frame_num]
    total_cell_in_first_frame: int = first_frame_mtx.shape[0]
    to_handle_cell_id_list: list = [CellId(start_frame_num, cell_idx) for cell_idx in range(0, total_cell_in_first_frame)]

    cell_id_frame_num_node_idx_best_index_list_dict_dict: dict = defaultdict(dict)
    cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict: dict = defaultdict(dict)
    cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict: dict = defaultdict(dict)
    cell_dependency_dict: dict = defaultdict(list)
    cell_id_frame_num_track_progress_dict: dict(int) = {}

    frame_num_node_idx_cell_occupation_list_list_dict: dict = initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_prof_matrix_dict)

    max_cell_redo_cnt_record = 0

    print(f"frame {start_frame_num}: ", end='')
    all_cell_id_track_list_dict, \
    cell_id_frame_num_node_idx_best_index_list_dict_dict, \
    cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict, \
    cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict, \
    frame_num_node_idx_cell_occupation_list_list_dict, \
    cell_dependency_dict, \
    max_cell_redo_cnt_record = \
                                _process_and_find_best_cell_track(all_cell_id_track_list_dict,
                                                                  to_handle_cell_id_list,
                                                                  frame_num_prof_matrix_dict,
                                                                  frame_num_node_idx_cell_occupation_list_list_dict,
                                                                  cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                  cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                                  cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                                                  is_use_cell_dependency_feature,
                                                                  cell_dependency_dict,
                                                                  cell_id_frame_num_track_progress_dict,
                                                                  hyper_para.merge_threshold,
                                                                  hyper_para.routing_strategy_enum,
                                                                  hyper_para.cut_strategy_enum,
                                                                  hyper_para.cut_threshold,
                                                                  hyper_para.both_cell_below_threshold_strategy_enum,
                                                                  hyper_para.discount_rate_per_layer,
                                                                  max_cell_redo_cnt_record)
    print("  --> finish")


    second_frame_num: int = 2
    last_frame_num: int = np.max(list(frame_num_prof_matrix_dict.keys())) # should be + 1?


    for frame_num in range(second_frame_num, last_frame_num):
        print(f"frame {frame_num}: ", end='')
        for cell_row_idx in range(frame_num_prof_matrix_dict[frame_num].shape[0]):  #skip all nodes which are already passed

            is_new_cell: bool = len(frame_num_node_idx_cell_occupation_list_list_dict[frame_num][cell_row_idx]) == 0
            if not is_new_cell:
                continue



            cell_id = CellId(frame_num, cell_row_idx)

            to_handle_cell_id_list: list = [cell_id]

            dev_print(f"\n----------------------> {cell_id.str_short()}")

            all_cell_id_track_list_dict, \
            cell_id_frame_num_node_idx_best_index_list_dict_dict, \
            cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict, \
            cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict, \
            frame_num_node_idx_cell_occupation_list_list_dict, \
            cell_dependency_dict, \
            max_cell_redo_cnt_record  = \
                                        _process_and_find_best_cell_track(all_cell_id_track_list_dict,
                                                                          to_handle_cell_id_list,
                                                                          frame_num_prof_matrix_dict,
                                                                          frame_num_node_idx_cell_occupation_list_list_dict,
                                                                          cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                          cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                                          cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                                                          is_use_cell_dependency_feature,
                                                                          cell_dependency_dict,
                                                                          cell_id_frame_num_track_progress_dict,
                                                                          hyper_para.merge_threshold,
                                                                          hyper_para.routing_strategy_enum,
                                                                          hyper_para.cut_strategy_enum,
                                                                          hyper_para.cut_threshold,
                                                                          hyper_para.both_cell_below_threshold_strategy_enum,
                                                                          hyper_para.discount_rate_per_layer,
                                                                          max_cell_redo_cnt_record)

            code_validate_track(all_cell_id_track_list_dict)

        print("  --> finish")

    return all_cell_id_track_list_dict




#loop each node on first frame to find the optimal path using probabilty multiply
def _process_and_find_best_cell_track(cell_id_track_list_dict,
                                      to_handle_cell_id_list: list,
                                      frame_num_prof_matrix_dict: dict,
                                      frame_num_node_idx_cell_occupation_list_list_dict: dict,
                                      cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                      cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                      cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                      is_use_cell_dependency_feature: bool,
                                      cell_dependency_dict: dict,
                                      cell_id_frame_num_track_progress_dict: dict,
                                      merge_above_threshold: float,
                                      routing_strategy_enum: ROUTING_STRATEGY_ENUM,
                                      cut_strategy_enum: CUT_STRATEGY_ENUM,
                                      cut_threshold: float,
                                      both_cell_below_threshold_strategy_enum: BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM,
                                      discount_rate_per_layer,
                                      max_cell_redo_cnt_record):

    to_skip_cell_id_list: list = []
    last_frame_num: int = np.max(list(frame_num_prof_matrix_dict.keys())) + 1

    handling_cell_redo_cnt = 0

    code_validate_track(cell_id_track_list_dict)

    while len(to_handle_cell_id_list) != 0:
        handling_cell_id: CellId = to_handle_cell_id_list[0]

        if is_use_cell_dependency_feature:
            handling_cell_id = find_independent_cell_id_recursive(cell_dependency_dict, handling_cell_id, to_handle_cell_id_list)



        is_use_partial_update_feature: bool = False
        if handling_cell_id in cell_id_track_list_dict:

            if is_use_partial_update_feature:
                last_progress_frame_num = cell_id_frame_num_track_progress_dict[handling_cell_id]
                delete_from_frame_num = last_progress_frame_num + 1

            else:
                delete_from_frame_num = 1

            cell_id_track_list_dict, \
            frame_num_node_idx_cell_occupation_list_list_dict, \
            cell_id_frame_num_node_idx_best_index_list_dict_dict, \
            cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict, \
            cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict = \
                                                                                remove_cell_data_from_specific_frame_num(handling_cell_id, delete_from_frame_num,
                                                                                                                         cell_id_track_list_dict,
                                                                                                                         frame_num_node_idx_cell_occupation_list_list_dict,
                                                                                                                         cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                                                                         cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                                                                                         cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict)



        if len(frame_num_node_idx_cell_occupation_list_list_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx]) == 0:
            is_still_new_cell = True
        elif len(frame_num_node_idx_cell_occupation_list_list_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx]) == 1 and \
                frame_num_node_idx_cell_occupation_list_list_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx][0] == handling_cell_id:
            is_still_new_cell = True
        else:
            is_still_new_cell = False


        if not is_still_new_cell:
            print("\n", handling_cell_id.__str__(), "is no longer a new cell. Occupied by: ", end='')


            for cell_id in frame_num_node_idx_cell_occupation_list_list_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx]:
                print(cell_id.__str__(), end='')
            print()

            if is_use_partial_update_feature:
                cell_id_track_list_dict, \
                frame_num_node_idx_cell_occupation_list_list_dict, \
                cell_id_frame_num_node_idx_best_index_list_dict_dict, \
                cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict, \
                cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict = \
                                                            remove_cell_data_from_specific_frame_num(handling_cell_id, handling_cell_id.start_frame_num,
                                                                                                     cell_id_track_list_dict,
                                                                                                     frame_num_node_idx_cell_occupation_list_list_dict,
                                                                                                     cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                                                     cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                                                                     cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict)

            to_handle_cell_id_list.remove(handling_cell_id)
            time.sleep(1)
            continue


        handling_cell_redo_cnt += 1

        handling_cell_idx: int = handling_cell_id.cell_idx

        start_frame_num: int = handling_cell_id.start_frame_num
        second_frame_num: int = start_frame_num + 1

        if is_use_partial_update_feature:
            if handling_cell_id in cell_id_frame_num_track_progress_dict:
                start_progress_frame_num: int = cell_id_frame_num_track_progress_dict[handling_cell_id]
                if start_progress_frame_num == handling_cell_id.start_frame_num:
                    start_progress_frame_num = second_frame_num
            else:
                start_progress_frame_num: int = second_frame_num
        else:
            start_progress_frame_num: int = second_frame_num


        print(f"{handling_cell_id.start_frame_num}-{handling_cell_idx}[{start_progress_frame_num}]({handling_cell_redo_cnt}|{max_cell_redo_cnt_record}); ", end='')

        for handling_frame_num in range(start_progress_frame_num, last_frame_num):
            if  handling_frame_num == second_frame_num:  last_layer_best_connection_value_list = frame_num_prof_matrix_dict[start_frame_num][handling_cell_idx]
            elif handling_frame_num > second_frame_num:  last_layer_best_connection_value_list = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id][handling_frame_num]

            last_layer_best_connection_value_list = last_layer_best_connection_value_list.reshape(last_layer_best_connection_value_list.shape[0], 1)
            total_cell_in_next_frame: int = frame_num_prof_matrix_dict[handling_frame_num].shape[1]
            last_layer_cell_mtx: np.array = np.repeat(last_layer_best_connection_value_list, total_cell_in_next_frame, axis=1)
            tmp_prof_matrix = frame_num_prof_matrix_dict[handling_frame_num]
            last_layer_all_connection_value_mtx: np.array = last_layer_cell_mtx * tmp_prof_matrix


            one_layer_all_probability_mtx = frame_num_prof_matrix_dict[handling_frame_num]


            # if handling_cell_id in [CellId(7, 3), CellId(7, 4)]  and handling_frame_num == 10:
            #     time.sleep(1)
            #     print("fndfn")

            adjusted_merge_above_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, routing_strategy_enum, handling_frame_num)


            index_ab_vec, one_layer_value_ab_vec, multi_layer_value_ab_vec = derive_last_layer_each_node_best_track(handling_cell_id,
                                                                                                                    one_layer_all_probability_mtx,
                                                                                                                    last_layer_all_connection_value_mtx,
                                                                                                                    frame_num_prof_matrix_dict,
                                                                                                                    handling_frame_num,
                                                                                                                    frame_num_node_idx_cell_occupation_list_list_dict,
                                                                                                                    merge_above_threshold,
                                                                                                                    adjusted_merge_above_threshold,
                                                                                                                    cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                                                                                    cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                                                                                                    routing_strategy_enum,
                                                                                                                    cell_id_track_list_dict,
                                                                                                                    cut_strategy_enum,
                                                                                                                    cut_threshold,
                                                                                                                    both_cell_below_threshold_strategy_enum,
                                                                                                                    discount_rate_per_layer)

            handling_frame_num_for_derive_best_index_from_specific_layer: int = handling_frame_num + 1
            adjusted_merge_above_threshold_for_derive_best_index_from_specific_layer = derive_merge_threshold_in_layer(merge_above_threshold, routing_strategy_enum, handling_frame_num_for_derive_best_index_from_specific_layer)
            current_maximize_index = derive_best_index_from_specific_layer(handling_cell_id,
                                                                           one_layer_value_ab_vec,
                                                                           multi_layer_value_ab_vec,
                                                                           handling_frame_num_for_derive_best_index_from_specific_layer,
                                                                           frame_num_node_idx_cell_occupation_list_list_dict,
                                                                           frame_num_prof_matrix_dict,
                                                                           cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                           cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                                           cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                                                           merge_above_threshold,
                                                                           adjusted_merge_above_threshold_for_derive_best_index_from_specific_layer,
                                                                           routing_strategy_enum,
                                                                           both_cell_below_threshold_strategy_enum,
                                                                           discount_rate_per_layer)

            has_any_available_and_non_zero_node: bool = (current_maximize_index != None)

            # if ( np.all(multi_layer_value_ab_vec == 0) ):
            if not has_any_available_and_non_zero_node:
                # print("sgreb", f"has_any_available_and_non_zero_node == False. No valid node find in layer {handling_frame_num_for_derive_best_index_from_specific_layer}")
                is_zero_track_length = (handling_frame_num == second_frame_num)
                if is_zero_track_length:
                    to_skip_cell_id_list.append(handling_cell_id)

                break

            else:
                next_frame_num: int = handling_frame_num + 1
                cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id][next_frame_num] = index_ab_vec
                cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id][next_frame_num] = one_layer_value_ab_vec
                cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id][next_frame_num] = multi_layer_value_ab_vec


        to_handle_cell_id_list.remove(handling_cell_id)

        to_redo_cell_id_list: list = []
        if handling_cell_id not in to_skip_cell_id_list:
            cell_track_list = derive_final_best_track(cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                      cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                      cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                                      frame_num_prof_matrix_dict,
                                                      frame_num_node_idx_cell_occupation_list_list_dict,
                                                      merge_above_threshold,
                                                      handling_cell_id,
                                                      routing_strategy_enum,
                                                      both_cell_below_threshold_strategy_enum,
                                                      discount_rate_per_layer,
                                                      cell_id_track_list_dict)

            if cut_strategy_enum == CUT_STRATEGY_ENUM.AFTER_ROUTING:
                cell_track_list = _cut_single_track(cell_track_list, cut_threshold, frame_num_prof_matrix_dict)




            # handle redo cell for existing cells
            is_print_newline: bool = True
            for cell_track_tuple in cell_track_list:
                node_idx: int = cell_track_tuple[0]
                frame_num: int = cell_track_tuple[1] + 1

                adjusted_merge_above_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, routing_strategy_enum, frame_num)

                occupied_cell_id_list: tuple = frame_num_node_idx_cell_occupation_list_list_dict[frame_num][node_idx]
                has_cell_occupation: bool = ( len(occupied_cell_id_list) != 0 )

                if has_cell_occupation:

                    second_frame_num: int = handling_cell_id.start_frame_num + 1

                    if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
                        if frame_num == handling_cell_id.start_frame_num: handling_cell_probability: float = adjusted_merge_above_threshold
                        elif frame_num == second_frame_num:               handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_idx][node_idx]
                        elif frame_num > second_frame_num:                handling_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id][frame_num][node_idx]
                        else: raise Exception()
                    elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
                        if frame_num == handling_cell_id.start_frame_num: handling_cell_probability: float = adjusted_merge_above_threshold
                        elif frame_num == second_frame_num:               handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_idx][node_idx]
                        elif frame_num > second_frame_num:                handling_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id][frame_num][node_idx]
                        else: raise Exception(frame_num, second_frame_num)

                        discount_rate = derive_discount_rate_from_cell_start_frame_num(handling_cell_id, merge_above_threshold, discount_rate_per_layer)
                        handling_cell_probability *= discount_rate
                    else: raise Exception(routing_strategy_enum)

                    for occupied_cell_id in occupied_cell_id_list:
                        if occupied_cell_id in to_redo_cell_id_list:
                            continue

                        occupied_cell_second_frame: int = occupied_cell_id.start_frame_num + 1
                        occupied_cell_idx: int = occupied_cell_id.cell_idx

                        if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
                            if frame_num == occupied_cell_id.start_frame_num: occupied_cell_probability: float = adjusted_merge_above_threshold
                            elif frame_num == occupied_cell_second_frame:     occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][node_idx]
                            elif frame_num > occupied_cell_second_frame:    occupied_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[occupied_cell_id][frame_num][node_idx]
                            else: raise Exception(frame_num, occupied_cell_second_frame)
                        elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
                            if frame_num == occupied_cell_id.start_frame_num: occupied_cell_probability: float = adjusted_merge_above_threshold
                            elif frame_num == occupied_cell_second_frame:     occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][node_idx]
                            elif frame_num > occupied_cell_second_frame:    occupied_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[occupied_cell_id][frame_num][node_idx]
                            else: raise Exception(frame_num, occupied_cell_second_frame)

                            discount_rate = derive_discount_rate_from_cell_start_frame_num(occupied_cell_id, merge_above_threshold, discount_rate_per_layer)
                            occupied_cell_probability *= discount_rate
                        else: raise Exception(routing_strategy_enum)

                        if handling_cell_probability >= adjusted_merge_above_threshold and occupied_cell_probability >= adjusted_merge_above_threshold:
                            pass
                        elif handling_cell_probability < adjusted_merge_above_threshold and occupied_cell_probability >= adjusted_merge_above_threshold:
                            pass
                        elif handling_cell_probability >= adjusted_merge_above_threshold and occupied_cell_probability < adjusted_merge_above_threshold:

                            if is_print_newline:
                                print()
                                is_print_newline = False

                            print(">>>", f"redo occupied_cell: {occupied_cell_id.__str__()};  frame_num: {frame_num}; node_idx: {node_idx}; last_frame_adjusted_threshold: {adjusted_merge_above_threshold}; node_probability_value: {np.round(handling_cell_probability, 20)}; occupied_cell_probability: {np.round(occupied_cell_probability, 20)} ;")

                            if occupied_cell_id not in to_redo_cell_id_list:
                                to_redo_cell_id_list.append(occupied_cell_id)

                            to_redo_cell_id = occupied_cell_id



                            # mark progress
                            if to_redo_cell_id not in cell_id_frame_num_track_progress_dict:
                                cell_id_frame_num_track_progress_dict[to_redo_cell_id] = (frame_num - 1)
                            elif (frame_num - 1) < cell_id_frame_num_track_progress_dict[to_redo_cell_id]:
                                cell_id_frame_num_track_progress_dict[to_redo_cell_id] = (frame_num - 1)


                            if to_redo_cell_id not in to_handle_cell_id_list:       # (if) to be removed
                                to_handle_cell_id_list.append(to_redo_cell_id)


                            if is_use_cell_dependency_feature:
                                if handling_cell_id.start_frame_num == occupied_cell_id.start_frame_num:
                                    dependency_cell_set = set()
                                    retrieve_all_dependency_cell_set_recursive(cell_dependency_dict, handling_cell_id, dependency_cell_set)
                                    if occupied_cell_id in dependency_cell_set:
                                        print("deadlock dependence found, skip", occupied_cell_id.__str__(), handling_cell_id.__str__())
                                    else:
                                        cell_dependency_dict[occupied_cell_id].append(handling_cell_id)


                            # frame_num_node_idx_occupation_tuple_vec_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, occupied_cell_id)

                        elif handling_cell_probability < adjusted_merge_above_threshold and occupied_cell_probability < adjusted_merge_above_threshold:
                            if both_cell_below_threshold_strategy_enum == BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM.SHARE:
                                pass
                            else:
                                raise Exception()

                        else:
                            raise Exception(occupied_cell_probability, merge_above_threshold)




            # remove whole track because new track can have new track for earlier section
            if handling_cell_id in cell_id_track_list_dict:
                frame_num_node_idx_cell_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id, cell_id_track_list_dict[handling_cell_id], remove_from_frame_num=1)
                del cell_id_track_list_dict[handling_cell_id]

            code_validate_if_cellid_not_exist_in_occupation_data(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id)

            if handling_cell_id in cell_id_frame_num_track_progress_dict:
                del cell_id_frame_num_track_progress_dict[handling_cell_id]


            cell_id_track_list_dict[handling_cell_id] = cell_track_list

            frame_num_node_idx_cell_occupation_list_list_dict = add_track_to_cell_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id, cell_track_list)




            # to_redo_cell_id_list: list = list(set(to_redo_cell_id_list))        # remove duplicated values
            # for to_redo_cell_id in to_redo_cell_id_list:
            #     if to_redo_cell_id in existing_cell_idx_track_list_dict:
            #         frame_num_node_idx_cell_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, to_redo_cell_id, existing_cell_idx_track_list_dict[to_redo_cell_id])
            #         del existing_cell_idx_track_list_dict[to_redo_cell_id]
            #     elif to_redo_cell_id in cell_id_track_list_dict:
            #         frame_num_node_idx_cell_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, to_redo_cell_id, cell_id_track_list_dict[to_redo_cell_id])
            #         del cell_id_track_list_dict[to_redo_cell_id]
            #     else:
            #         raise Exception(to_redo_cell_id)
            #
            #     del cell_id_frame_num_node_idx_best_index_list_dict_dict[to_redo_cell_id]
            #     del cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[to_redo_cell_id]
            #     del cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[to_redo_cell_id]
            #
            #     to_handle_cell_id_list.append(to_redo_cell_id)

            # check if new cell is still new cell

            to_handle_cell_id_list.sort(key=cmp_to_key(compare_cell_id))

            # frame_num_node_idx_cell_occupation_list_list_dict = initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_prof_matrix_dict)
            # frame_num_node_idx_cell_occupation_list_list_dict = update_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, cell_id_track_list_dict)
            # frame_num_node_idx_cell_occupation_list_list_dict = update_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, existing_cell_idx_track_list_dict)

    if handling_cell_redo_cnt > max_cell_redo_cnt_record:
        max_cell_redo_cnt_record = handling_cell_redo_cnt

    return cell_id_track_list_dict, \
           cell_id_frame_num_node_idx_best_index_list_dict_dict, \
           cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict, \
           cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict, \
           frame_num_node_idx_cell_occupation_list_list_dict, \
           cell_dependency_dict, \
           max_cell_redo_cnt_record





def __________unit_function_start_label():
    raise Exception("for labeling only")



def update_below_cut_threshold_value_to_zero(frame_num_prof_matrix_dict: dict, cut_threshold: float):
    for frame_num, prof_mtx in frame_num_prof_matrix_dict.items():
        for row_idx in range(prof_mtx.shape[0]):
            for col_idx in range(prof_mtx.shape[1]):
                if frame_num_prof_matrix_dict[frame_num][row_idx][col_idx] <= cut_threshold:
                    frame_num_prof_matrix_dict[frame_num][row_idx][col_idx] = 0

    return frame_num_prof_matrix_dict



def derive_discount_rate_from_cell_start_frame_num(cell_id: CellId, merge_above_threshold: float, discount_rate_per_layer):
    if merge_above_threshold == 0:
        discount_rate: float = 1.0
    elif discount_rate_per_layer == "merge_above_threshold":
        discount_rate: float = pow(merge_above_threshold, cell_id.start_frame_num - 1)
    else:
        discount_rate: float = pow(discount_rate_per_layer, cell_id.start_frame_num - 1)

    return discount_rate




def remove_cell_data_from_specific_frame_num(handling_cell_id, delete_from_frame_num,
                                             cell_id_track_list_dict,
                                             frame_num_node_idx_cell_occupation_list_list_dict,
                                             cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                             cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                             cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict):

    # handle delete
    if handling_cell_id in cell_id_track_list_dict:
        frame_num_node_idx_cell_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id, cell_id_track_list_dict[handling_cell_id], delete_from_frame_num)
        cell_id_track_list_dict[handling_cell_id] = remove_track_from_frame_num(cell_id_track_list_dict[handling_cell_id], delete_from_frame_num)
        if len(cell_id_track_list_dict[handling_cell_id]) == 0:
            del cell_id_track_list_dict[handling_cell_id]


        for tmp_frame_num in list(cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id].keys()):
            if tmp_frame_num >= delete_from_frame_num:
                del cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id][tmp_frame_num]
        # del cell_id_frame_num_node_idx_best_index_list_dict_dict[to_redo_cell_id]

        for tmp_frame_num in list(cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id].keys()):
            if tmp_frame_num >= delete_from_frame_num:
                del cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id][tmp_frame_num]
        # del cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[to_redo_cell_id]

        for tmp_frame_num in list(cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id].keys()):
            if tmp_frame_num >= delete_from_frame_num:
                del cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id][tmp_frame_num]
    # del cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[to_redo_cell_id]

    return  cell_id_track_list_dict, \
            frame_num_node_idx_cell_occupation_list_list_dict, \
            cell_id_frame_num_node_idx_best_index_list_dict_dict, \
            cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict, \
            cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict



def remove_track_from_frame_num(cell_track_tuple_list: list, remove_from_frame_num: int):
    new_track_list: list = []
    for cell_track_tuple in cell_track_tuple_list:
        frame_num: int = cell_track_tuple[1] + 1
        if frame_num < remove_from_frame_num:
            new_track_list.append(cell_track_tuple)
        else:
            return new_track_list


def retrieve_all_dependency_cell_set_recursive(cell_dependency_dict: dict, handling_cell_id: CellId, dependency_set: set):
    if handling_cell_id in cell_dependency_dict:
        dependent_cell_id_list = cell_dependency_dict[handling_cell_id]
        for dependent_cell_id in dependent_cell_id_list:
            dependency_set.add(dependent_cell_id)
            retrieve_all_dependency_cell_set_recursive(cell_dependency_dict, dependent_cell_id, dependency_set)

    return



def find_independent_cell_id_recursive(cell_dependency_dict: dict, handling_cell_id: CellId, to_handle_cell_id_list):
    if handling_cell_id in cell_dependency_dict:
        dependent_cell_id_list = cell_dependency_dict[handling_cell_id]
        for dependent_cell_id in dependent_cell_id_list:
            if dependent_cell_id in to_handle_cell_id_list:
                lower_dependent_cell_id = find_independent_cell_id_recursive(cell_dependency_dict, dependent_cell_id, to_handle_cell_id_list)
                if handling_cell_id.start_frame_num != lower_dependent_cell_id.start_frame_num:
                    raise Exception(handling_cell_id.__str__(), lower_dependent_cell_id.__str__())
                return lower_dependent_cell_id

    return handling_cell_id






def derive_best_index_from_specific_layer(handling_cell_id: CellId,
                                          frame_num_node_idx_best_one_layer_value_vec,
                                          frame_num_node_idx_best_multi_layer_value_vec,
                                          handling_frame_num,
                                          frame_cell_occupation_vec_list_dict,
                                          frame_num_prof_matrix_dict,
                                          cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                          cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                          cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                          merge_above_threshold: float,
                                          last_frame_adjusted_threshold: float,
                                          routing_strategy_enum,
                                          both_cell_below_threshold_strategy_enum,
                                          discount_rate_per_layer
                                          ):

    # if handling_cell_id == CellId(7, 3) and handling_frame_num == 10:
    #     time.sleep(1)
    #     print("sbsfdb")




    current_maximize_index: float = None
    current_maximize_value: float = 0

    for node_idx, node_multi_layer_probability_value in enumerate(frame_num_node_idx_best_multi_layer_value_vec):

        is_new_value_higher: bool = (node_multi_layer_probability_value > current_maximize_value)

        if not is_new_value_higher:
            continue

        occupied_cell_id_list: tuple = frame_cell_occupation_vec_list_dict[handling_frame_num][node_idx]
        has_cell_occupation: bool = ( len(occupied_cell_id_list) != 0 )

        if not has_cell_occupation:
            current_maximize_index = node_idx
            current_maximize_value = node_multi_layer_probability_value

        elif has_cell_occupation:
            handling_cell_sec_frame_num: int = handling_cell_id.start_frame_num + 1
            if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
                # print("abfafb", handling_cell_id, handling_frame_num, node_idx)
                # if handling_frame_num == handling_cell_sec_frame_num:     handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx][node_idx]
                # elif handling_frame_num > handling_cell_sec_frame_num:    handling_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id][handling_frame_num][node_idx]
                # else: raise Exception(handling_frame_num, handling_cell_sec_frame_num)

                handling_cell_probability = frame_num_node_idx_best_one_layer_value_vec[node_idx]

            elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
                # if handling_frame_num == handling_cell_sec_frame_num:     handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx][node_idx]
                # elif handling_frame_num > handling_cell_sec_frame_num:    handling_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id][handling_frame_num][node_idx]
                # else: raise Exception(handling_frame_num, handling_cell_sec_frame_num)
                handling_cell_probability = node_multi_layer_probability_value

                discount_rate = derive_discount_rate_from_cell_start_frame_num(handling_cell_id, merge_above_threshold, discount_rate_per_layer)
                handling_cell_probability *= discount_rate
            else: raise Exception(routing_strategy_enum)


            is_node_available_for_handling_cell: bool = True
            for occupied_cell_id in occupied_cell_id_list:
                occupied_cell_idx = occupied_cell_id.cell_idx

                occupied_cell_second_frame_num: int = occupied_cell_id.start_frame_num + 1
                if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
                    if handling_frame_num == occupied_cell_id.start_frame_num:  occupied_cell_probability: float = last_frame_adjusted_threshold
                    elif handling_frame_num == occupied_cell_second_frame_num:      occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][node_idx]
                    elif handling_frame_num > occupied_cell_second_frame_num:     occupied_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[occupied_cell_id][handling_frame_num][node_idx]
                    else: raise Exception(handling_frame_num, occupied_cell_id.__str__)

                    discount_rate = derive_discount_rate_from_cell_start_frame_num(occupied_cell_id, merge_above_threshold, discount_rate_per_layer)
                    occupied_cell_probability *= discount_rate
                elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
                    if handling_frame_num == occupied_cell_id.start_frame_num:  occupied_cell_probability: float = last_frame_adjusted_threshold
                    elif handling_frame_num == occupied_cell_second_frame_num:  occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][node_idx]
                    elif handling_frame_num > occupied_cell_second_frame_num:   occupied_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[occupied_cell_id][handling_frame_num][node_idx]
                    else: raise Exception(handling_frame_num, occupied_cell_second_frame_num)

                else:
                    raise Exception(routing_strategy_enum)



                if handling_cell_probability >= last_frame_adjusted_threshold and occupied_cell_probability >= last_frame_adjusted_threshold:
                    pass
                elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability >= last_frame_adjusted_threshold:
                    is_node_available_for_handling_cell = False
                    break
                elif handling_cell_probability >= last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    pass
                elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    if both_cell_below_threshold_strategy_enum == BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM.SHARE:
                        pass
                    else:
                        raise Exception()

                else:
                    raise Exception(node_multi_layer_probability_value, occupied_cell_probability, last_frame_adjusted_threshold)

            if is_node_available_for_handling_cell and is_new_value_higher:
                current_maximize_index = node_idx
                current_maximize_value = node_multi_layer_probability_value

    return current_maximize_index





# def derive_has_any_available_and_non_zero_node(handling_cell_id: CellId,
#                                                frame_num_node_idx_best_multi_layer_value_vec,
#                                                handling_frame_num,
#                                                frame_cell_occupation_vec_list_dict,
#                                                frame_num_prof_matrix_dict,
#                                                cell_id_frame_num_node_idx_best_index_list_dict_dict,
#                                                cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
#                                                cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
#                                                merge_above_threshold: float,
#                                                frame_adjusted_threshold: float,
#                                                routing_strategy_enum,
#                                                both_cell_below_threshold_strategy_enum,
#                                                discount_rate_per_layer
#                                                ):
#     derive_has_any_available_and_non_zero_node: bool = True
#
#     current_maximize_index: float = None
#     current_maximize_value: float = 0
#
#     for node_idx, node_multi_layer_probability_value in enumerate(frame_num_node_idx_best_multi_layer_value_vec):
#
#         is_new_value_higher: bool = (node_multi_layer_probability_value > current_maximize_value)
#
#         if not is_new_value_higher:
#             continue
#
#         occupied_cell_id_list: tuple = frame_cell_occupation_vec_list_dict[handling_frame_num][node_idx]
#         has_cell_occupation: bool = ( len(occupied_cell_id_list) != 0 )
#
#         if not has_cell_occupation:
#             current_maximize_index = node_idx
#             current_maximize_value = node_multi_layer_probability_value
#
#         elif has_cell_occupation:
#             handling_cell_sec_frame_num: int = handling_cell_id.start_frame_num + 1
#             if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
#                 if handling_frame_num == handling_cell_sec_frame_num:     handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx][node_idx]
#                 elif handling_frame_num > handling_cell_sec_frame_num:    handling_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id][handling_frame_num][node_idx]
#                 else: raise Exception(handling_frame_num, handling_cell_sec_frame_num)
#             elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
#                 if handling_frame_num == handling_cell_sec_frame_num:     handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx][node_idx]
#                 elif handling_frame_num > handling_cell_sec_frame_num:    handling_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id][handling_frame_num][node_idx]
#                 else: raise Exception(handling_frame_num, handling_cell_sec_frame_num)
#
#                 discount_rate = derive_discount_rate_from_cell_start_frame_num(handling_cell_id, merge_above_threshold, discount_rate_per_layer)
#                 handling_cell_probability *= discount_rate
#             else: raise Exception(routing_strategy_enum)
#
#
#             is_node_available_for_handling_cell: bool = True
#             for occupied_cell_id in occupied_cell_id_list:
#                 occupied_cell_idx = occupied_cell_id.cell_idx
#
#                 occupied_cell_second_frame_num: int = occupied_cell_id.start_frame_num + 1
#                 if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
#                     if handling_frame_num == occupied_cell_id.start_frame_num:  occupied_cell_probability: float = frame_adjusted_threshold
#                     elif handling_frame_num == occupied_cell_second_frame_num:      occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][node_idx]
#                     elif handling_frame_num > occupied_cell_second_frame_num:     occupied_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[occupied_cell_id][handling_frame_num][node_idx]
#                     else: raise Exception(handling_frame_num, occupied_cell_id.__str__)
#
#                     discount_rate = derive_discount_rate_from_cell_start_frame_num(occupied_cell_id, merge_above_threshold, discount_rate_per_layer)
#                     occupied_cell_probability *= discount_rate
#                 elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
#                     if handling_frame_num == occupied_cell_id.start_frame_num:  occupied_cell_probability: float = frame_adjusted_threshold
#                     elif handling_frame_num == occupied_cell_second_frame_num:  occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][node_idx]
#                     elif handling_frame_num > occupied_cell_second_frame_num:   occupied_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[occupied_cell_id][handling_frame_num][node_idx]
#                     else: raise Exception(handling_frame_num, occupied_cell_second_frame_num)
#
#                 else:
#                     raise Exception(routing_strategy_enum)
#
#
#
#                 if handling_cell_probability >= frame_adjusted_threshold and occupied_cell_probability >= frame_adjusted_threshold:
#                     pass
#                 elif handling_cell_probability < frame_adjusted_threshold and occupied_cell_probability >= frame_adjusted_threshold:
#                     is_node_available_for_handling_cell = False
#                     break
#                 elif handling_cell_probability >= frame_adjusted_threshold and occupied_cell_probability < frame_adjusted_threshold:
#                     pass
#                 elif handling_cell_probability < frame_adjusted_threshold and occupied_cell_probability < frame_adjusted_threshold:
#                     if both_cell_below_threshold_strategy_enum == BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM.SHARE:
#                         pass
#                     else:
#                         raise Exception()
#
#                 else:
#                     raise Exception(node_multi_layer_probability_value, occupied_cell_probability, frame_adjusted_threshold)
#
#             if is_node_available_for_handling_cell and is_new_value_higher:
#                 current_maximize_index = node_idx
#                 current_maximize_value = node_multi_layer_probability_value
#
#     return current_maximize_index


def _cut_single_track(track_tuple_list: list, cut_threshold: float, frame_num_prof_matrix_dict: dict):
    if len(track_tuple_list) == 1:
        return track_tuple_list

    second_last_frame_idx: int = (len(track_tuple_list) - 2)
    for idx in range(len(track_tuple_list) - 1):
        frame_idx = track_tuple_list[idx][1]
        frame_num: int = frame_idx + 1
        current_frame_node_idx = track_tuple_list[idx][0]
        next_frame_node_idx = track_tuple_list[idx + 1][0]

        connection_probability: float = frame_num_prof_matrix_dict[frame_num][current_frame_node_idx][next_frame_node_idx]

        if connection_probability <= cut_threshold:
            return track_tuple_list[0: idx + 1]

        is_reached_second_last_idx: bool = (idx == second_last_frame_idx)
        if is_reached_second_last_idx:
            return track_tuple_list

    raise Exception()



def filter_track_dict_by_length(all_track_dict: dict, minimum_track_length: int):
    filtered_track_dict: dict = {}

    for cell_id, track_list in list(all_track_dict.items()):
        if len(track_list) > minimum_track_length:
            filtered_track_dict[cell_id] = track_list

    return filtered_track_dict



def derive_merge_threshold_in_layer(merge_above_threshold:float, routing_strategy_enum: ROUTING_STRATEGY_ENUM, frame_num:int):
    if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
        if frame_num > 1: threshold_exponential: float = float(frame_num - 1)
        elif frame_num == 1: threshold_exponential = 1                          # avoid 0.5^0 becomes 1
        else: raise Exception()

        merge_threshold_in_layer: float = pow(merge_above_threshold, threshold_exponential)

        return merge_threshold_in_layer

    elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
        return merge_above_threshold

    else:
        raise Exception(routing_strategy_enum)



def filter_track_list_by_length(track_list_list: list, min_track_length: int):
    result_track_list: list = []
    for track_list in track_list_list:
        if (len(track_list) > min_track_length):
            result_track_list.append(track_list)

    return result_track_list



def post_adjustment_old(result: list, prof_mat_list: list):

    for j in range(len(result)-1):
        for k in range(j + 1, len(result)):
            pre_track = result[j]
            next_track = result[k]
            overlap_track = sorted(set([i[0:2] for i in pre_track]) & set([i[0:2] for i in next_track]), key = lambda x : (x[1], x[0]))
            if overlap_track == []:
                continue
            overlap_frame1 =  overlap_track[0][1]
            node_combine = overlap_track[0][0]
            pre_frame = overlap_frame1 - 1
            for i, tuples in enumerate(pre_track):
                if tuples[1]==pre_frame:
                    index_merge1 = i
                    break
                else:
                    continue
            node_merge1 = pre_track[index_merge1][0]
            for ii, tuples in enumerate(next_track):
                if tuples[1]==pre_frame:
                    index_merge2 = ii
                    break
                else:
                    continue
            node_merge2 = next_track[index_merge2][0]
            sub_matrix = prof_mat_list[pre_frame]
            threSh1 = sub_matrix[node_merge1][node_combine]
            threSh2 = sub_matrix[node_merge2][node_combine]
            if threSh1 < threSh2:
                result[k] = next_track
                pre_track_new = copy.deepcopy(pre_track[0:index_merge1 + 1])
                result[j] = pre_track_new
            else:
                result[j] = pre_track
                next_track_new = copy.deepcopy(next_track[0:index_merge2 + 1])
                result[k] = next_track_new

    #print(result)
    final_result = []
    for i in range(len(result)):
        if (len(result[i])>5):
            final_result.append(result[i])

    return final_result



def post_adjustment(track_list_list: list, prof_mat_list: list):
    for pre_track_j in range(len(track_list_list) - 1):
        for next_track_k in range(pre_track_j + 1, len(track_list_list)):
            pre_track_list = track_list_list[pre_track_j]

            next_track_list = track_list_list[next_track_k]

            overlap_track_list = sorted(set([i[0:2] for i in pre_track_list]) & set([i[0:2] for i in next_track_list]), key = lambda x : (x[1], x[0]))

            if overlap_track_list == []:
                continue

            overlap_frame1_list = overlap_track_list[0][1]
            node_combine_list = overlap_track_list[0][0]
            pre_frame = overlap_frame1_list - 1
            for i, tuples in enumerate(pre_track_list):
                if tuples[1] == pre_frame:
                    index_merge1 = i
                    break
                else:
                    continue

            node_merge1 = pre_track_list[index_merge1][0]
            for ii, tuples in enumerate(next_track_list):
                if tuples[1] == pre_frame:
                    index_merge2 = ii
                    break
                else:
                    continue

            node_merge2 = next_track_list[index_merge2][0]
            sub_matrix = prof_mat_list[pre_frame]
            thre_sh1 = sub_matrix[node_merge1][node_combine_list]
            thre_sh2 = sub_matrix[node_merge2][node_combine_list]
            if thre_sh1 < thre_sh2:
                track_list_list[next_track_k] = next_track_list
                pre_track_new = copy.deepcopy(pre_track_list[0: index_merge1 + 1])
                track_list_list[pre_track_j] = pre_track_new
            else:
                track_list_list[pre_track_j] = pre_track_list
                next_track_new = copy.deepcopy(next_track_list[0: index_merge2 + 1])
                track_list_list[next_track_k] = next_track_new

    final_result_list = filter_track_list_by_length(track_list_list)

    return final_result_list



def derive_final_best_track(cell_id_frame_num_node_idx_best_index_list_dict_dict: dict,
                            cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict: dict,
                            cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict: dict,
                            frame_num_prof_matrix_dict: dict,
                            frame_num_node_idx_cell_occupation_list_list_dict: dict,
                            merge_above_threshold: float,
                            handling_cell_id,
                            routing_strategy_enum,
                            both_cell_below_threshold_strategy_enum: BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM,
                            discount_rate_per_layer,
                            cell_track_list_dict):


    frame_num_node_idx_best_index_list_dict: dict = cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id]
    frame_num_node_idx_best_one_layer_value_list_dict: dict = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id]
    frame_num_node_idx_best_multi_layer_value_list_dict: dict = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id]

    handling_cell_idx: int = handling_cell_id.cell_idx
    cell_track_list: list = []

    last_frame_num: int = np.max(list(frame_num_node_idx_best_multi_layer_value_list_dict.keys()))
    second_frame_num: int = handling_cell_id.start_frame_num + 1

    if last_frame_num == handling_cell_id.start_frame_num:
        raise Exception(handling_cell_id.__str__())



    frame_num_node_idx_best_one_layer_value_vec: list = frame_num_node_idx_best_one_layer_value_list_dict[last_frame_num]
    frame_num_node_idx_best_multi_layer_value_vec: list = frame_num_node_idx_best_multi_layer_value_list_dict[last_frame_num]


    # must be ROUTING_STRATEGY_ENUM.ALL_LAYER in all case because ONE_LAYER is compared with multi_layer_probability score
    # if handling_cell_id == CellId(7, 3):
    #     time.sleep(1)
    #     print("sbsfbsdbsdfsddb")

    last_frame_adjusted_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, ROUTING_STRATEGY_ENUM.ALL_LAYER, last_frame_num)

    current_maximize_index = derive_best_index_from_specific_layer(handling_cell_id,
                                                                   frame_num_node_idx_best_one_layer_value_vec,
                                                                   frame_num_node_idx_best_multi_layer_value_vec,
                                                                   last_frame_num,
                                                                   frame_num_node_idx_cell_occupation_list_list_dict,
                                                                   frame_num_prof_matrix_dict,
                                                                   cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                   cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
                                                                   cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
                                                                   merge_above_threshold,
                                                                   last_frame_adjusted_threshold,
                                                                   routing_strategy_enum,
                                                                   both_cell_below_threshold_strategy_enum,
                                                                   discount_rate_per_layer)


    is_all_nodes_invalid: bool = (current_maximize_index == None)
    if is_all_nodes_invalid and routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
        print("no valid nodes found in the last layers with ONE_LAYER strategy, return self node")
        start_frame_idx: int = handling_cell_id.start_frame_num - 1
        cell_track_list.append((handling_cell_idx, start_frame_idx, -1))

        return cell_track_list


    if is_all_nodes_invalid:
        print(handling_cell_id.__str__())
        print(last_frame_num)
        raise Exception("is_all_nodes_invalid")






    # is_all_nodes_invalid: bool = (current_maximize_index == None)
    # # this could happen because last layer is not checked with occupation (but last layer -1 is checked). Therefore, it happens if:
    # # 1) probability value is lower than threshold
    # # 2) all node is occupied by another cell which has a value higher than threshold
    # if is_all_nodes_invalid and last_frame_num == second_frame_num:
    #     frame_idx: int = handling_cell_id.start_frame_num - 1
    #     cell_track_list.append((handling_cell_id.cell_idx, frame_idx, -1))
    #     return cell_track_list#, list(to_redo_cell_id_set)
    #
    # elif is_all_nodes_invalid and last_frame_num > second_frame_num:
    #     print(f"\n'is_all_nodes_invalid == True' detected in layer {last_frame_num}, move one layer backward.")
    #
    #     # # find the last -1 layer, since as it is checked in previous process it must be valid
    #     # last_layer_max_probability_idx: int = np.argmax(frame_num_node_idx_best_multi_layer_value_vec)
    #     # second_last_layer_max_probability_idx: int = frame_num_node_idx_best_index_list_dict[last_frame_num][last_layer_max_probability_idx]
    #     #
    #     # last_frame_num -= 1
    #     # current_maximize_index = second_last_layer_max_probability_idx
    #
    #     last_frame_num -= 1
    #
    #     handling_cell_second_frame_num: int = handling_cell_id.start_frame_num + 1
    #     if last_frame_num == handling_cell_second_frame_num:
    #         frame_num_node_idx_best_multi_layer_value_vec = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_id.cell_idx]
    #     elif last_frame_num > handling_cell_second_frame_num:
    #         frame_num_node_idx_best_multi_layer_value_vec: list = frame_num_node_idx_best_multi_layer_value_list_dict[last_frame_num]
    #     else:
    #         raise Exception()
    #
    #     # must be ROUTING_STRATEGY_ENUM.ALL_LAYER in all case because ONE_LAYER is compared with multi_layer_probability score
    #     last_frame_adjusted_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, ROUTING_STRATEGY_ENUM.ALL_LAYER, last_frame_num)
    #
    #     current_maximize_index = derive_best_index_from_specific_layer(handling_cell_id,
    #                                                                    frame_num_node_idx_best_multi_layer_value_vec,
    #                                                                    last_frame_num,
    #                                                                    frame_num_node_idx_cell_occupation_list_list_dict,
    #                                                                    frame_num_prof_matrix_dict,
    #                                                                    cell_id_frame_num_node_idx_best_index_list_dict_dict,
    #                                                                    cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict,
    #                                                                    cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict,
    #                                                                    merge_above_threshold,
    #                                                                    last_frame_adjusted_threshold,
    #                                                                    routing_strategy_enum,
    #                                                                    both_cell_below_threshold_strategy_enum,
    #                                                                    discount_rate_per_layer)
    #
    #
    # # debug check
    # is_all_nodes_invalid_in_second_last_layer: bool = (current_maximize_index == None)
    #
    # if is_all_nodes_invalid_in_second_last_layer and routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
    #     print("no valid nodes found in the last two layers with ONE_LAYER strategy, return self node")
    #     start_frame_idx: int = handling_cell_id.start_frame_num - 1
    #     cell_track_list.append((handling_cell_idx, start_frame_idx, -1))
    #
    #     return cell_track_list
    #
    #
    # if is_all_nodes_invalid_in_second_last_layer:
    #     print(handling_cell_id.__str__())
    #     print(last_frame_num)
    #     raise Exception("is_all_nodes_invalid_in_second_last_layer")




    if last_frame_num == handling_cell_id.start_frame_num:
        start_frame_idx: int = handling_cell_id.start_frame_num - 1
        cell_track_list.append((handling_cell_idx, start_frame_idx, -1))
        return cell_track_list



    handling_cell_second_frame_num: int = handling_cell_id.start_frame_num + 1
    if last_frame_num == handling_cell_second_frame_num:
        previous_maximize_index: int = current_maximize_index

    elif last_frame_num > handling_cell_second_frame_num:
        previous_maximize_index: int = frame_num_node_idx_best_index_list_dict[last_frame_num][current_maximize_index]

        last_frame_idx: int = last_frame_num - 1
        cell_track_list.append((current_maximize_index, last_frame_idx, previous_maximize_index))

    else:
        raise Exception()




    for reversed_frame_num in range(last_frame_num-1, second_frame_num, -1): #119 to 3 for total_frames = 120

        reversed_frame_idx = reversed_frame_num - 1

        current_maximize_index = previous_maximize_index
        previous_maximize_index = frame_num_node_idx_best_index_list_dict[reversed_frame_num][current_maximize_index]

        cell_track_list.append((current_maximize_index, reversed_frame_idx, previous_maximize_index))

        # last_frame_adjusted_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, routing_strategy_enum, reversed_frame_num)

        # occupied_cell_id_list: tuple = frame_num_node_idx_cell_occupation_list_list_dict[reversed_frame_num][current_maximize_index]
        # has_cell_occupation: bool = ( len(occupied_cell_id_list) != 0 )
        #
        # if has_cell_occupation:
        #     second_frame_num: int = handling_cell_id.start_frame_num + 1
        #
        #     if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
        #         if reversed_frame_num == handling_cell_id.start_frame_num:     handling_cell_probability: float = last_frame_adjusted_threshold
        #         elif reversed_frame_num == second_frame_num:                    handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_idx][current_maximize_index]
        #         elif reversed_frame_num > second_frame_num:                     handling_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[handling_cell_id][reversed_frame_num][current_maximize_index]
        #         else: Exception(reversed_frame_num, second_frame_num)
        #     elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
        #         if reversed_frame_num == handling_cell_id.start_frame_num:     handling_cell_probability: float = last_frame_adjusted_threshold
        #         elif reversed_frame_num == second_frame_num:                    handling_cell_probability: float = frame_num_prof_matrix_dict[handling_cell_id.start_frame_num][handling_cell_idx][current_maximize_index]
        #         elif reversed_frame_num > second_frame_num:                     handling_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[handling_cell_id][reversed_frame_num][current_maximize_index]
        #         else: raise Exception(reversed_frame_num, second_frame_num)
        #     else: raise Exception(routing_strategy_enum)
        #
        #     for occupied_cell_id in occupied_cell_id_list:
        #         occupied_cell_second_frame: int = occupied_cell_id.start_frame_num + 1
        #         occupied_cell_idx: int = occupied_cell_id.cell_idx
        #
        #         if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
        #             if reversed_frame_num == occupied_cell_id.start_frame_num: occupied_cell_probability: float = last_frame_adjusted_threshold
        #             elif reversed_frame_num == occupied_cell_second_frame:     occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][current_maximize_index]
        #             elif reversed_frame_num > occupied_cell_second_frame:    occupied_cell_probability: float = cell_id_frame_num_node_idx_best_one_layer_value_list_dict_dict[occupied_cell_id][reversed_frame_num][current_maximize_index]
        #             else: raise Exception(reversed_frame_num, occupied_cell_second_frame)
        #         elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
        #             if reversed_frame_num == occupied_cell_id.start_frame_num: occupied_cell_probability: float = last_frame_adjusted_threshold
        #             elif reversed_frame_num == occupied_cell_second_frame:     occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][current_maximize_index]
        #             elif reversed_frame_num > occupied_cell_second_frame:    occupied_cell_probability: float = cell_id_frame_num_node_idx_best_multi_layer_value_list_dict_dict[occupied_cell_id][reversed_frame_num][current_maximize_index]
        #             else: raise Exception(reversed_frame_num, occupied_cell_second_frame)
        #         else: raise Exception(routing_strategy_enum)
        #
        #         if handling_cell_probability >= last_frame_adjusted_threshold and occupied_cell_probability >= last_frame_adjusted_threshold:
        #             pass
        #         elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability >= last_frame_adjusted_threshold:
        #             pass
        #         elif handling_cell_probability >= last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
        #             # print(">>>", f"redo occupied_cell: {occupied_cell_id.__str__()};  last_frame_num: {last_frame_num}; current_maximize_index: {current_maximize_index}; last_frame_adjusted_threshold: {last_frame_adjusted_threshold}; node_probability_value: {np.round(handling_cell_probability, 20)}; occupied_cell_probability: {np.round(occupied_cell_probability, 20)} ;")
        #             pass
        #
        #         elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
        #             if both_cell_below_threshold_strategy_enum == BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM.SHARE:
        #                 pass
        #             else:
        #                 raise Exception()
        #
        #         else:
        #             raise Exception(occupied_cell_probability, merge_above_threshold)


    start_frame_idx: int = handling_cell_id.start_frame_num - 1
    cell_track_list.append((previous_maximize_index, start_frame_idx + 1, handling_cell_idx))
    cell_track_list.append((handling_cell_idx, start_frame_idx, -1))

    list.reverse(cell_track_list)

    return cell_track_list



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



def derive_matrix_value_by_index_list(last_layer_all_probability_mtx: np.array, index_value_list: list, axis=0):
    if axis == 0:
        num_of_col: int = last_layer_all_probability_mtx.shape[1]
        is_valid: bool = (num_of_col == len(index_value_list))
        if not is_valid:
            raise Exception("num_of_col != len(index_value_list)")

        result_list: list = []
        for idx, index_value in enumerate(index_value_list):
            result_list.append(last_layer_all_probability_mtx[index_value][idx])

        return np.array(result_list)

    raise Exception(axis)



def deprecated_derive_frame_num_node_id_occupation_tuple_list_dict(frame_num_prof_matrix_dict: dict, track_tuple_list_dict: dict):
    frame_num_node_idx_occupation_tuple_list_dict: dict = {}

    # initiate frame_num_node_idx_occupation_tuple_list_dict
    for frame_num, profit_matrix in frame_num_prof_matrix_dict.items():
        next_frame_num: int = frame_num + 1
        total_cell: int = profit_matrix.shape[1]
        frame_num_node_idx_occupation_tuple_list_dict[next_frame_num] = [()] * total_cell


    # assign True to cell that is occupied
    for cell_idx, track_tuple_list in track_tuple_list_dict.items():
        for track_idx, track_tuple in enumerate(track_tuple_list):
            frame_num: int = track_tuple[1] + 1

            if frame_num == 1:
                continue

            occupied_node_idx: int = track_tuple[0]

            frame_num_node_idx_occupation_tuple_list_dict[frame_num][occupied_node_idx] += (cell_idx,)

    return frame_num_node_idx_occupation_tuple_list_dict



def initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_prof_matrix_dict: dict):
    frame_num_node_idx_occupation_tuple_vec_dict: dict = {}

    # initiate slots from frame 1 to (last_frame-1)
    for frame_num, profit_matrix in frame_num_prof_matrix_dict.items():
        total_cell: int = profit_matrix.shape[0]
        frame_num_node_idx_occupation_tuple_vec_dict[frame_num] = [[] for _ in range(total_cell)]

    # initiate slots for last_frame
    last_frame_num: int = np.max(list(frame_num_prof_matrix_dict.keys())) + 1
    second_last_frame_num: int = last_frame_num - 1
    total_cell_last_frame: int = frame_num_prof_matrix_dict[second_last_frame_num].shape[1]
    frame_num_node_idx_occupation_tuple_vec_dict[last_frame_num] = [[] for _ in range(total_cell_last_frame)]

    return frame_num_node_idx_occupation_tuple_vec_dict



def add_track_to_cell_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, track_tuple_list: list):
    for track_tuple in track_tuple_list:
        frame_num: int = track_tuple[1] + 1
        occupied_node_idx: int = track_tuple[0]

        if cell_id not in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][occupied_node_idx]:
            frame_num_node_idx_occupation_tuple_vec_dict[frame_num][occupied_node_idx].append(cell_id)

    return frame_num_node_idx_occupation_tuple_vec_dict



def remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, track_tuple_list: list, remove_from_frame_num: int):
    for track_tuple in track_tuple_list:
        frame_num: int = track_tuple[1] + 1
        occupied_node_idx: int = track_tuple[0]

        if frame_num < remove_from_frame_num:
            continue

        frame_num_node_idx_occupation_tuple_vec_dict[frame_num][occupied_node_idx].remove(cell_id)

    return frame_num_node_idx_occupation_tuple_vec_dict



def update_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id_track_tuple_list_dict: dict):
    for occupied_cell_id, track_tuple_list in cell_id_track_tuple_list_dict.items():

        for track_tuple in track_tuple_list:
            frame_num: int = track_tuple[1] + 1

            occupied_node_idx: int = track_tuple[0]

            if frame_num < occupied_cell_id.start_frame_num:
                time.sleep(2)
                print(occupied_cell_id)
                print(track_tuple_list)
                raise Exception("occupied_cell_id")

            frame_num_node_idx_occupation_tuple_vec_dict[frame_num][occupied_node_idx].append(occupied_cell_id)



    is_validate_code = False
    if is_validate_code:
        for frame_num, node_idx_occupation_tuple in frame_num_node_idx_occupation_tuple_vec_dict.items():
            for node_idx_occupation_list in node_idx_occupation_tuple:
                for cell_id_occupation in node_idx_occupation_list:
                    if frame_num <= cell_id_occupation.start_frame_num:
                        time.sleep(2)

                        print("bsvh", frame_num, cell_id_occupation.start_frame_num)
                        raise Exception("frame_num >= occupation_cell_start_frame_num")



    return frame_num_node_idx_occupation_tuple_vec_dict




def deprecate_derive_prof_matrix_list(segmentation_folder_path: str, output_folder_path: str, series: str, segmented_filename_list):
    prof_mat_list = []

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

        prof_mat_list.append(prof_mat)

        #make next frame current frame
        cellnb_img = cellnb_img_next
        label_img = label_img_next

    return prof_mat_list



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



def derive_last_layer_each_node_best_track(handling_cell_id,  # CellId
                                           one_layer_all_probability_mtx: np.array,
                                           multi_layer_all_probability_mtx: np.array,
                                           frame_num_prof_matrix_dict: dict,
                                           handling_frame_num: int,
                                           frame_num_node_idx_cell_id_occupation_list_list_dict: dict,
                                           merge_above_threshold: float,
                                           adjusted_merge_above_threshold: float,
                                           cell_id_frame_num_node_idx_one_layer_best_value_list_dict_dict: dict,
                                           cell_id_frame_num_node_idx_multi_layer_best_value_list_dict_dict: dict,
                                           routing_strategy_enum: ROUTING_STRATEGY_ENUM,
                                           cell_id_track_list_dict,
                                           cut_strategy_enum: CUT_STRATEGY_ENUM,
                                           cut_threshold: float,
                                           both_cell_below_threshold_strategy_enum: BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM,
                                           discount_rate_per_layer):

    handling_cell_idx: int = handling_cell_id.cell_idx
    start_frame_num: int = handling_cell_id.start_frame_num



    total_node_next_frame: int = multi_layer_all_probability_mtx.shape[1]
    index_ab_vec: list = [None] * total_node_next_frame
    one_layer_value_ab_vec: list = [None] * total_node_next_frame
    multi_layer_value_ab_vec: list = [None] * total_node_next_frame
    second_frame_num: int = start_frame_num + 1

    for next_frame_node_idx in range(total_node_next_frame):
        best_idx: int = 0
        one_layer_best_score: float = 0
        multi_layer_best_score: float = 0
        generic_best_score: float = 0

        one_layer_node_connection_score_list = one_layer_all_probability_mtx[:, next_frame_node_idx]
        multi_layer_node_connection_score_list = multi_layer_all_probability_mtx[:, next_frame_node_idx]

        if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
            generic_node_connection_score_list = one_layer_node_connection_score_list
        elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
            generic_node_connection_score_list = multi_layer_node_connection_score_list
        else: raise Exception(routing_strategy_enum)


        for node_idx, generic_node_connection_score in enumerate(generic_node_connection_score_list):


            if cut_strategy_enum == CUT_STRATEGY_ENUM.DURING_ROUTING:
                if one_layer_node_connection_score_list[node_idx] <= cut_threshold:
                    continue


            is_new_connection_score_higher: bool = (generic_node_connection_score > generic_best_score)
            if not is_new_connection_score_higher:
                continue

            occupied_cell_id_list: list = frame_num_node_idx_cell_id_occupation_list_list_dict[handling_frame_num][node_idx]
            has_cell_occupation: bool = (len(occupied_cell_id_list) > 0)


            if (not has_cell_occupation) and is_new_connection_score_higher:
                best_idx = node_idx
                one_layer_best_score = one_layer_node_connection_score_list[best_idx]
                multi_layer_best_score = multi_layer_node_connection_score_list[best_idx]
                generic_best_score = generic_node_connection_score

            elif has_cell_occupation and is_new_connection_score_higher:


                is_node_idx_available_for_handling_cell: bool = True

                if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
                    if handling_frame_num == second_frame_num:     handling_cell_probability: float = frame_num_prof_matrix_dict[start_frame_num][handling_cell_idx][node_idx]
                    elif handling_frame_num > second_frame_num:    handling_cell_probability: float = cell_id_frame_num_node_idx_one_layer_best_value_list_dict_dict[handling_cell_id][handling_frame_num][node_idx]
                    else: raise Exception(handling_frame_num, second_frame_num)
                elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
                    if handling_frame_num == second_frame_num:     handling_cell_probability: float = frame_num_prof_matrix_dict[start_frame_num][handling_cell_idx][node_idx]
                    elif handling_frame_num > second_frame_num:    handling_cell_probability: float = cell_id_frame_num_node_idx_multi_layer_best_value_list_dict_dict[handling_cell_id][handling_frame_num][node_idx]
                    else: raise Exception(handling_frame_num, second_frame_num)

                    discount_rate = derive_discount_rate_from_cell_start_frame_num(handling_cell_id, merge_above_threshold, discount_rate_per_layer)
                    handling_cell_probability *= discount_rate
                else: raise Exception(routing_strategy_enum)

                for occupied_cell_id in occupied_cell_id_list:


                    occupied_cell_start_frame_num: int = occupied_cell_id.start_frame_num
                    occupied_cell_idx: int = occupied_cell_id.cell_idx
                    occupied_cell_second_frame: int = occupied_cell_id.start_frame_num + 1
                    if routing_strategy_enum == ROUTING_STRATEGY_ENUM.ONE_LAYER:
                        if handling_frame_num == occupied_cell_start_frame_num:  occupied_cell_probability_1: float = adjusted_merge_above_threshold
                        elif handling_frame_num == occupied_cell_second_frame:   occupied_cell_probability_1: float = frame_num_prof_matrix_dict[occupied_cell_start_frame_num][occupied_cell_idx][node_idx]
                        elif handling_frame_num > occupied_cell_second_frame:    occupied_cell_probability_1: float = cell_id_frame_num_node_idx_one_layer_best_value_list_dict_dict[occupied_cell_id][handling_frame_num][node_idx]
                        else: raise Exception(occupied_cell_id.__str__(), handling_frame_num, node_idx, occupied_cell_second_frame)
                    elif routing_strategy_enum == ROUTING_STRATEGY_ENUM.ALL_LAYER:
                        # # dev_print("srtnb", occupied_cell_id.__str__(), node_idx)


                        if handling_frame_num == occupied_cell_start_frame_num:  occupied_cell_probability_1: float = adjusted_merge_above_threshold
                        elif handling_frame_num == occupied_cell_second_frame:   occupied_cell_probability_1: float = frame_num_prof_matrix_dict[occupied_cell_start_frame_num][occupied_cell_idx][node_idx]
                        elif handling_frame_num > occupied_cell_second_frame:    occupied_cell_probability_1: float = cell_id_frame_num_node_idx_multi_layer_best_value_list_dict_dict[occupied_cell_id][handling_frame_num][node_idx]
                        else: raise Exception(occupied_cell_id.__str__(), handling_frame_num, node_idx, occupied_cell_second_frame)

                        discount_rate = derive_discount_rate_from_cell_start_frame_num(occupied_cell_id, merge_above_threshold, discount_rate_per_layer)
                        occupied_cell_probability_1 *= discount_rate
                    else: raise Exception(routing_strategy_enum)



                    code_validate_cell_probability(generic_node_connection_score, occupied_cell_probability_1)

                    if handling_cell_probability >= adjusted_merge_above_threshold and occupied_cell_probability_1 >= adjusted_merge_above_threshold:
                        pass

                    elif handling_cell_probability < adjusted_merge_above_threshold and occupied_cell_probability_1 >= adjusted_merge_above_threshold:
                        is_node_idx_available_for_handling_cell = False
                        break

                    elif handling_cell_probability >= adjusted_merge_above_threshold and occupied_cell_probability_1 < adjusted_merge_above_threshold:
                        pass

                    elif handling_cell_probability < adjusted_merge_above_threshold and occupied_cell_probability_1 < adjusted_merge_above_threshold:
                        if both_cell_below_threshold_strategy_enum == BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM.SHARE:
                            pass

                        else:
                            raise Exception(both_cell_below_threshold_strategy_enum)

                    else:
                        print("sdgberb")
                        print("handling_cell_probability, occupied_cell_probability_1, merge_above_threshold: ", handling_cell_probability, occupied_cell_probability_1, adjusted_merge_above_threshold)
                        raise Exception("else")

                if is_node_idx_available_for_handling_cell:
                    best_idx = node_idx
                    one_layer_best_score = one_layer_node_connection_score_list[best_idx]
                    multi_layer_best_score = multi_layer_node_connection_score_list[best_idx]
                    generic_best_score = generic_node_connection_score


        index_ab_vec[next_frame_node_idx] = best_idx
        one_layer_value_ab_vec[next_frame_node_idx] = one_layer_best_score
        multi_layer_value_ab_vec[next_frame_node_idx] = multi_layer_best_score

    return index_ab_vec, np.array(one_layer_value_ab_vec), np.array(multi_layer_value_ab_vec)


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
        if cell_id.start_frame_num < max_frame_num:
            raise Exception("cell_id.start_frame_num < max_frame_num")
        if cell_id.start_frame_num == max_frame_num and cell_id.cell_idx < max_cell_idx:
            raise Exception("cell_id.start_frame_num = max_frame_num and cell_id.cell_idx < max_cell_idx")

        max_frame_num = cell_id.start_frame_num
        max_cell_idx = cell_id.cell_idx


if __name__ == '__main__':
    main()