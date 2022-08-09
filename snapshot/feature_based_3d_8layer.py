# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:31:51 2021

@author: 13784
"""
import itertools
import json
import math
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
from collections import defaultdict, namedtuple

import time
from multiprocessing.pool import ThreadPool

from other.shared_cell_data import obtain_ground_truth_cell_dict, convert_track_frame_idx_to_frame_num
from math import atan2, degrees, radians, cos

global_max_distance = 0

def main():
    folder_path: str = 'D:/viterbi linkage/dataset/'

    save_dir = folder_path + 'save_directory_enhancement/'
    coord_dir = folder_path + "coord_data_3d/"

    is_use_thread: bool = False

    ## hyper parameter settings
    distance = 0.4
    degree = 0.3
    average_movement = 0.3
    weight_tuple_list: list = [  WeightTuple(0.3, 0.4, 0.3),
                                 WeightTuple(0.4, 0.3, 0.3),
                                 WeightTuple(0.8, 0.1, 0.1),
                                 WeightTuple(0.1, 0.8, 0.1),
                                 WeightTuple(0.1, 0.1, 0.8),
                                 WeightTuple(0.1, 0.2, 0.7),
                                 WeightTuple(0.2, 0.7, 0.1),
                                 WeightTuple(0.7, 0.2, 0.1),
                                 WeightTuple(0.6, 0.2, 0.2),
                                 WeightTuple(0.2, 0.6, 0.2),
                                 WeightTuple(0.2, 0.2, 0.6),
                                 WeightTuple(0.5, 0.25, 0.25),
                                 WeightTuple(0.25, 0.5, 0.25),
                                 WeightTuple(0.25, 0.25, 0.5)
                               ]
    max_moving_distance_list: list = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    coord_length_for_vector_list: list = [6]
    average_movement_step_length_list: list = [6]
    minimum_track_length_list: list = [1]
    discount_rate_per_layer_list: list = [0.9] #{"merge_threshold", any_float},



    date_str: str = datetime.now().strftime("%Y%m%d-%H%M%S")

    hyper_para_combination_list = list(itertools.product(
                                                         weight_tuple_list,
                                                         max_moving_distance_list,
                                                         coord_length_for_vector_list,
                                                         average_movement_step_length_list,
                                                         minimum_track_length_list,
                                                         discount_rate_per_layer_list
                                                         ))

    hyper_para_list: list = []
    for hyper_para_combination in hyper_para_combination_list:
        weight_tuple = hyper_para_combination[0]
        max_moving_distance = hyper_para_combination[1]
        coord_length_for_vector = hyper_para_combination[2]
        average_movement_step_length = hyper_para_combination[3]
        minimum_track_length: int = hyper_para_combination[4]
        discount_rate_per_layer: [str, float] = hyper_para_combination[5]

        hyper_para: HyperPara = HyperPara(
                                            weight_tuple,
                                            max_moving_distance,
                                            coord_length_for_vector,
                                            average_movement_step_length,
                                            minimum_track_length,
                                            discount_rate_per_layer)

        hyper_para_list.append(hyper_para)

    total_para_size: int = len(hyper_para_list)
    for idx, hyper_para in enumerate(hyper_para_list):
        para_set_num: int = idx+1

        print(f"start. Parameter set: {para_set_num}/ {total_para_size}; {hyper_para.__str__()}")

        start_time = time.perf_counter()

        input_series_name_list = [input_series.replace(".json", "") for input_series in listdir(coord_dir)]
        # input_series_name_list = ['6_33layers_inter_mask_data__20200802--2_inter_33layers_mask_3a']


        filtered_series_list = []
        output_file_name_suffix = "8"
        include_series_list = ['_8layers_', '_9layers_']

        # include_series_list = ['_29layers_', '_33layers_']

        for input_series in input_series_name_list:
            for include_series in include_series_list:
                if include_series in input_series:
                    filtered_series_list.append(input_series)

        input_series_name_list = filtered_series_list


        feature_based_result_dict = {}

        try:
            for series_name in input_series_name_list:
                print(f"working on series_name: {series_name}. ")

                frame_num_node_id_coord_dict_dict = read_series_data(coord_dir, series_name)


                return_series, final_result_list, score_log_mtx = cell_tracking_core_flow(series_name,
                                                                                          frame_num_node_id_coord_dict_dict,
                                                                                          hyper_para
                                                                                          )
                feature_based_result_dict[series_name] = final_result_list

                is_generate_score_log = True
                if is_generate_score_log:
                    score_log_dir = save_dir + "score_log/"
                    result_file_name: str = Path(__file__).name.replace(".py", "_")

                    # # remove last \n
                    # for frame_num, mtx_row_list in score_log_mtx.items():
                    #     for row_idx in range(len(mtx_row_list)):
                    #         for col_idx in range(len(mtx_row_list[row_idx])):
                    #             if len(score_log_mtx[frame_num][row_idx][col_idx]) > 2:
                    #                 score_log_mtx[frame_num][row_idx][col_idx] = score_log_mtx[frame_num][row_idx][col_idx][0:-2]

                    save_score_log_to_excel(series_name, score_log_mtx, score_log_dir, result_file_name)




        except Exception as e:
            time.sleep(1)
            print()
            traceback.print_exc()
            print(f"series_name {series_name}. para {para_set_num}.  hyper_para: {hyper_para.__str__()}")
            exit()




        is_print_ground_truth_only: bool = False
        if is_print_ground_truth_only:
            series_ground_truth_cell_dict = obtain_ground_truth_cell_dict()

            tmp_dict = defaultdict(list)
            for series_name, track_tuple_list_list in feature_based_result_dict.items():
                ground_truth_cell_id_list = series_ground_truth_cell_dict[series_name]
                for track_tuple_list in track_tuple_list_list:
                    cell_tuple_id = track_tuple_list[0]
                    if cell_tuple_id in ground_truth_cell_id_list:
                        tmp_dict[series_name].append(track_tuple_list)

            feature_based_result_dict = tmp_dict


        is_convert_to_frame_num = True
        result_file_name: str = Path(__file__).name.replace(".py", "")

        hyper_para_indicator: str = str(hyper_para.weight_tuple) + \
                                    "MD(" + str(hyper_para.max_moving_distance) + ")_" + \
                                    "CL(" + str(hyper_para.coord_length_for_vector) + ")_" + \
                                    "AML(" + str(hyper_para.average_movement_step_length) + ")_" + \
                                    "MTL(" + str(hyper_para.minimum_track_length) + ")_" + \
                                    "DR(" +  str(hyper_para.discount_rate_per_layer) + ")_"

        result_dir = save_dir + date_str + "_" + result_file_name + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)


        abs_save_dir: str = result_dir + result_file_name + "_hp" + str(idx+1).zfill(3) + "__" + hyper_para_indicator

        renamed_feature_based_result_dict = {}
        for series_name, result in feature_based_result_dict.items():
            start_idx = series_name.index("__") + 2
            end_idx = start_idx + 11
            renamed_series_name = series_name[start_idx: end_idx]
            renamed_feature_based_result_dict[renamed_series_name] = result

        save_track_dictionary(renamed_feature_based_result_dict, abs_save_dir + output_file_name_suffix + ".pkl")


        execution_time = time.perf_counter() - start_time
        with open(abs_save_dir + ".txt", 'w') as f:
            f.write(f"Execution time: {np.round(execution_time, 4)} seconds\n")
            f.write("hyper_para--- ID: " + str(idx+1) + "; \n" + hyper_para.__str_newlines__())
            f.write("\n")
            for series_name in input_series_name_list:
                f.write("======================" + str(series_name) + "================================")
                f.write("\n")

                cell_track_list_list = sorted(feature_based_result_dict[series_name])
                for cell_track_list in cell_track_list_list:
                    # for cell_track_list in feature_based_result_dict[series_name]:
                    f.write(str(cell_track_list))
                    f.write("\n")

                f.write("\n\n")



        tmp_abs_save_dir: str = save_dir + result_file_name
        with open(tmp_abs_save_dir + ".txt", 'w') as f:
            f.write(f"Execution time: {np.round(execution_time, 4)} seconds\n")
            f.write("hyper_para--- ID: " + str(idx+1) + "; \n" + hyper_para.__str_newlines__())
            f.write("\n")
            for series_name in input_series_name_list:
                f.write("======================" + str(series_name) + "================================")
                f.write("\n")

                cell_track_list_list = sorted(feature_based_result_dict[series_name])
                for cell_track_list in cell_track_list_list:
                    if is_convert_to_frame_num:
                        cell_track_list = convert_track_frame_idx_to_frame_num(cell_track_list)
                    # for cell_track_list in feature_based_result_dict[series_name]:
                    f.write(str(cell_track_list))
                    f.write("\n")
                f.write("\n\n")



        print("save_track_dictionary: ", abs_save_dir)

        print(f"Execution time: {np.round(execution_time, 4)} seconds")






def __________object_start_label():
    raise Exception("for labeling only")

CoordTuple = namedtuple("CoordTuple", "x y z")
WeightTuple = namedtuple("WeightTuple", "degree distance average_movement")


class DataStore(object):


    def __init__(self):
        profit_mtx_data_dict = {}



    @staticmethod
    def obtain_data_a(input_str: str):
        # return data_map_a_dict[input_str]
        print("Hi,this tutorial is about static class in python: " + input_str)



class ROUTING_STRATEGY_ENUM(enum.Enum):
    ALL_LAYER = 1
    ONE_LAYER = 2


class CUT_STRATEGY_ENUM(enum.Enum):
    DURING_ROUTING = 1
    AFTER_ROUTING = 2


class BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM(enum.Enum):
    SHARE = 1


class HyperPara():
    def __init__(self,
                 weight_tuple: tuple,
                 max_moving_distance: int,
                 coord_length_for_vector: int,
                 average_movement_step_length: int,
                 minimum_track_length: int,
                 discount_rate_per_layer: float):

        self.weight_tuple                     = weight_tuple
        self.max_moving_distance              = max_moving_distance
        self.coord_length_for_vector          = coord_length_for_vector
        self.average_movement_step_length     = average_movement_step_length
        self.minimum_track_length             = minimum_track_length
        self.discount_rate_per_layer          = discount_rate_per_layer



    def __str__(self):
        return  f"weight_tuple: {self.weight_tuple}; " \
                f"max_moving_distance: {self.max_moving_distance}; " \
                f"coord_length_for_vector: {self.coord_length_for_vector}; " \
                f"average_movement_step_length: {self.average_movement_step_length}; " \
                f"minimum_track_length: {self.minimum_track_length}; " \
                f"discount_rate_per_layer: {self.discount_rate_per_layer}; "


    def __str_newlines__(self):
        return f"weight_tuple: {self.weight_tuple}; \n" \
               f"max_moving_distance: {self.max_moving_distance}; \n" \
               f"coord_length_for_vector: {self.coord_length_for_vector}; \n" \
               f"average_movement_step_length: {self.average_movement_step_length}; \n" \
               f"minimum_track_length: {self.minimum_track_length}; \n" \
               f"discount_rate_per_layer: {self.discount_rate_per_layer}; "



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





def cell_tracking_core_flow(series: str, frame_num_node_id_coord_dict_dict: dict, hyper_para: HyperPara):

    all_track_dict, score_log_mtx = execute_cell_tracking_task_bon(frame_num_node_id_coord_dict_dict,
                                                                   hyper_para
                                                                   )


    sorted_cell_id_key_list = sorted(list(all_track_dict.keys()), key=cmp_to_key(compare_cell_id))
    sorted_dict: dict = {}
    for sorted_key in sorted_cell_id_key_list:
        sorted_dict[sorted_key] = all_track_dict[sorted_key]
    all_track_dict = sorted_dict

    track_list_list: list = list(all_track_dict.values())

    code_validate_track_list(track_list_list)

    return series, track_list_list, score_log_mtx






def __________component_function_start_label():
    raise Exception("for labeling only")




def execute_cell_tracking_task_bon(frame_num_node_id_coord_dict_dict: dict, hyper_para: HyperPara):

    to_handle_cell_id_list: list = derive_initial_cell_id_list(frame_num_node_id_coord_dict_dict)

    frame_num_node_idx_occupation_list_list_dict = initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_id_coord_dict_dict)

    total_frame: int = max(frame_num_node_id_coord_dict_dict.keys())

    all_valid_cell_track_idx_dict: dict = {}        # [cell_id][frame_num] = node_idx
    all_valid_cell_track_score_dict: dict = {}       # [cell_id][frame_num] = probability

    score_log_mtx = init_score_log(frame_num_node_id_coord_dict_dict)

    while len(to_handle_cell_id_list) != 0:
        to_handle_cell_id_list.sort(key=cmp_to_key(compare_cell_id))

        to_handle_cell_id: CellId = to_handle_cell_id_list[0]

        print(f"{to_handle_cell_id.str_short()}")

        if to_handle_cell_id in all_valid_cell_track_idx_dict:
            frame_num_node_idx_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_occupation_list_list_dict, to_handle_cell_id, all_valid_cell_track_idx_dict[to_handle_cell_id])
            del all_valid_cell_track_idx_dict[to_handle_cell_id]
            del all_valid_cell_track_score_dict[to_handle_cell_id]


        is_new_cell: bool = (len(frame_num_node_idx_occupation_list_list_dict[to_handle_cell_id.start_frame_num][to_handle_cell_id.cell_idx]) == 0)
        if not is_new_cell:
            has_exist_data: bool = (to_handle_cell_id in all_valid_cell_track_idx_dict)
            if has_exist_data:
                frame_num_node_idx_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_occupation_list_list_dict, to_handle_cell_id, all_valid_cell_track_idx_dict[to_handle_cell_id])
                del all_valid_cell_track_idx_dict[to_handle_cell_id]
                del all_valid_cell_track_score_dict[to_handle_cell_id]

            to_handle_cell_id_list.remove(to_handle_cell_id)
            continue


        handling_cell_frame_num_track_idx_dict: dict = { to_handle_cell_id.start_frame_num: to_handle_cell_id.cell_idx}
        handling_cell_frame_num_score_dict: dict = {}

        redo_cell_id_set: set = set()

        second_frame_num: int = to_handle_cell_id.start_frame_num + 1
        for connect_to_frame_num in inclusive_range(second_frame_num, total_frame):
            total_node: int = len(frame_num_node_id_coord_dict_dict[connect_to_frame_num])
            is_connect_to_frame_has_no_cell = (total_node == 0)
            if is_connect_to_frame_has_no_cell:
                break


            previous_frame_num: int = connect_to_frame_num - 1
            # previous_frame_cell_idx: int = handling_cell_frame_num_track_idx_dict[previous_frame_num]
            node_id_coord_dict: dict = frame_num_node_id_coord_dict_dict[connect_to_frame_num]

            cut_threshold = 1000
            best_idx, best_prob, score_log_mtx, redo_cell_id_set = derive_best_node_idx_to_connect(to_handle_cell_id,
                                                                                                   handling_cell_frame_num_track_idx_dict,
                                                                                                   node_id_coord_dict,
                                                                                                   connect_to_frame_num,
                                                                                                   frame_num_node_idx_occupation_list_list_dict,
                                                                                                   all_valid_cell_track_idx_dict,
                                                                                                   all_valid_cell_track_score_dict,
                                                                                                   frame_num_node_id_coord_dict_dict,
                                                                                                   score_log_mtx,
                                                                                                   cut_threshold,
                                                                                                   redo_cell_id_set,
                                                                                                   hyper_para.weight_tuple,
                                                                                                   hyper_para.max_moving_distance,
                                                                                                   hyper_para.coord_length_for_vector,
                                                                                                   hyper_para.average_movement_step_length)




            # is_best_prob_lower_than_threshold: bool = best_prob < hyper_para.cut_threshold
            if best_prob == 0:
                break

            if connect_to_frame_num in handling_cell_frame_num_track_idx_dict:   raise Exception("code validation check")
            if connect_to_frame_num in handling_cell_frame_num_score_dict:       raise Exception("code validation check")

            handling_cell_frame_num_track_idx_dict[connect_to_frame_num] = best_idx
            handling_cell_frame_num_score_dict[connect_to_frame_num] = best_prob

        is_track_above_min_length: bool = (len(handling_cell_frame_num_track_idx_dict.keys()) > hyper_para.minimum_track_length)

        if not is_track_above_min_length:
            to_handle_cell_id_list.remove(to_handle_cell_id)
            continue


        # use final track to check if any redo cells occur
        for redo_cell_id in redo_cell_id_set:
            # print(f"redo cell {redo_cell_id}. ", end='')
            if redo_cell_id not in to_handle_cell_id_list:
                to_handle_cell_id_list.append(redo_cell_id)

        # add to occupation list
        frame_num_node_idx_occupation_list_list_dict = add_track_to_cell_occupation_list_list_dict(frame_num_node_idx_occupation_list_list_dict, to_handle_cell_id, handling_cell_frame_num_track_idx_dict)

        # add track to valid track list
        all_valid_cell_track_idx_dict[to_handle_cell_id] = handling_cell_frame_num_track_idx_dict
        all_valid_cell_track_score_dict[to_handle_cell_id] = handling_cell_frame_num_score_dict

        to_handle_cell_id_list.remove(to_handle_cell_id)

    print("----> finish")


    all_cell_id_track_list_dict: dict = defaultdict(list)
    for cell_id, handling_cell_frame_num_track_idx_dict in all_valid_cell_track_idx_dict.items():
        track_tuple_list: list = []
        previous_node_idx: int = -1
        for connect_to_frame_num, node_idx in handling_cell_frame_num_track_idx_dict.items():
            frame_idx: int = connect_to_frame_num - 1
            track_tuple_list.append((node_idx, frame_idx, previous_node_idx))
            previous_node_idx = node_idx

        all_cell_id_track_list_dict[cell_id] = track_tuple_list

    return all_cell_id_track_list_dict, score_log_mtx








def execute_cell_tracking_task_1(frame_num_prof_matrix_dict: dict, hyper_para, is_use_cell_dependency_feature: bool):
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
                frame_num_node_idx_cell_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict_old(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id, cell_id_track_list_dict[handling_cell_id], remove_from_frame_num=1)
                del cell_id_track_list_dict[handling_cell_id]

            code_validate_if_cellid_not_exist_in_occupation_data(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id)

            if handling_cell_id in cell_id_frame_num_track_progress_dict:
                del cell_id_frame_num_track_progress_dict[handling_cell_id]


            cell_id_track_list_dict[handling_cell_id] = cell_track_list

            frame_num_node_idx_cell_occupation_list_list_dict = add_track_to_cell_occupation_list_list_dict_old(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id, cell_track_list)




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


def read_series_data(coord_dir, series_name):

    coord_file_path = coord_dir + series_name + ".json"

    with open(coord_file_path) as json_file:
        key_str_cell_coord_list_dict = json.load(json_file)


    frame_num_node_id_coord_dict_dict = defaultdict(dict)
    for key_str, cell_coord_list in key_str_cell_coord_list_dict.items():
        frame_num = int(key_str.split(":")[0])
        node_num = int(key_str.split(":")[1])

        z = cell_coord_list[0]
        x = cell_coord_list[1]
        y = cell_coord_list[2]
        coord_tuple: CoordTuple = CoordTuple(x, y, z)

        frame_num_node_id_coord_dict_dict[frame_num][node_num] = coord_tuple

    return frame_num_node_id_coord_dict_dict



def derive_average_movement_score(to_handle_cell_id,
                                  current_frame_num,
                                  last_frame_node_coord,
                                  candidate_node_coord,
                                  step_length,
                                  handling_cell_frame_num_track_idx_dict,
                                  frame_num_node_id_coord_dict_dict,
                                  max_moving_distance: float):

    start_frame = current_frame_num - step_length
    start_frame = max(start_frame, 1, to_handle_cell_id.start_frame_num)
    end_frame = current_frame_num


    actual_step_length: int = (end_frame - start_frame) + 1
    if actual_step_length > 1:
        sum_distance: float = 0

        previous_node_idx: int = handling_cell_frame_num_track_idx_dict[start_frame]
        previous_coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[start_frame][previous_node_idx]
        for frame_num in inclusive_range(start_frame+1, end_frame):
            current_node_idx: int = handling_cell_frame_num_track_idx_dict[frame_num]
            current_coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[frame_num][current_node_idx]

            x_y_distance: float = ((current_coord_tuple.x - previous_coord_tuple.x)**2 + (current_coord_tuple.y - previous_coord_tuple.y)**2)**0.5
            distance: float = (x_y_distance**2 + (current_coord_tuple.z - previous_coord_tuple.z)**2)**0.5
            sum_distance += distance

            previous_coord_tuple = current_coord_tuple
        average_movement: float = sum_distance / actual_step_length

        new_candidate_node_distance: float = ((candidate_node_coord.x - last_frame_node_coord.x)**2 + (candidate_node_coord.y - last_frame_node_coord.y)**2)**0.5

        movement_difference: float = abs(average_movement - new_candidate_node_distance)

        if movement_difference > max_moving_distance:
            normalized_average_movement_score = 0
        else:
            normalized_average_movement_score: float = (max_moving_distance - movement_difference) / max_moving_distance

    else:
        normalized_average_movement_score = 0.5

    return normalized_average_movement_score





def update_below_cut_threshold_value_to_zero(frame_num_prof_matrix_dict: dict, cut_threshold: float):
    for frame_num, prof_mtx in frame_num_prof_matrix_dict.items():
        for row_idx in range(prof_mtx.shape[0]):
            for col_idx in range(prof_mtx.shape[1]):
                if frame_num_prof_matrix_dict[frame_num][row_idx][col_idx] <= cut_threshold:
                    frame_num_prof_matrix_dict[frame_num][row_idx][col_idx] = 0

    return frame_num_prof_matrix_dict


def derive_distance_score(current_node_coord, last_frame_node_coord, max_moving_distance):
    x_y_distance: float = ((current_node_coord.x - last_frame_node_coord.x)**2 + (current_node_coord.y - last_frame_node_coord.y)**2)**0.5

    x_y_z_distance: float = (x_y_distance**2 + (current_node_coord.z - last_frame_node_coord.z)**2)**0.5
    x_y_z_distance = np.round(x_y_z_distance, 4)

    if x_y_z_distance > max_moving_distance:
        distance_score = 0
    else:
        distance_score = (max_moving_distance - x_y_z_distance) / max_moving_distance

    return distance_score


def derive_degree_score(to_handle_cell_id, current_frame_num, last_frame_node_coord, candidate_node_coord,
                        coord_length_of_vector, handling_cell_frame_num_track_idx_dict, frame_num_node_id_coord_dict_dict):
    previous_coord_list = []
    start_frame = current_frame_num - coord_length_of_vector
    start_frame = max(start_frame, 1, to_handle_cell_id.start_frame_num)
    end_frame = current_frame_num

    coord_length: int = (end_frame - start_frame) + 1
    if coord_length > 1:
        for frame_num in inclusive_range(start_frame, end_frame):
            node_id: int = handling_cell_frame_num_track_idx_dict[frame_num]
            coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[frame_num][node_id]
            previous_coord_list.append(coord_tuple)
        previous_vec: CoordTuple = derive_vector_from_coord_list(previous_coord_list)

        new_candidate_vec: CoordTuple = derive_vector_from_coord_list([last_frame_node_coord, candidate_node_coord])

        degree_diff: float = derive_degree_diff_from_two_vectors(previous_vec, new_candidate_vec)

        degree_score: float = (cos(radians(degree_diff)) + 1) * 0.5  # +1 and *0.5 to shift up and make it stay between 1 and 0

    else:
        degree_score: float = 0.5

    # if math.isnan(degree_score):
    #     print("isnan")

    return degree_score



def init_score_log(frame_num_node_id_coord_dict_dict):
    frame_num_score_log_mtx = {}

    start_frame_num = min(frame_num_node_id_coord_dict_dict.keys())
    end_frame_num = max(frame_num_node_id_coord_dict_dict.keys())
    for frame_num in range(start_frame_num, end_frame_num):
        num_node_id_coord_dict = frame_num_node_id_coord_dict_dict[frame_num]
        next_frame_num_node_id_coord_dict = frame_num_node_id_coord_dict_dict[frame_num+1]
        total_row = max(num_node_id_coord_dict.keys()) + 1
        total_col = max(next_frame_num_node_id_coord_dict) + 1
        row_list = []

        for row_idx in range(total_row):
            col_list = []
            for col_idx in range(total_col):
                col_list.append(0)
            row_list.append(col_list)
        frame_num_score_log_mtx[frame_num] = row_list

    return frame_num_score_log_mtx



def save_score_log_to_excel(series: str, frame_num_score_matrix_dict, excel_output_dir_path: str, file_name_prefix: str = ""):
    import pandas as pd
    # num_of_segementation_img: int = len(frame_num_prof_matrix_dict)

    file_name: str = file_name_prefix + f"series_{series}.xlsx"
    filepath = excel_output_dir_path + file_name;
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter') #pip install xlsxwriter

    for frame_num, prof_matrix in frame_num_score_matrix_dict.items():
        tmp_array: np.arrays = frame_num_score_matrix_dict[frame_num]

        df = pd.DataFrame (tmp_array)

        # def highlight_cells(val):
        #     color = 'yellow' if val >= 1  else 'white'
        #     return 'background-color: {}'.format(color)
        #
        # df = df.style.applymap(highlight_cells)

        sheet_name: str = "frame_1" if frame_num == 1 else str(frame_num)

        df.to_excel(writer, sheet_name=sheet_name, index=True)

        workbook = writer.book
        cell_format = workbook.add_format({'text_wrap': True})        # to make \n work

        worksheet = writer.sheets[sheet_name]
        worksheet.set_column('A:BO', cell_format=cell_format)

        for col_idx in range(40):
            if col_idx == 0:    col_width = 3
            else:               col_width = 25
            worksheet.set_column(col_idx, col_idx, col_width)

    writer.save()




def derive_frame_num_distance_matrix_dict(frame_num_node_idx_coord_list_dict: dict):
    start_frame = min(list(frame_num_node_idx_coord_list_dict.keys()))
    end_frame = max(list(frame_num_node_idx_coord_list_dict.keys()))

    frame_num_distance_matrix_dict: dict = {}
    for frame_num in inclusive_range(start_frame, end_frame-1):

        current_frame_node_list = frame_num_node_idx_coord_list_dict[frame_num]
        next_frame_node_list = frame_num_node_idx_coord_list_dict[frame_num]

        total_row = len(current_frame_node_list)
        total_col = len(next_frame_node_list)
        default_value: float = float(-1)
        distance_matrix: np.array = np.full((total_row, total_col), default_value)
        for row_idx, current_frame_coord in enumerate(current_frame_node_list):
            for col_idx, next_frame_coord in enumerate(next_frame_node_list):
                distance: float = ((next_frame_coord.x - current_frame_coord.x)**2 + (next_frame_coord.y - current_frame_coord.x)**2)**0.5
                distance_matrix[row_idx][col_idx] = np.round(distance, 4)

        frame_num_distance_matrix_dict[frame_num] = distance_matrix

    return frame_num_distance_matrix_dict



def derive_vector_from_coord_list(coord_tuple_list: list):
    if len(coord_tuple_list) == 1:
        raise Exception("code validation check")

    start_vec: CoordTuple = coord_tuple_list[0]
    end_vec: CoordTuple = coord_tuple_list[-1]

    reversed_y_coord = -(end_vec.y - start_vec.y) # reverse y because coord raw data of y-axis is reversed
    z_coord = end_vec.z - start_vec.z
    final_coord_tuple = CoordTuple(end_vec.x - start_vec.x, reversed_y_coord, z_coord)

    return final_coord_tuple




def derive_initial_cell_id_list(frame_num_node_id_coord_dict_dict: dict):
    to_handle_cell_id_list: list = []
    for frame_num, node_id_coord_dict in frame_num_node_id_coord_dict_dict.items():
        for node_id in node_id_coord_dict.keys():
            cell_start_frame: int = frame_num
            to_handle_cell_id_list.append(CellId(cell_start_frame, node_id))

    return to_handle_cell_id_list


def derive_directional_score(track_coord_list: list, new_coord: CoordTuple):
    if len(track_coord_list) <= 1:
        return 1






def derive_best_node_idx_to_connect(to_handle_cell_id,
                                    handling_cell_frame_num_track_idx_dict,
                                    node_id_coord_dict,
                                    connect_to_frame_num,
                                    frame_num_node_idx_occupation_list_list_dict,
                                    all_valid_cell_track_idx_dict,
                                    all_valid_cell_track_prob_dict,
                                    frame_num_node_id_coord_dict_dict,
                                    score_log_mtx,
                                    cut_threshold,
                                    redo_cell_id_set,
                                    weight_tuple: WeightTuple,
                                    max_moving_distance: int,
                                    coord_length_for_vector: int,
                                    average_movement_step_length: int):


    current_frame_num = connect_to_frame_num - 1
    round_to = 2

    current_frame_node_id: int = handling_cell_frame_num_track_idx_dict[current_frame_num]
    current_frame_node_coord: CoordTuple = frame_num_node_id_coord_dict_dict[current_frame_num][current_frame_node_id]

    best_prob: float = 0
    best_node_idx: int = None

    tmp_redo_node_idx_cell_id_dict: dict = defaultdict(set)
    for candidate_node_id, candidate_node_coord in node_id_coord_dict.items():
        final_score: float = 0

        is_use_degree_score: bool = (weight_tuple.degree > 0)
        if is_use_degree_score:
            degree_score: float = derive_degree_score(to_handle_cell_id,
                                                      current_frame_num,
                                                      current_frame_node_coord,
                                                      candidate_node_coord,
                                                      coord_length_for_vector,
                                                      handling_cell_frame_num_track_idx_dict,
                                                      frame_num_node_id_coord_dict_dict)

            weighted_degree_score = np.round(weight_tuple.degree * degree_score, round_to)

            if math.isnan(weighted_degree_score):
                weighted_degree_score = 0

            # if current_frame_num == 5 and current_frame_node_id == 8:
            #     print("sgsfd", weighted_degree_score, math.isnan(weighted_degree_score))
            #     time.sleep(2)

            final_score += weighted_degree_score
        else:
            weighted_degree_score = 0.5

        # if weighted_degree_score == nan:
        #     raise Exception("weighted_degree_score == None")


        is_use_distance_score: bool = (weight_tuple.distance > 0)
        if is_use_distance_score:
            distance_score = derive_distance_score(candidate_node_coord, current_frame_node_coord, max_moving_distance)

            weighted_distance_score = np.round(weight_tuple.distance * distance_score, round_to)
            final_score += weighted_distance_score
        else:
            weighted_distance_score = 0.5


        is_use_avg_movement_score: bool = (weight_tuple.average_movement > 0)
        if is_use_avg_movement_score:
            avg_movement_score = derive_average_movement_score(to_handle_cell_id,
                                                               current_frame_num,
                                                               current_frame_node_coord,
                                                               candidate_node_coord,
                                                               average_movement_step_length,
                                                               handling_cell_frame_num_track_idx_dict,
                                                               frame_num_node_id_coord_dict_dict,
                                                               max_moving_distance)
            weighted_avg_mov_score = np.round(weight_tuple.average_movement * avg_movement_score, round_to)
            final_score += weighted_avg_mov_score
        else:
            weighted_avg_mov_score = 0.5

        if distance_score == 0:            final_score = 0
        else:                              final_score = np.round(final_score, round_to)

        # current_node_idx: int = handling_cell_frame_num_track_idx_dict[current_frame_num]
        # log_msg = f"{to_handle_cell_id.str_short()} {final_score}={weighted_probability_score}+{weighted_degree_score}+{weighted_distance_score}+{weighted_avg_mov_score}"
        # if log_msg not in score_log_mtx[current_frame_num][current_node_idx][candidate_node_id]:
        #     score_log_mtx[current_frame_num][current_node_idx][candidate_node_id] += log_msg + "\n"


        # score_log_mtx[current_frame_num][current_frame_node_id][candidate_node_id] = f"final_score:{final_score}; dist:{weighted_distance_score}; deg:{weighted_degree_score}; avgM:{weighted_avg_mov_score}"
        score_log_mtx[current_frame_num][current_frame_node_id][candidate_node_id] = final_score

        # print("sadbfsdf", connect_to_frame_num, candidate_node_id)
        occupied_cell_id_list: list = frame_num_node_idx_occupation_list_list_dict[connect_to_frame_num][candidate_node_id]
        has_cell_occupation: bool = (len(occupied_cell_id_list) > 0)
        has_cell_occupation = False

        if not has_cell_occupation:
            if final_score > best_prob:
                best_prob = final_score
                best_node_idx = candidate_node_id
        else:
            continue

            # has_other_connection_option_current_cell: bool = (count_non_zero_data_in_list(node_id_coord_dict) > 1)
            # for occupied_cell_id in occupied_cell_id_list:
            #
            #     if node_id_coord_dict == occupied_cell_id.start_frame_num:
            #         if final_score > best_prob:
            #             best_prob = final_score
            #             best_node_idx = candidate_node_id
            #             tmp_redo_node_idx_cell_id_dict[best_node_idx].add(occupied_cell_id)
            #         continue
            #
            #
            #     occupied_cell_current_frame_idx = all_valid_cell_track_idx_dict[occupied_cell_id][current_frame_num]
            #     occupied_cell_node_connection_prob_list = frame_num_prof_matrix_dict[current_frame_num][occupied_cell_current_frame_idx]
            #
            #     has_other_connection_option_occupied_cell: bool = (count_non_zero_data_in_list(occupied_cell_node_connection_prob_list) > 1)
            #
            #     if not has_other_connection_option_current_cell and not has_other_connection_option_occupied_cell:
            #         #merge as usual
            #         if final_score > best_prob:
            #             best_prob = final_score
            #             best_node_idx = candidate_node_id
            #
            #     elif has_other_connection_option_current_cell and not has_other_connection_option_occupied_cell:
            #         # current cell explore other options
            #         continue
            #
            #     elif not has_other_connection_option_current_cell and has_other_connection_option_occupied_cell:
            #         # occupied cell explore other options
            #         if final_score > best_prob:
            #             best_prob = final_score
            #             best_node_idx = candidate_node_id
            #             tmp_redo_node_idx_cell_id_dict[candidate_node_id].add(occupied_cell_id)
            #
            #     elif has_other_connection_option_current_cell and has_other_connection_option_occupied_cell:
            #         # higher probability cell takes over
            #         occupied_cell_prob: float = all_valid_cell_track_prob_dict[occupied_cell_id][node_id_coord_dict]
            #
            #         if occupied_cell_prob > final_score:
            #             #current cell explore other opportunity
            #             continue
            #         elif occupied_cell_prob < final_score:
            #             if final_score > best_prob:
            #                 # occupied cell explore other opportunity
            #                 best_prob = final_score
            #                 best_node_idx = candidate_node_id
            #                 tmp_redo_node_idx_cell_id_dict[best_node_idx].add(occupied_cell_id)
            #
            #         elif occupied_cell_prob == final_score:
            #             if final_score > best_prob:
            #                 # print("(let them share for now) to handle biz scenario", occupied_cell_prob, candidate_node_coord, to_handle_cell_id.str_short(), occupied_cell_id.str_short(), current_frame_num)
            #                 best_prob = final_score
            #                 best_node_idx = candidate_node_id
            #                 # continue
            #
            #             # raise Exception("Unexpected biz scenario", occupied_cell_prob, candidate_node_coord, to_handle_cell_id.str_short(), occupied_cell_id.str_short(), current_frame_num)
            #         else:
            #             raise Exception("code validation check")
            #
            #
            #     else:
            #         raise Exception("code validation check")

    # if best_node_idx in tmp_redo_node_idx_cell_id_dict:
    #     for redo_cell_id in tmp_redo_node_idx_cell_id_dict[best_node_idx]:
    #         if redo_cell_id == CellId(84, 6) and node_id_coord_dict == 84:
    #             tmp = all_valid_cell_track_prob_dict[redo_cell_id]
    #             print("sadg", redo_cell_id.str_short(), node_id_coord_dict)
    #
    #         occu_cell_prob = all_valid_cell_track_prob_dict[redo_cell_id][node_id_coord_dict]
    #         print(f"redo cell {redo_cell_id}; collision at frame {node_id_coord_dict}; node_idx: {best_node_idx}; curr_cell_prob: {best_prob}; occu_cell_prob: {occu_cell_prob}")
    #         redo_cell_id_set.add(redo_cell_id)

    return best_node_idx, best_prob, score_log_mtx, redo_cell_id_set


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
        frame_num_node_idx_cell_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict_old(frame_num_node_idx_cell_occupation_list_list_dict, handling_cell_id, cell_id_track_list_dict[handling_cell_id], delete_from_frame_num)
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



def initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_id_coord_dict_dict: dict):
    frame_num_node_idx_occupation_list_list_dict: dict = defaultdict(dict)

    for frame_num, node_id_coord_dict in frame_num_node_id_coord_dict_dict.items():
        for node_id in node_id_coord_dict.keys():
            frame_num_node_idx_occupation_list_list_dict[frame_num][node_id] = []

    return frame_num_node_idx_occupation_list_list_dict



def remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, frame_num_cell_track_idx_dict: dict):
    for frame_num, node_idx in frame_num_cell_track_idx_dict.items():
        if cell_id not in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx]:
            raise Exception("code validation")

        # if cell_id in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx]:
        frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx].remove(cell_id)


    return frame_num_node_idx_occupation_tuple_vec_dict




def add_track_to_cell_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, frame_num_cell_track_idx_dict: dict):
    for frame_num, node_idx in frame_num_cell_track_idx_dict.items():
        if cell_id in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx]:
            raise Exception("code validation")

        # if cell_id not in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx]:
        frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx].append(cell_id)


    return frame_num_node_idx_occupation_tuple_vec_dict


def add_track_to_cell_occupation_list_list_dict_old(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, track_tuple_list: list):
    for track_tuple in track_tuple_list:
        frame_num: int = track_tuple[1] + 1
        occupied_node_idx: int = track_tuple[0]

        if cell_id not in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][occupied_node_idx]:
            frame_num_node_idx_occupation_tuple_vec_dict[frame_num][occupied_node_idx].append(cell_id)

    return frame_num_node_idx_occupation_tuple_vec_dict





def remove_track_from_cell_occupation_list_list_dict_old(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, track_tuple_list: list, remove_from_frame_num: int):
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



def derive_frame_num_node_idx_coord_list_dict(frame_num_cell_coord_list_dict):

    return frame_num_cell_coord_list_dict



def derive_frame_num_prof_matrix_dict(frame_num_node_num_cell_coord_list_dict):
    frame_num_prof_matrix_dict: dict = {}

    for frame_num, node_num_cell_coord_list in frame_num_node_num_cell_coord_list_dict.items():
        if frame_num == 1:
            previous_frame_total_cell = len(node_num_cell_coord_list)
            continue

        current_frame_total_cell = len(node_num_cell_coord_list)
        # print(frame_num-1, previous_frame_total_cell, )
        frame_num_prof_matrix_dict[frame_num-1] = np.zeros((previous_frame_total_cell, current_frame_total_cell))

        previous_frame_total_cell = current_frame_total_cell


    total_frame = max(frame_num_node_num_cell_coord_list_dict.keys())
    for frame_num in range(1, total_frame):
        current_frame_coord_list = frame_num_node_num_cell_coord_list_dict[frame_num]
        next_frame_coord_list = frame_num_node_num_cell_coord_list_dict[frame_num + 1]

        for current_frame_idx, current_frame_coord in enumerate(current_frame_coord_list):
            if current_frame_coord == None:
                continue

            for next_frame_idx, next_frame_coord in enumerate(next_frame_coord_list):
                if next_frame_coord == None:
                    continue

                frame_num_prof_matrix_dict[frame_num][current_frame_idx][next_frame_idx] = 1


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

def condition(x):
    return x!=0

def count_non_zero_data_in_list(data_list: list):
    total_non_zero_data: int = sum(condition(x) for x in data_list)
    return total_non_zero_data



def derive_degree_diff_from_two_vectors(vector_coord_tuple_1: CoordTuple, vector_coord_tuple_2: CoordTuple): #These can also be four parameters instead of two arrays
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

        angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        angle_between((1, 0, 0), (1, 0, 0))
        0.0
        angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """

    vec1_unit = unit_vector(vector_coord_tuple_1)
    vec2_unit = unit_vector(vector_coord_tuple_2)

    diff_radius: float = np.arccos(np.clip(np.dot(vec1_unit, vec2_unit), -1.0, 1.0))
    diff_degree: float = degrees(diff_radius)

    if diff_degree < 0:
        diff_degree = abs(diff_degree)

    return diff_degree



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)





def derive_degree_from_vector(vector_coord_tuple_1: CoordTuple): #These can also be four parameters instead of two arrays
    default_zero_degree_vector = CoordTuple(0, 1)

    dot = default_zero_degree_vector.x * vector_coord_tuple_1.x + default_zero_degree_vector.y * vector_coord_tuple_1.y      # dot product
    det = default_zero_degree_vector.y * vector_coord_tuple_1.x - default_zero_degree_vector.x * vector_coord_tuple_1.y      # determinant
    angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    angle = degrees(angle)

    if angle < 0:
        angle += 360

    return angle



def inclusive_range(start: int, end: int):
    return range(start, end+1)



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