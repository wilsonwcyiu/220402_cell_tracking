# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:31:51 2021

@author: 13784
"""
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

class STRATEGY_ENUM(enum.Enum):
    ALL_LAYER = 1
    ONE_LAYER = 2

def main():

    folder_path: str = 'D:/viterbi linkage/dataset/'

    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    images_folder = folder_path + 'dataset//images//'
    output_folder = folder_path + 'output_unet_seg_finetune//'
    save_dir = folder_path + 'save_directory_enhancement/'


    is_use_thread: bool = False

    ## hyper parameter settings
    strategy_enum: STRATEGY_ENUM = STRATEGY_ENUM.ALL_LAYER
    merge_threshold: float = float(0.0)
    minimum_track_length: int = 5
    cut_threshold: float = float(0.01)
    hyper_para: HyperPara = HyperPara(strategy_enum, merge_threshold, minimum_track_length, cut_threshold)



    print("start")
    start_time = time.perf_counter()


    input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                         'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    # input_series_list = ['S13']

    all_segmented_filename_list = listdir(segmentation_folder)
    all_segmented_filename_list.sort()

    existing_series_list = derive_existing_series_list(input_series_list, listdir(output_folder))

    viterbi_result_dict = {}

    if is_use_thread:
        for input_series in input_series_list:
            viterbi_result_dict[input_series] = []

        pool = ThreadPool(processes=8)
        thread_list: list = []

        for series in existing_series_list:
            print(f"\n\n\n\nworking on series: {series}. ", end="\t")

            async_result = pool.apply_async(cell_tracking_core_flow, (series, segmentation_folder, all_segmented_filename_list, output_folder, hyper_para, )) # tuple of args for foo
            thread_list.append(async_result)

        for thread_idx in range(len(thread_list)):
            final_result_list = thread_list[thread_idx].get()
            viterbi_result_dict[series] = final_result_list
            print(f"Thread {thread_idx} completed")

    else:
        for series in existing_series_list:
            print(f"working on series: {series}", end="\t")
            final_result_list = cell_tracking_core_flow(series, segmentation_folder, all_segmented_filename_list, output_folder, hyper_para)
            viterbi_result_dict[series] = final_result_list


    print("save_track_dictionary")
    save_track_dictionary(viterbi_result_dict, save_dir + "viterbi_results_dict.pkl")

    result_txt_file_name: str = Path(__file__).name
    with open(save_dir + result_txt_file_name + ".txt", 'w') as f:
        for series in existing_series_list:
            f.write("======================" + str(series) + "================================")
            f.write("\n")
            cell_track_list_list = sorted(viterbi_result_dict[series])
            for cell_track_list in cell_track_list_list:
                f.write(str(cell_track_list))
                f.write("\n")

            f.write("\n\n")



    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {np.round(execution_time, 4)} seconds")




def __________flow_function_start_label():
    raise Exception("for labeling only")





def cell_tracking_core_flow(series: str, segmentation_folder: str, all_segmented_filename_list: list, output_folder: str, hyper_para, is_create_excel:bool=False):

    segmented_filename_list: list = derive_segmented_filename_list_by_series(series, all_segmented_filename_list)


    frame_num_prof_matrix_dict: dict = derive_frame_num_prof_matrix_dict(segmentation_folder, output_folder, series, segmented_filename_list)


    if is_create_excel:
        save_prof_matrix_to_excel(series, frame_num_prof_matrix_dict, excel_output_dir_path="d:/tmp/")


    all_track_dict = execute_cell_tracking_task(frame_num_prof_matrix_dict, hyper_para)


    all_track_dict = filter_track_dict_by_length(all_track_dict, hyper_para.minimum_track_length)


    sorted_cell_id_key_list = sorted(list(all_track_dict.keys()), key=cmp_to_key(compare_cell_id))
    sorted_dict: dict = {}
    for sorted_key in sorted_cell_id_key_list:
        sorted_dict[sorted_key] = all_track_dict[sorted_key]
    all_track_dict = sorted_dict


    track_list_list = filter_track_list_by_length(all_track_dict.values(), hyper_para.minimum_track_length)


    is_do_post_adjustment: bool = True
    if is_do_post_adjustment:
        prof_mat_list: list = deprecate_derive_prof_matrix_list(segmentation_folder, output_folder, series, segmented_filename_list)
        final_track_list = post_adjustment_old(track_list_list, prof_mat_list)

        return final_track_list

    else:
        return track_list_list






def __________component_function_start_label():
    raise Exception("for labeling only")





def execute_cell_tracking_task(frame_num_prof_matrix_dict: dict, hyper_para):
    all_cell_id_track_list_dict: dict = {}

    start_frame_num: int = 1
    first_frame_mtx: np.array = frame_num_prof_matrix_dict[start_frame_num]
    total_cell_in_first_frame: int = first_frame_mtx.shape[0]
    to_handle_cell_id_list: list = [CellId(1, cell_idx) for cell_idx in range(0, total_cell_in_first_frame)]

    cell_id_frame_num_node_idx_best_index_list_dict_dict: dict = defaultdict(dict)
    cell_id_frame_num_node_idx_best_value_list_dict_dict: dict = defaultdict(dict)



    cell_idx_track_list_dict, \
    cell_id_frame_num_node_idx_best_index_list_dict_dict, \
    cell_id_frame_num_node_idx_best_value_list_dict_dict = \
                                                            _process_and_find_best_cell_track(all_cell_id_track_list_dict,
                                                                                              to_handle_cell_id_list,
                                                                                              frame_num_prof_matrix_dict,
                                                                                              cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                                              cell_id_frame_num_node_idx_best_value_list_dict_dict,
                                                                                              hyper_para.merge_threshold,
                                                                                              hyper_para.strategy_enum)


    cell_idx_short_track_list_dict = _cut_1(cell_idx_track_list_dict, hyper_para.cut_threshold, frame_num_prof_matrix_dict)   # filter out cells that does not make sense (e.g. too low probability)

    all_cell_id_track_list_dict.update(cell_idx_short_track_list_dict)



    ##
    ## handle new cells that enter the image
    ##
    mask_transition_group_mtx_list = _initiate_mask(frame_num_prof_matrix_dict)
    mask_transition_group_mtx_list = _mask_update(cell_idx_short_track_list_dict, mask_transition_group_mtx_list)

    second_frame_num: int = 2
    last_frame_num: int = np.max(list(frame_num_prof_matrix_dict.keys()))

    for frame_num in range(second_frame_num, last_frame_num):
        profit_matrix_idx = frame_num - 1
        for cell_row_idx in range(frame_num_prof_matrix_dict[frame_num].shape[0]):  #skip all nodes which are already passed

            is_old_call: bool = (mask_transition_group_mtx_list[profit_matrix_idx][cell_row_idx] == True)

            if is_old_call:
                continue

            cell_id = CellId(frame_num, cell_row_idx)



            to_handle_cell_id_list: list = [cell_id]
            new_cell_idx_track_list_dict, \
            cell_id_frame_num_node_idx_best_index_list_dict_dict, \
            cell_id_frame_num_node_idx_best_value_list_dict_dict = \
                                                                    _process_and_find_best_cell_track(all_cell_id_track_list_dict,
                                                                                                      to_handle_cell_id_list,
                                                                                                      frame_num_prof_matrix_dict,
                                                                                                      cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                                                      cell_id_frame_num_node_idx_best_value_list_dict_dict,
                                                                                                      hyper_para.merge_threshold,
                                                                                                      hyper_para.strategy_enum)

            code_validate_track(new_cell_idx_track_list_dict)

            new_short_cell_id_track_list_dict = _cut_1(new_cell_idx_track_list_dict, hyper_para.cut_threshold, frame_num_prof_matrix_dict)   # filter out cells that does not make sense (e.g. too low probability)

            code_validate_track(new_short_cell_id_track_list_dict)

            mask_transition_group_mtx_list = _mask_update(new_short_cell_id_track_list_dict, mask_transition_group_mtx_list)

            all_cell_id_track_list_dict.update(new_short_cell_id_track_list_dict)

            code_validate_track(all_cell_id_track_list_dict)

    return all_cell_id_track_list_dict




#loop each node on first frame to find the optimal path using probabilty multiply
def _process_and_find_best_cell_track(existing_cell_idx_track_list_dict,
                                      to_handle_cell_id_list: list, frame_num_prof_matrix_dict: dict,
                                      cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                      cell_id_frame_num_node_idx_best_value_list_dict_dict,
                                      merge_above_threshold: float,
                                      strategy_enum: STRATEGY_ENUM):

    cell_id_track_list_dict: dict = {}


    to_skip_cell_id_list: list = []
    last_frame_num: int = np.max(list(frame_num_prof_matrix_dict.keys())) + 1
    frame_num_node_idx_cell_id_occupation_list_list_dict: dict = initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_prof_matrix_dict)


    code_validate_track(existing_cell_idx_track_list_dict)
    frame_num_node_idx_cell_id_occupation_list_list_dict = update_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_idx_cell_id_occupation_list_list_dict, existing_cell_idx_track_list_dict)


    while len(to_handle_cell_id_list) != 0:
        handling_cell_id: CellId = to_handle_cell_id_list[0]
        handling_cell_idx: int = handling_cell_id.cell_idx
        print(f"{handling_cell_id.start_frame_num}-{handling_cell_idx}; ", end='')

        start_frame_num: int = handling_cell_id.start_frame_num
        second_frame: int = start_frame_num + 1


        # debug
        if handling_cell_idx in cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id]:
            if len(cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id]) != 0:
                print(len(cell_id_frame_num_node_idx_best_value_list_dict_dict[handling_cell_id]))
                print("handling_cell_idx", handling_cell_idx)
                raise Exception("cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id]) != 0")


        for handling_frame_num in range(second_frame, last_frame_num):
            if  handling_frame_num == second_frame:  last_layer_best_connection_value_list = frame_num_prof_matrix_dict[start_frame_num][handling_cell_idx]
            elif handling_frame_num > second_frame:  last_layer_best_connection_value_list = cell_id_frame_num_node_idx_best_value_list_dict_dict[handling_cell_id][handling_frame_num]

            last_layer_best_connection_value_list = last_layer_best_connection_value_list.reshape(last_layer_best_connection_value_list.shape[0], 1)

            total_cell_in_next_frame: int = frame_num_prof_matrix_dict[handling_frame_num].shape[1]
            last_layer_cell_mtx: np.array = np.repeat(last_layer_best_connection_value_list, total_cell_in_next_frame, axis=1)

            tmp_prof_matrix = frame_num_prof_matrix_dict[handling_frame_num]
            last_layer_all_connection_value_mtx: np.array = last_layer_cell_mtx * tmp_prof_matrix

            adjusted_merge_above_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, strategy_enum , handling_frame_num)


            index_ab_vec, value_ab_vec = derive_last_layer_each_node_best_track(handling_cell_id,
                                                                                last_layer_all_connection_value_mtx,
                                                                                frame_num_prof_matrix_dict,
                                                                                handling_frame_num,
                                                                                frame_num_node_idx_cell_id_occupation_list_list_dict,
                                                                                adjusted_merge_above_threshold,
                                                                                cell_id_frame_num_node_idx_best_value_list_dict_dict,
                                                                                cell_id_track_list_dict,
                                                                                existing_cell_idx_track_list_dict)

            if ( np.all(value_ab_vec == 0) ):
                is_zero_track_length = (handling_frame_num == second_frame)
                if is_zero_track_length:
                    to_skip_cell_id_list.append(handling_cell_id)

                break

            else:
                next_frame_num: int = handling_frame_num + 1
                cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id][next_frame_num] = index_ab_vec
                cell_id_frame_num_node_idx_best_value_list_dict_dict[handling_cell_id][next_frame_num] = value_ab_vec

        to_handle_cell_id_list.remove(handling_cell_id)

        if handling_cell_id not in to_skip_cell_id_list:
            cell_track_list, to_redo_cell_id_list = derive_final_best_track(cell_id_frame_num_node_idx_best_index_list_dict_dict,
                                                                            cell_id_frame_num_node_idx_best_value_list_dict_dict,
                                                                            frame_num_prof_matrix_dict,
                                                                            frame_num_node_idx_cell_id_occupation_list_list_dict,
                                                                            merge_above_threshold,
                                                                            handling_cell_id)

            cell_id_track_list_dict[handling_cell_id] = cell_track_list



            tmp_frame_num = cell_track_list[0][1] + 1
            if handling_cell_id.start_frame_num != tmp_frame_num:
                dev_print("erbt", tmp_frame_num, handling_cell_id.start_frame_num)
                raise Exception("handling_cell_id.start_frame_num != tmp_frame_num")





            for to_redo_cell_id in to_redo_cell_id_list:
                if to_redo_cell_id in existing_cell_idx_track_list_dict:
                    del existing_cell_idx_track_list_dict[to_redo_cell_id]
                elif to_redo_cell_id in cell_id_track_list_dict:
                    del cell_id_track_list_dict[to_redo_cell_id]
                else:
                    raise Exception(to_redo_cell_id)

                del cell_id_frame_num_node_idx_best_index_list_dict_dict[to_redo_cell_id]
                del cell_id_frame_num_node_idx_best_value_list_dict_dict[to_redo_cell_id]

                to_handle_cell_id_list.append(to_redo_cell_id)

            to_handle_cell_id_list.sort(key=cmp_to_key(compare_cell_id))

            frame_num_node_idx_cell_id_occupation_list_list_dict = initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_prof_matrix_dict)
            frame_num_node_idx_cell_id_occupation_list_list_dict = update_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_idx_cell_id_occupation_list_list_dict, existing_cell_idx_track_list_dict)

    print("  --> finish")

    return cell_id_track_list_dict, cell_id_frame_num_node_idx_best_index_list_dict_dict, cell_id_frame_num_node_idx_best_value_list_dict_dict



def save_prof_matrix_to_excel(series: str, frame_num_prof_matrix_dict, excel_output_dir_path: str):
    import pandas as pd
    # num_of_segementation_img: int = len(frame_num_prof_matrix_dict)

    file_name: str = f"series_{series}.xlsx"
    filepath = excel_output_dir_path + file_name;
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter') #pip install xlsxwriter


    # for seg_img_idx in range(0, num_of_segementation_img):
    for frame_num, prof_matrix in frame_num_prof_matrix_dict.items():
        tmp_array: np.arrays = frame_num_prof_matrix_dict[frame_num]

        df = pd.DataFrame (tmp_array)
        sheet_name: str = "frame_1" if frame_num == 1 else str(frame_num+1)
        df.to_excel(writer, sheet_name=sheet_name, index=True)

    writer.save()




def __________unit_function_start_label():
    raise Exception("for labeling only")




def filter_track_dict_by_length(all_track_dict: dict, minimum_track_length: int):
    filtered_track_dict: dict = {}

    for cell_id, track_list in list(all_track_dict.items()):
        if len(track_list) >= minimum_track_length:
            filtered_track_dict[cell_id] = track_list

    return filtered_track_dict



def derive_merge_threshold_in_layer(merge_above_threshold:float, strategy_enum:STRATEGY_ENUM , frame_num:int, frame_num_profit_mtx:dict=None):

    if strategy_enum == STRATEGY_ENUM.ALL_LAYER:
        threshold_exponential: float = float(frame_num - 2)

        merge_threshold_in_layer: float = pow(merge_above_threshold, threshold_exponential)

        return merge_threshold_in_layer

    elif strategy_enum == STRATEGY_ENUM.ONE_LAYER:
        return merge_above_threshold

    else:
        raise Exception(strategy_enum)



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
                            cell_id_frame_num_node_idx_best_value_list_dict_dict: dict,
                            frame_num_prof_matrix_dict: dict,
                            frame_cell_occupation_vec_list_dict: dict,
                            merge_above_threshold: float,
                            handling_cell_id):           # CellId

    frame_num_node_idx_best_index_list_dict: dict = cell_id_frame_num_node_idx_best_index_list_dict_dict[handling_cell_id]
    frame_num_node_idx_best_value_list_dict: dict = cell_id_frame_num_node_idx_best_value_list_dict_dict[handling_cell_id]

    handling_cell_idx: int = handling_cell_id.cell_idx
    cell_track_list: list = []

    last_frame_num: int = np.max(list(frame_num_node_idx_best_value_list_dict.keys()))
    second_frame_num: int = np.min(list(frame_num_node_idx_best_value_list_dict.keys()))

    current_maximize_index: int = None
    current_maximize_value: float = 0
    frame_num_node_idx_best_value_vec: list = frame_num_node_idx_best_value_list_dict[last_frame_num]
    to_redo_cell_id_set: set = set()
    last_frame_adjusted_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, STRATEGY_ENUM.ALL_LAYER , last_frame_num)


    for node_idx, node_probability_value in enumerate(frame_num_node_idx_best_value_vec):

        is_new_value_higher: bool = (node_probability_value > current_maximize_value)

        if not is_new_value_higher:
            continue

        occupied_cell_id_list: tuple = frame_cell_occupation_vec_list_dict[last_frame_num][node_idx]
        has_cell_occupation: bool = ( len(occupied_cell_id_list) != 0 )

        if not has_cell_occupation:
            current_maximize_index = node_idx
            current_maximize_value = node_probability_value

        elif has_cell_occupation:
            for occupied_cell_id in occupied_cell_id_list:
                occupied_cell_idx = occupied_cell_id.cell_idx

                occupied_cell_second_frame_num: int = occupied_cell_id.start_frame_num + 1
                if last_frame_num == occupied_cell_second_frame_num:      occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_id.start_frame_num][occupied_cell_idx][node_idx]
                elif last_frame_num > occupied_cell_second_frame_num:     occupied_cell_probability: float = cell_id_frame_num_node_idx_best_value_list_dict_dict[occupied_cell_id][last_frame_num][node_idx]
                else: raise Exception()

                if node_probability_value > last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    # print(f"let both cell share the same node; {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {node_idx}vs{occupied_cell_idx}")
                    current_maximize_index = node_idx
                    current_maximize_value = node_probability_value

                elif node_probability_value < last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    # may happen in last layer, as viterbi does not check last layer score
                    print(f"no valid cell in last layer due to other cell occupation with higher threshold. Use layer - 1 track; {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {node_idx}vs{occupied_cell_idx}")
                    pass

                elif node_probability_value > last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    print(f"redo trajectory of occupied_cell_idx {occupied_cell_idx}; {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {node_idx}vs{occupied_cell_idx}")
                    start_frame_num: int = 1
                    to_redo_cell_id_set.add(CellId(start_frame_num, occupied_cell_idx))
                    # may not be final track, handle at find track

                    current_maximize_index = node_idx
                    current_maximize_value = node_probability_value

                    # time.sleep(2)

                elif node_probability_value < last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    # print(f"??? have to define what to do (For now, let both cell share the same node ). {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {node_idx}vs{occupied_cell_idx}")

                    current_maximize_index = node_idx
                    current_maximize_value = node_probability_value


                else:
                    print("sdgberbee")
                    print(node_probability_value, occupied_cell_probability, merge_above_threshold)
                    raise Exception("else")


    is_all_tracks_invalid: bool = (current_maximize_index == None)
    # this could happen because last layer is not checked with occupation (but last layer -1 is checked). Therefore, it happens if:
    # 1) probability value is lower than threshold
    # 2) all node is occupied by another cell which has a value higher than threshold
    if is_all_tracks_invalid:
        print("'is_all_tracks_invalid == True' detected, move one layer backward")

        # find the last -1 layer, since as it is checked in previous process it must be valid
        last_layer_max_probability_idx: int = np.argmax(frame_num_node_idx_best_value_vec)
        second_last_layer_max_probability_idx: int = frame_num_node_idx_best_index_list_dict[last_frame_num][last_layer_max_probability_idx]

        last_frame_num -= 1
        current_maximize_index = second_last_layer_max_probability_idx


    handling_cell_second_frame_num: int = handling_cell_id.start_frame_num + 1
    if last_frame_num == handling_cell_second_frame_num:
        previous_maximize_index: int = handling_cell_idx

    elif last_frame_num > handling_cell_second_frame_num:
        previous_maximize_index: int = frame_num_node_idx_best_index_list_dict[last_frame_num][current_maximize_index]

        last_frame_idx: int = last_frame_num - 1
        cell_track_list.append((current_maximize_index, last_frame_idx, previous_maximize_index))

    else:
        raise Exception()





    # check if this is working fine
    for reversed_frame_num in range(last_frame_num-1, second_frame_num-1, -1): #119 to 3 for total_frames = 120

        reversed_frame_idx = reversed_frame_num - 1

        current_maximize_index = previous_maximize_index
        previous_maximize_index = frame_num_node_idx_best_index_list_dict[reversed_frame_num][current_maximize_index]

        cell_track_list.append((current_maximize_index, reversed_frame_idx, previous_maximize_index))

        last_frame_adjusted_threshold: float = derive_merge_threshold_in_layer(merge_above_threshold, STRATEGY_ENUM.ALL_LAYER , reversed_frame_num)

        ### add redo track here
        occupied_cell_id_list: tuple = frame_cell_occupation_vec_list_dict[reversed_frame_num][current_maximize_index]
        has_cell_occupation: bool = ( len(occupied_cell_id_list) != 0 )

        if has_cell_occupation:
            first_frame_num: int = handling_cell_id.start_frame_num
            second_frame: int = first_frame_num + 1

            if reversed_frame_num == second_frame:     handling_cell_probability: float = frame_num_prof_matrix_dict[first_frame_num][handling_cell_idx][current_maximize_index]
            elif reversed_frame_num > second_frame:    handling_cell_probability: float = cell_id_frame_num_node_idx_best_value_list_dict_dict[handling_cell_id][reversed_frame_num][current_maximize_index]


            for occupied_cell_id in occupied_cell_id_list:
                occupied_cell_start_frame_num: int = occupied_cell_id.start_frame_num
                occupied_cell_second_frame: int = occupied_cell_start_frame_num + 1
                if reversed_frame_num == occupied_cell_second_frame:     occupied_cell_probability: float = frame_num_prof_matrix_dict[occupied_cell_start_frame_num][occupied_cell_idx][current_maximize_index]
                elif reversed_frame_num > occupied_cell_second_frame:    occupied_cell_probability: float = cell_id_frame_num_node_idx_best_value_list_dict_dict[occupied_cell_id][reversed_frame_num][current_maximize_index]

                # occupied_cell_probability: float = cell_id_frame_num_node_idx_best_value_list_dict_dict[occupied_cell_id][reversed_frame_num][current_maximize_index]

                if handling_cell_probability > last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    # print(f"let both cell share the same node; {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {node_idx}vs{occupied_cell_idx}")
                    # print("s", end='')
                    pass
                elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    # print(f"handling_cell_probability merge to other cell; {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {node_idx}vs{occupied_cell_idx}")
                    # print("o", end='')
                    pass
                elif handling_cell_probability > last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    print("vwavb", f"redo trajectory of occupied_cell_idx {occupied_cell_id}; {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {current_maximize_index}vs{occupied_cell_id.cell_idx}")
                    # time.sleep(5)
                    # start_frame_num: int = 1
                    to_redo_cell_id_set.add(occupied_cell_id)

                elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    # print(f"??? have to define what to do (For now, let both cell share the same node ). {last_frame_adjusted_threshold}; {np.round(node_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {node_idx}vs{occupied_cell_idx}")
                    # print("s1", end='')
                    pass
                else:
                    print("vebj")
                    print(node_probability_value, occupied_cell_probability, merge_above_threshold)
                    raise Exception("else")


    start_frame_idx: int = handling_cell_id.start_frame_num - 1
    cell_track_list.append((previous_maximize_index, start_frame_idx + 1, handling_cell_idx))
    cell_track_list.append((handling_cell_idx, start_frame_idx, -1))

    list.reverse(cell_track_list)

    return cell_track_list, list(to_redo_cell_id_set)




# after got tracks which started from first frame, check if there are very lower prob between each two cells, then truncate it.
# store_dict, threshold, profit_matrix_list
def _cut_1(cell_idx_track_list_dict: dict, threshold: float, frame_num_prof_matrix_dict: dict):
    short_track_list_dict: dict = {}

    for cell_id, track_content_list in cell_idx_track_list_dict.items():
        short_track_list = []
        for index in range(len(cell_idx_track_list_dict[cell_id]) - 1):
            frame_idx = cell_idx_track_list_dict[cell_id][index][1]
            frame_num: int = frame_idx + 1

            current_node = cell_idx_track_list_dict[cell_id][index][0]
            next_node = cell_idx_track_list_dict[cell_id][index + 1][0]

            weight_between_nodes = frame_num_prof_matrix_dict[frame_num][current_node][next_node]
            if (weight_between_nodes > threshold):
                short_track_list.append(cell_idx_track_list_dict[cell_id][index])
            else:
                short_track_list = copy.deepcopy(cell_idx_track_list_dict[cell_id][0: index])
                break


        if (len(short_track_list) == len(cell_idx_track_list_dict[cell_id])-1):
            tmp = cell_idx_track_list_dict[cell_id][-1]
            short_track_list.append(tmp)
            short_track_list_dict[cell_id] = short_track_list

        else:
            short_track_list.append(cell_idx_track_list_dict[cell_id][len(short_track_list)])
            short_track_list_dict[cell_id] = short_track_list

    return short_track_list_dict



def _initiate_mask(frame_num_prof_matrix_dict: dict):
    mask_frame_cell_id_list: list = []      #list list that stores [frame_id][cell_id]

    # initialize the transition group with all False
    for profit_matrix in frame_num_prof_matrix_dict.values():
        num_of_cell: int = profit_matrix.shape[0]
        mask_frame_cell_id_list.append(np.array([False for i in range(num_of_cell)]))

    mask_frame_cell_id_list.append(np.array([False for i in range(profit_matrix.shape[1])]))

    return mask_frame_cell_id_list



#update the mask matrix based on each track obtained in iteration
def _mask_update(short_Tracks, mask_transition_group):
    # if the node was passed, lable it to True
    for cell_id, vv in short_Tracks.items():
        for iindex in range(len(short_Tracks[cell_id])):
            frame = short_Tracks[cell_id][iindex][1]
            node = short_Tracks[cell_id][iindex][0]
            mask_transition_group[frame][node] = True

    return mask_transition_group



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

    # initiate frame_num_node_idx_occupation_tuple_vec_dict
    for frame_num, profit_matrix in frame_num_prof_matrix_dict.items():
        next_frame_num: int = frame_num + 1
        total_cell_next_frame: int = profit_matrix.shape[1]
        frame_num_node_idx_occupation_tuple_vec_dict[next_frame_num] = [[] for _ in range(total_cell_next_frame)]

    return frame_num_node_idx_occupation_tuple_vec_dict



def update_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id_track_tuple_list_dict: dict):
    for occupied_cell_id, track_tuple_list in cell_id_track_tuple_list_dict.items():
        start_frame_idx: int = track_tuple_list[0][1]
        start_frame_num: int = start_frame_idx + 1

        for track_tuple in track_tuple_list:
            frame_num: int = track_tuple[1] + 1

            if frame_num == start_frame_num:
                continue

            occupied_node_idx: int = track_tuple[0]


            if frame_num <= occupied_cell_id.start_frame_num:
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
                                           last_layer_all_probability_mtx: np.array,
                                           frame_num_prof_matrix_dict: dict,
                                           handling_frame_num: int,
                                           frame_num_node_idx_cell_id_occupation_list_list_dict: dict,
                                           merge_above_threshold: float,
                                           cell_id_frame_num_node_idx_best_value_list_dict_dict: dict,
                                           cell_id_track_list_dict,
                                           existing_cell_idx_track_list_dict):

    handling_cell_idx: int = handling_cell_id.cell_idx
    start_frame_num: int = handling_cell_id.start_frame_num

    total_node_next_frame: int = last_layer_all_probability_mtx.shape[1]
    index_ab_vec: list = [None] * total_node_next_frame
    value_ab_vec: list = [None] * total_node_next_frame
    second_frame_num: int = start_frame_num + 1

    for next_frame_node_idx in range(total_node_next_frame):
        best_idx: int = 0
        best_score: float = 0

        node_connection_score_list = last_layer_all_probability_mtx[:, next_frame_node_idx]
        for node_idx, node_connection_score in enumerate(node_connection_score_list):

            is_new_connection_score_higher: bool = (node_connection_score > best_score)
            if not is_new_connection_score_higher:
                continue

            occupied_cell_id_list: list = frame_num_node_idx_cell_id_occupation_list_list_dict[handling_frame_num][node_idx]
            has_cell_occupation: bool = (len(occupied_cell_id_list) > 0)


            if (not has_cell_occupation) and is_new_connection_score_higher:
                best_idx = node_idx
                best_score = node_connection_score

            elif has_cell_occupation:
                if handling_frame_num == second_frame_num:     handling_cell_probability: float = frame_num_prof_matrix_dict[start_frame_num][handling_cell_idx][node_idx]
                elif handling_frame_num > second_frame_num:    handling_cell_probability: float = cell_id_frame_num_node_idx_best_value_list_dict_dict[handling_cell_id][handling_frame_num][node_idx]
                else: raise Exception(handling_frame_num, second_frame_num)

                for occupied_cell_id in occupied_cell_id_list:
                    occupied_cell_start_frame_num: int = occupied_cell_id.start_frame_num
                    occupied_cell_idx: int = occupied_cell_id.cell_idx
                    occupied_cell_second_frame: int = occupied_cell_id.start_frame_num + 1

                    if handling_frame_num == occupied_cell_second_frame:     occupied_cell_probability_1: float = frame_num_prof_matrix_dict[occupied_cell_start_frame_num][occupied_cell_idx][node_idx]
                    elif handling_frame_num > occupied_cell_second_frame:    occupied_cell_probability_1: float = cell_id_frame_num_node_idx_best_value_list_dict_dict[occupied_cell_id][handling_frame_num][node_idx]
                    else: raise Exception(occupied_cell_id.__str__(), handling_frame_num, occupied_cell_second_frame)

                    if handling_cell_probability > merge_above_threshold and occupied_cell_probability_1 > merge_above_threshold:
                        # print(f"let both cell share the same node; {merge_above_threshold}; {np.round(node_connection_score, 20)}, {np.round(occupied_cell_probability_1, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")
                        if not is_new_connection_score_higher:
                            raise Exception("not is_new_connection_score_higher")

                        best_idx = node_idx
                        best_score = node_connection_score

                    elif handling_cell_probability < merge_above_threshold and occupied_cell_probability_1 > merge_above_threshold:
                        pass
                        # print(f"handling_cell_probability merge to other cell; {merge_above_threshold}; {np.round(node_connection_score, 20)}, {np.round(occupied_cell_probability_1, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")

                    elif handling_cell_probability > merge_above_threshold and occupied_cell_probability_1 < merge_above_threshold:
                        # print(f"redo trajectory of occupied_cell_idx {occupied_cell_idx}; {merge_above_threshold}; {np.round(node_connection_score, 20)}, {np.round(occupied_cell_probability_1, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")
                        # to_redo_cell_idx_set.add(occupied_cell_idx)
                        # may not be final track, handle at find track
                        best_idx = node_idx
                        best_score = node_connection_score


                    elif handling_cell_probability < merge_above_threshold and occupied_cell_probability_1 < merge_above_threshold:
                        # print(f"??? have to define what to do (For now, let both cell share the same node ). {merge_above_threshold}; {np.round(node_connection_score, 20)}, {np.round(occupied_cell_probability_1, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")
                        best_idx = node_idx
                        best_score = node_connection_score


                    else:
                        print("sdgberb")
                        print("handling_cell_id: ", handling_cell_id)
                        print("occupied_cell_id, handling_frame_num, node_idx: ", occupied_cell_id, handling_frame_num, node_idx)
                        print("handling_cell_probability, occupied_cell_probability_1, merge_above_threshold: ", handling_cell_probability, occupied_cell_probability_1, merge_above_threshold)
                        raise Exception("else")


        index_ab_vec[next_frame_node_idx] = best_idx
        value_ab_vec[next_frame_node_idx] = best_score

    return index_ab_vec, np.array(value_ab_vec)



def __________object_start_label():
    raise Exception("for labeling only")



class HyperPara():
    def __init__(self, strategy_enum: STRATEGY_ENUM, merge_threshold: float, minimum_track_length: int, cut_threshold: float):
        self.strategy_enum: STRATEGY_ENUM = strategy_enum
        self.merge_threshold: float = merge_threshold
        self.minimum_track_length: int = minimum_track_length
        self.cut_threshold: float = cut_threshold



    def __eq__(self, other):
        if self.strategy_enum == other.strategy_enum and \
            self.merge_threshold == other.merge_threshold and \
                self.minimum_track_length == other.minimum_track_length and \
                self.cut_threshold == other.cut_threshold:

            return True

        return False

    def __hash__(self):
        return hash((self.strategy_enum, self.merge_threshold, self.minimum_track_length, self.cut_threshold))



class CellId():
    def __init__(self, start_frame_num: int, cell_idx: int):
        self.start_frame_num = start_frame_num
        self.cell_idx = cell_idx

    def __str__(self):
        return f"CellId(start_frame_num: {self.start_frame_num}; cell_idx: {self.cell_idx})"

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