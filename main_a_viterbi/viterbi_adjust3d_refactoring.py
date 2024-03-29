# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:31:51 2021

@author: 13784
"""
import os
from decimal import Decimal
from os import listdir
from os.path import join, basename
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
    images_folder = folder_path + 'dataset//images//'
    output_folder = folder_path + 'output_unet_seg_finetune//'
    save_dir = folder_path + 'save_directory_enhancement/'


    print("start")
    start_time = time.perf_counter()


    input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                         'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    # input_series_list = ['S10']

    #all tracks shorter than DELTA_TIME are false postives and not included in tracks
    result_list = []

    all_segmented_filename_list = listdir(segmentation_folder)
    all_segmented_filename_list.sort()

    existing_series_list = derive_existing_series_list(input_series_list, listdir(output_folder))


    viterbi_result_dict = {}
    is_use_thread: bool = False

    if is_use_thread:
        for input_series in input_series_list:
            viterbi_result_dict[input_series] = []

        pool = ThreadPool(processes=8)
        thread_list: list = []

        for series in existing_series_list:
            print(f"working on series: {series}. ", end="\t")

            async_result = pool.apply_async(cell_tracking_core_flow, (series, segmentation_folder, all_segmented_filename_list, output_folder,)) # tuple of args for foo
            thread_list.append(async_result)

        for thread_idx in range(len(thread_list)):
            final_result_list = thread_list[thread_idx].get()
            viterbi_result_dict[series] = final_result_list
            print(f"Thread {thread_idx} completed")

    else:
        for series in existing_series_list:
            print(f"working on series: {series}", end="\t")
            final_result_list = cell_tracking_core_flow(series, segmentation_folder, all_segmented_filename_list, output_folder)
            viterbi_result_dict[series] = final_result_list


    print("save_track_dictionary")
    save_track_dictionary(viterbi_result_dict, save_dir + "viterbi_results_dict_adj3.pkl")


    with open(save_dir + "viterbi_adjust3d_refactoring.txt", 'w') as f:
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





def cell_tracking_core_flow(series: str, segmentation_folder: str, all_segmented_filename_list: list, output_folder: str, is_create_excel:bool=False):

    segmented_filename_list: list = derive_segmented_filename_list_by_series(series, all_segmented_filename_list)

    prof_mat_list: list = deprecate_derive_prof_matrix_list(segmentation_folder, output_folder, series, segmented_filename_list)
    frame_num_prof_matrix_dict: dict = derive_frame_num_prof_matrix_dict(segmentation_folder, output_folder, series, segmented_filename_list)

    if is_create_excel:
        save_prof_matrix_to_excel(series, frame_num_prof_matrix_dict, excel_output_dir_path="d:/tmp/")

    start_frame_num: int = 1
    all_track_dict = execute_cell_tracking_task(prof_mat_list, frame_num_prof_matrix_dict, start_frame_num)


    count_dict = {}
    for key, value_list in all_track_dict.items():
        seq_length = len(value_list)
        if seq_length not in count_dict:
            count_dict[seq_length] = 0

        count_dict[seq_length] += 1


    result_list = []
    for i in range(len(all_track_dict)):
        if i not in all_track_dict.keys():
            raise Exception("if i not in all_track_dict.keys():")
            continue
        else:
            min_track_length: int = 5
            if (len(all_track_dict[i]) > min_track_length):
                result_list.append(all_track_dict[i])




    # post adjustment
    for j in range(len(result_list) - 1):
        for k in range(j + 1, len(result_list)):
            pre_track_list = result_list[j]
            next_track_list = result_list[k]
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
                result_list[k] = next_track_list
                pre_track_new = copy.deepcopy(pre_track_list[0: index_merge1 + 1])
                result_list[j] = pre_track_new
            else:
                result_list[j] = pre_track_list
                next_track_new = copy.deepcopy(next_track_list[0: index_merge2 + 1])
                result_list[k] = next_track_new

    #print(result)
    final_result_list = []
    for i in range(len(result_list)):
        if ( len(result_list[i]) > 5 ):
            final_result_list.append(result_list[i])

    return final_result_list
    # identifier = series
    # viterbi_result_dict[identifier] = final_result_list



def __________component_function_start_label():
    raise Exception("for labeling only")




#start from first frame and loop the unvisited nodes in the other frames
def execute_cell_tracking_task(profit_matrix_list: list, frame_num_prof_matrix_dict: dict, start_frame_num: int, cut_threshold:float=0.01):
    all_cell_idx_track_list_dict: dict = {}

    cell_idx_track_list_dict: dict = _process_and_find_best_cell_track(frame_num_prof_matrix_dict, start_frame_num) # 2D array list

    cell_idx_short_track_list_dict = _cut_1(cell_idx_track_list_dict, cut_threshold, frame_num_prof_matrix_dict)   # filter out cells that does not make sense (e.g. too low probability)
    all_cell_idx_track_list_dict.update(cell_idx_short_track_list_dict)



    count_dict = {}
    for key, value_tuple_list in all_cell_idx_track_list_dict.items():
        seq_length = len(value_tuple_list)
        if seq_length not in count_dict:
            count_dict[seq_length] = 0

        count_dict[seq_length] += 1


    ##
    ## handle new cells that enter the image
    ##
    max_id_in_dict: int = len(all_cell_idx_track_list_dict)           # dict key is cell idx
    mask_transition_group_mtx_list: list = _mask_1(cell_idx_short_track_list_dict, frame_num_prof_matrix_dict)

    # total_step: int = len(profit_matrix_list)
    last_frame_num: int = np.max(list(frame_num_prof_matrix_dict.keys()))
    for frame_num in range(2, last_frame_num):
        profit_matrix_idx = frame_num - 1
        for cell_row_idx in range(frame_num_prof_matrix_dict[frame_num].shape[0]):  #skip all nodes which are already passed

            is_old_call: bool = (mask_transition_group_mtx_list[profit_matrix_idx][cell_row_idx] == True)

            if is_old_call:
                continue

            new_transition_group_list_ = profit_matrix_list[profit_matrix_idx:]

            new_transition_group_list: list = [new_transition_group_list_[0][cell_row_idx]]
            new_transition_group_list[1:] = profit_matrix_list[profit_matrix_idx + 1:]

            new_transition_group_list[0] = new_transition_group_list[0].reshape(1, new_transition_group_list[0].shape[0])


            # new_store_dict: dict = _process_and_find_best_cell_track(new_transition_group_list)
            next_list_index_vec_list_dict, next_list_value_vec_list_dict = _process_calculate_best_cell_track(new_transition_group_list)

            new_store_dict: dict = {}
            for cell_idx, start_list_value_vec_list in next_list_value_vec_list_dict.items():
                start_list_index_vec_list: list = next_list_index_vec_list_dict[cell_idx]
                track_list: list = _find_iter_one_track_version1(start_list_index_vec_list, start_list_value_vec_list, profit_matrix_idx, cell_row_idx)
                new_store_dict[cell_idx] = track_list


            new_short_Tracks = _cut_iter(new_store_dict, cut_threshold, new_transition_group_list, profit_matrix_idx)

            mask_transition_group_mtx_list = _mask_update(new_short_Tracks, mask_transition_group_mtx_list)
            for ke, val in new_short_Tracks.items():
                all_cell_idx_track_list_dict[max_id_in_dict + ke + 1] = val

            max_id_in_dict = len(all_cell_idx_track_list_dict)

    return all_cell_idx_track_list_dict



#loop each node on first frame to find the optimal path using probabilty multiply
def _process_and_find_best_cell_track(frame_num_prof_matrix_dict: dict, start_frame_num: int, merge_above_threshold:Decimal=Decimal(0.5)):
    cell_idx_track_list_dict: dict = {}
    # cell_id_track_list_dict: dict = {}

    cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict: dict = defaultdict(dict)
    cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict: dict = defaultdict(dict)

    #loop each row on first prob matrix. return the maximum value and index through all the frames, the first prob matrix in profit_matrix_list is a matrix (2D array)
    first_frame_mtx: np.array = frame_num_prof_matrix_dict[start_frame_num]
    total_cell_in_first_frame: int = first_frame_mtx.shape[0]


    to_skip_cell_idx_list: list = []
    last_frame_num: int = np.max(list(frame_num_prof_matrix_dict.keys()))
    total_frame: int = (last_frame_num - start_frame_num) + 1
    frame_cell_occupation_vec_list_dict: dict = derive_frame_num_cell_slot_id_occupation_tuple_vec_dict(frame_num_prof_matrix_dict, cell_idx_track_list_dict)
    to_handle_cell_idx_list: list = [cell_idx for cell_idx in range(0, total_cell_in_first_frame)]

    print(f"handling_cell_idx: ", end='')
    second_frame: int = start_frame_num + 1

    while len(to_handle_cell_idx_list) != 0:
        handling_cell_idx: int = to_handle_cell_idx_list[0]
        print(f"{handling_cell_idx}, ", end='')

        # debug
        if handling_cell_idx in cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[handling_cell_idx]:
            if len(cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[handling_cell_idx]) != 0:
                print(len(cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx]))
                print("handling_cell_idx", handling_cell_idx)
                raise Exception("cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[handling_cell_idx]) != 0")


        for frame_num in range(2, total_frame):
            start_list_value_idx: int = frame_num

            if  frame_num == second_frame:  last_layer_cell_vec = first_frame_mtx[handling_cell_idx]
            elif frame_num > second_frame:  last_layer_cell_vec = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][start_list_value_idx]

            last_layer_cell_vec = last_layer_cell_vec.reshape(last_layer_cell_vec.shape[0], 1)

            total_cell_in_next_frame: int = frame_num_prof_matrix_dict[frame_num].shape[1]
            last_layer_cell_mtx: np.array = np.repeat(last_layer_cell_vec, total_cell_in_next_frame, axis=1)

            last_layer_all_probability_mtx: np.array = last_layer_cell_mtx * frame_num_prof_matrix_dict[frame_num]


            # index_ab_vec = np.argmax(last_layer_all_probability_mtx, axis=0)
            # value_ab_vec = np.max(last_layer_all_probability_mtx, axis=0)
            adjusted_merge_above_threshold: Decimal = Decimal(merge_above_threshold) ** Decimal(frame_num)
            index_ab_vec, value_ab_vec = derive_best_track_of_all_cell_slots(handling_cell_idx,
                                                                             last_layer_all_probability_mtx,
                                                                             frame_num_prof_matrix_dict,
                                                                             frame_num,
                                                                             frame_cell_occupation_vec_list_dict,
                                                                             adjusted_merge_above_threshold,
                                                                             cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict,
                                                                             start_frame_num)


            if ( np.all(value_ab_vec == 0) ):
                to_skip_cell_idx_list.append(handling_cell_idx)
                break

            else:
                next_frame_num: int = frame_num + 1
                cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[handling_cell_idx][next_frame_num] = index_ab_vec
                cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][next_frame_num] = value_ab_vec

        to_handle_cell_idx_list.remove(handling_cell_idx)


        if handling_cell_idx not in to_skip_cell_idx_list:
            start_frame_idx: int = 0

            cell_track_list, to_redo_cell_idx_list = derive_cell_idx_best_track(cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[handling_cell_idx],
                                                                               cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx],
                                                                               cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict,
                                                                               frame_cell_occupation_vec_list_dict,
                                                                               merge_above_threshold,
                                                                               start_frame_idx,
                                                                               handling_cell_idx)

            cell_idx_track_list_dict[handling_cell_idx] = cell_track_list

            for to_redo_cell_idx in to_redo_cell_idx_list:
                del cell_idx_track_list_dict[to_redo_cell_idx]
                del cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[to_redo_cell_idx]
                del cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[to_redo_cell_idx]
                to_handle_cell_idx_list.append(to_redo_cell_idx)

            to_handle_cell_idx_list.sort()

            frame_cell_occupation_vec_list_dict = derive_frame_num_cell_slot_id_occupation_tuple_vec_dict(frame_num_prof_matrix_dict, cell_idx_track_list_dict)

    print("  --> finish")

    return cell_idx_track_list_dict



#loop each node on first frame to find the optimal path using probabilty multiply
def _process_calculate_best_cell_track(profit_mtx_list: list, merge_above_threshold:float=1.0, split_below_threshold:float=1.0):   # former method _process
    # print("_process_1")

    start_list_index_vec_dict: int = defaultdict(list)
    start_list_value_vec_dict: int = defaultdict(list)

    #loop each row on first prob matrix. return the maximum value and index through all the frames, the first prob matrix in profit_matrix_list is a matrix (2D array)
    first_frame_mtx: np.array = profit_mtx_list[0]
    total_cell_in_first_frame: int = first_frame_mtx.shape[0]

    cell_idx_frame_num_tuple_list: list = []
    total_frame: int = len(profit_mtx_list)
    for frame_num in range(1, total_frame):
        for cell_idx in range(0, total_cell_in_first_frame):
            cell_idx_frame_idx_tuple: tuple = (cell_idx, frame_num)
            cell_idx_frame_num_tuple_list.append(cell_idx_frame_idx_tuple)


    to_skip_cell_idx_list: list = []
    for cell_idx_frame_idx_tuple in cell_idx_frame_num_tuple_list:
        cell_idx = cell_idx_frame_idx_tuple[0]
        frame_num = cell_idx_frame_idx_tuple[1]

        if cell_idx in to_skip_cell_idx_list:
            continue

        if frame_num == 1:
            single_cell_vec = first_frame_mtx[cell_idx]
        elif frame_num > 1:
            frame_idx: int = frame_num - 2
            single_cell_vec = start_list_value_vec_dict[cell_idx][frame_idx]
        else:
            raise Exception()


        single_cell_mtx: np.array = single_cell_vec.reshape(single_cell_vec.shape[0], 1)

        total_cell_in_next_frame: int = profit_mtx_list[frame_num].shape[1]
        single_cell_mtx = np.repeat(single_cell_mtx, total_cell_in_next_frame, axis=1)

        last_layer_all_probability_mtx: np.array = single_cell_mtx * profit_mtx_list[frame_num]


        index_ab_vec = np.argmax(last_layer_all_probability_mtx, axis=0)
        # value_ab_vec = np.max(last_layer_all_probability_mtx, axis=0)
        value_ab_vec = derive_matrix_value_by_index_list(last_layer_all_probability_mtx, index_ab_vec)

        if ( np.all(value_ab_vec == 0) ):
            to_skip_cell_idx_list.append(cell_idx)
            continue

        start_list_index_vec_dict[cell_idx].append(index_ab_vec)
        start_list_value_vec_dict[cell_idx].append(value_ab_vec)




    return start_list_index_vec_dict, start_list_value_vec_dict



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



# _find_iter: find the best track start from current frame, and current node based on dict which returned from _process_iter
def derive_cell_idx_best_track(frame_num_cell_slot_idx_best_index_vec_dict: dict,
                               frame_num_cell_slot_idx_best_value_vec_dict: dict,
                               cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict: dict,
                               frame_cell_occupation_vec_list_dict: dict,
                               merge_above_threshold: Decimal,
                               start_frame_idx: int,
                               handling_cell_idx: int):

    cell_track_list: list = []

    last_frame_num: int = np.max(list(frame_num_cell_slot_idx_best_value_vec_dict.keys()))
    second_frame_num: int = np.min(list(frame_num_cell_slot_idx_best_value_vec_dict.keys()))
    total_frame: int = last_frame_num - second_frame_num + 1

    last_frame_idx: int = last_frame_num - 1

    current_maximize_index: int = None
    current_maximize_value: float = 0
    frame_num_cell_slot_idx_best_value_vec: list = frame_num_cell_slot_idx_best_value_vec_dict[last_frame_num]
    to_redo_cell_idx_set: set = set()
    threshold_exponential: float = float(total_frame - 2)
    last_frame_adjusted_threshold: Decimal = Decimal(merge_above_threshold) ** (Decimal(threshold_exponential))

    for cell_slot_idx, cell_slot_probability_value in enumerate(frame_num_cell_slot_idx_best_value_vec):
        is_new_value_higher: bool = (cell_slot_probability_value > current_maximize_value)

        if not is_new_value_higher:
            continue

        occupied_cell_idx_tuple: tuple = frame_cell_occupation_vec_list_dict[last_frame_num][cell_slot_idx]
        has_cell_occupation: bool = ( len(occupied_cell_idx_tuple) != 0 )

        if not has_cell_occupation:
            current_maximize_index = cell_slot_idx
            current_maximize_value = cell_slot_probability_value

        elif has_cell_occupation:
            for occupied_cell_idx in occupied_cell_idx_tuple:
                occupied_cell_probability: float = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[occupied_cell_idx][last_frame_num][cell_slot_idx]

                if cell_slot_probability_value > last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    # print(f"let both cell share the same cell slot; {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}")
                    current_maximize_index = cell_slot_idx
                    current_maximize_value = cell_slot_probability_value

                elif cell_slot_probability_value < last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    print(f"handling_cell_probability merge to other cell; {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}")
                    pass

                elif cell_slot_probability_value > last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    print("w4grw", f"redo trajectory of occupied_cell_idx {occupied_cell_idx}; {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}")
                    to_redo_cell_idx_set.add(occupied_cell_idx)
                    # may not be final track, handle at find track

                    current_maximize_index = cell_slot_idx
                    current_maximize_value = cell_slot_probability_value

                    # time.sleep(2)

                elif cell_slot_probability_value < last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    # print(f"??? have to define what to do (For now, let both cell share the same cell slot ). {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}")

                    current_maximize_index = cell_slot_idx
                    current_maximize_value = cell_slot_probability_value


                else:
                    print("sdgberbee")
                    print(cell_slot_probability_value, occupied_cell_probability, merge_above_threshold)
                    raise Exception("else")

    previous_maximize_index: int = frame_num_cell_slot_idx_best_index_vec_dict[last_frame_num][current_maximize_index]

    cell_track_list.append((current_maximize_index, last_frame_idx, previous_maximize_index))


    # check if this is working fine
    for reversed_frame_num in range(last_frame_num-1, second_frame_num-1, -1): #119 to 3 for total_frames = 120
        reversed_frame_idx = reversed_frame_num - 1

        current_maximize_index = previous_maximize_index
        previous_maximize_index = frame_num_cell_slot_idx_best_index_vec_dict[reversed_frame_num][current_maximize_index]

        cell_track_list.append((current_maximize_index, reversed_frame_idx, previous_maximize_index))


        threshold_exponential = float(reversed_frame_num - 2)
        last_frame_adjusted_threshold = Decimal(merge_above_threshold) ** (Decimal(threshold_exponential))


        ### add redo track here
        occupied_cell_idx_tuple: tuple = frame_cell_occupation_vec_list_dict[reversed_frame_num][current_maximize_index]
        has_cell_occupation: bool = ( len(occupied_cell_idx_tuple) != 0 )

        if has_cell_occupation:
            for occupied_cell_idx in occupied_cell_idx_tuple:
                occupied_cell_probability: float = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[occupied_cell_idx][reversed_frame_num][current_maximize_index]
                handling_cell_probability: float = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][reversed_frame_num][current_maximize_index]

                if handling_cell_probability > last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    # print(f"let both cell share the same cell slot; {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}")
                    # print("s", end='')
                    pass
                elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability > last_frame_adjusted_threshold:
                    # print(f"handling_cell_probability merge to other cell; {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}")
                    # print("o", end='')
                    pass
                elif handling_cell_probability > last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    print("\nvwavb", f"redo trajectory of occupied_cell_idx {occupied_cell_idx}; {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}; frame_num: {reversed_frame_num}")
                    # time.sleep(5)
                    to_redo_cell_idx_set.add(occupied_cell_idx)
                elif handling_cell_probability < last_frame_adjusted_threshold and occupied_cell_probability < last_frame_adjusted_threshold:
                    # print(f"??? have to define what to do (For now, let both cell share the same cell slot ). {last_frame_adjusted_threshold}; {np.round(cell_slot_probability_value, 20)}, {np.round(occupied_cell_probability, 20)} ; {cell_slot_idx}vs{occupied_cell_idx}")
                    # print("s1", end='')
                    pass
                else:
                    print("vebj")
                    print(cell_slot_probability_value, occupied_cell_probability, merge_above_threshold)
                    raise Exception("else")



    # if len(frame_num_cell_slot_idx_best_value_vec_dict) > 1:
    cell_track_list.append((previous_maximize_index, start_frame_idx + 1, handling_cell_idx))
    cell_track_list.append((handling_cell_idx, start_frame_idx, -1))

    # elif len(frame_num_cell_slot_idx_best_value_vec_dict) == 1:
    #     cell_track_list.append((previous_maximize_index, start_frame_idx + 1, track_cell_idx))
    #     cell_track_list.append((track_cell_idx, start_frame_idx, -1))


    list.reverse(cell_track_list)

    return cell_track_list, list(to_redo_cell_idx_set)



#find the best track start from current frame, and current node based on dict which returned from _process_iter
def _find_iter_one_track_version1(start_list_index_vec_list: list, start_list_value_vec_list: list, start_frame_idx: int, track_idx: int):

    track_data_list: list = []

    current_step = len(start_list_value_vec_list) - 1

    current_maximize_index = np.argmax(start_list_value_vec_list[current_step])
    previous_maximize_index = start_list_index_vec_list[current_step][current_maximize_index]

    frame_num: int = start_frame_idx + current_step + 2
    track_data_list.append((current_maximize_index, frame_num, previous_maximize_index))

    for frame_idx in range(len(start_list_value_vec_list)-1):
        current_maximize_index_ = previous_maximize_index
        previous_maximize_index_1 = start_list_index_vec_list[current_step - frame_idx - 1][current_maximize_index_]
        track_data_list.append((current_maximize_index_, start_frame_idx + current_step + 1 - frame_idx, previous_maximize_index_1))
        previous_maximize_index = previous_maximize_index_1


    if len(start_list_value_vec_list) > 1:
        track_data_list.append((previous_maximize_index_1, start_frame_idx + 1, track_idx))
        track_data_list.append((track_idx, start_frame_idx + 0, -1))
    elif len(start_list_value_vec_list) == 1:
        track_data_list.append((previous_maximize_index, start_frame_idx + 1, track_idx))
        track_data_list.append((track_idx, start_frame_idx + 0, -1))
    else:
        raise Exception()

    # for values in store_dict.values():
    list.reverse(track_data_list)

    return track_data_list



# truncate the long tracks
def _cut_iter(longTracks, Threshold, transition_group, start_frame):
    short_Tracks = {}

    for key, track in enumerate(longTracks):
        short_tracks = []
        for index in range(len(longTracks[key])-1):
            current_frame = longTracks[key][index][1]
            next_frame = longTracks[key][index+1][1]
            #find the correct frame and ID of the nodes on tracks, and find the corresponded prob on transition_group matrix
            if (index == 0):
                current_node = 0
                #print(transition_group[current_frame - start_frame].shape)
                transition_group[current_frame - start_frame] = transition_group[current_frame - start_frame].reshape(1,-1)
            else:
                current_node = longTracks[key][index][0]
            next_node = longTracks[key][index+1][0]
            weight_between_nodes = transition_group[current_frame - start_frame][current_node][next_node]
            #if prob > Threshold, add each node of each track, otherwise, copy all the former nodes which are before the first lower_threshold value into a short track
            if (weight_between_nodes > Threshold):
                short_tracks.append(longTracks[key][index])
            else:
                short_tracks = copy.deepcopy(longTracks[key][0:index])
                break
        #add the final node into tracks
        if (len(short_tracks)==len(longTracks[key])-1):
            short_tracks.append(longTracks[key][-1])
            short_Tracks[key] = short_tracks
        else:
            short_tracks.append(longTracks[key][len(short_tracks)])
            short_Tracks[key] = short_tracks
    return short_Tracks



#after got all tracks start from first frame, define a mask matrix. all the nodes which are passed by any tracks are labels as True
def _mask(short_Tracks, transition_group):
    mask_transition_group = []
    # initialize the transition group with all False
    for prob_mat in transition_group:
        mask_transition_group.append(np.array([False for i in range(prob_mat.shape[0])]))
    mask_transition_group.append(np.array([False for i in range(prob_mat.shape[1])]))
    # if the node was passed, lable it to True
    for kk, vv in short_Tracks.items():
        for iindex in range(len(short_Tracks[kk])):
            frame = short_Tracks[kk][iindex][1]
            node = short_Tracks[kk][iindex][0]
            mask_transition_group[frame][node] = True

    return mask_transition_group



#update the mask matrix based on each track obtained in iteration
def _mask_update(short_Tracks, mask_transition_group):
    # if the node was passed, lable it to True
    for kk, vv in short_Tracks.items():
        for iindex in range(len(short_Tracks[kk])):
            frame = short_Tracks[kk][iindex][1]
            node = short_Tracks[kk][iindex][0]
            mask_transition_group[frame][node] = True

    return mask_transition_group



def derive_existing_series_list(input_series_list: list, series_dir_list: list):
    existing_series_list:list = []
    for input_series in input_series_list:
        if input_series in series_dir_list:
            existing_series_list.append(input_series)

    return existing_series_list;



def derive_segmented_filename_list_by_series(series: str, segmented_filename_list: list):
    result_segmented_filename_list: list = []

    for segmented_filename in segmented_filename_list:
        if series in segmented_filename:
            result_segmented_filename_list.append(segmented_filename)

    return result_segmented_filename_list



# after got tracks which started from first frame, check if there are very lower prob between each two cells, then truncate it.
# store_dict, threshold, profit_matrix_list
def _cut_1(original_track_dict: dict, threshold: float, frame_num_prof_matrix_dict: dict):
    short_track_list_dict: dict = {}


    is_first_cell_id_not_zero: bool = list(original_track_dict.keys())[0] != 0
    if is_first_cell_id_not_zero:
        for miss_key in range(list(original_track_dict.keys())[0]):
            short_track_list = []
            short_track_list.append((miss_key,0, -1))
            short_track_list_dict[miss_key] = short_track_list


    for cell_id, track_content_list in original_track_dict.items():
        short_track_list = []
        for index in range(len(original_track_dict[cell_id]) - 1):
            frame_idx = original_track_dict[cell_id][index][1]
            frame_num: int = frame_idx + 1

            current_node = original_track_dict[cell_id][index][0]
            next_node = original_track_dict[cell_id][index + 1][0]

            # weight_between_nodes = profit_matrix_list[frame_idx][current_node][next_node]
            weight_between_nodes = frame_num_prof_matrix_dict[frame_num][current_node][next_node]
            if (weight_between_nodes > threshold):
                short_track_list.append(original_track_dict[cell_id][index])
            else:
                short_track_list = copy.deepcopy(original_track_dict[cell_id][0: index])
                break


        if (len(short_track_list) == len(original_track_dict[cell_id])-1 ):
            tmp = original_track_dict[cell_id][-1]
            short_track_list.append(tmp)
            short_track_list_dict[cell_id] = short_track_list

        else:
            short_track_list.append(original_track_dict[cell_id][len(short_track_list)])
            short_track_list_dict[cell_id] = short_track_list

    return short_track_list_dict




#after got all tracks start from first frame, define a mask matrix. all the nodes which are passed by any tracks are labels as True
def _mask_1(short_track_list_dict: dict, frame_num_prof_matrix_dict: dict):
    # print("_mask_1")
    mask_frame_cell_id_list: list = []      #list list that stores [frame_id][cell_id]

    # initialize the transition group with all False
    for profit_matrix in frame_num_prof_matrix_dict.values():
    # for profit_matrix in profit_matrix_list:
        num_of_cell: int = profit_matrix.shape[0]
        mask_frame_cell_id_list.append(np.array([False for i in range(num_of_cell)]))

    mask_frame_cell_id_list.append(np.array([False for i in range(profit_matrix.shape[1])]))

    # if the cell_id was passed, lable it to True
    for short_track_cell_idx in short_track_list_dict.keys():
        for short_track_frame_idx in range(len(short_track_list_dict[short_track_cell_idx])):
            cell_idx_data = short_track_list_dict[short_track_cell_idx][short_track_frame_idx][0]
            frame_idx_data = short_track_list_dict[short_track_cell_idx][short_track_frame_idx][1]

            mask_frame_cell_id_list[frame_idx_data][cell_idx_data] = True

    return mask_frame_cell_id_list



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



def derive_frame_num_cell_slot_id_occupation_tuple_vec_dict(frame_num_prof_matrix_dict: dict, track_tuple_list_dict: dict):
    frame_num_cell_slot_idx_occupation_tuple_vec_dict: dict = {}

    # initiate frame_num_cell_slot_idx_occupation_tuple_vec_dict
    for frame_num, profit_matrix in frame_num_prof_matrix_dict.items():
        next_frame_num: int = frame_num + 1
        total_cell: int = profit_matrix.shape[1]
        frame_num_cell_slot_idx_occupation_tuple_vec_dict[next_frame_num] = [()] * total_cell


    # assign True to cell that is occupied
    for cell_idx, track_tuple_list in track_tuple_list_dict.items():
        for track_idx, track_tuple in enumerate(track_tuple_list):
            frame_num: int = track_tuple[1] + 1

            if frame_num == 1:
                continue

            occupied_cell_slot_idx: int = track_tuple[0]

            frame_num_cell_slot_idx_occupation_tuple_vec_dict[frame_num][occupied_cell_slot_idx] += (cell_idx,)

    return frame_num_cell_slot_idx_occupation_tuple_vec_dict



def deprecate_derive_prof_matrix_list(segmentation_folder_path: str, output_folder_path: str, series: str, segmented_filename_list):
    prof_mat_list = []

    #get the first image (frame 0) and label the cells:
    img = plt.imread(segmentation_folder_path + segmented_filename_list[0])

    label_img = measure.label(img, background=0, connectivity=1)
    cellnb_img = np.max(label_img)

    for frame_num in range(1, len(segmented_filename_list)):
        # for frame_num in range(0, len(segmented_filename_list)):
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
        # for frame_num in range(0, len(segmented_filename_list)):
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



def derive_best_track_of_all_cell_slots(handling_cell_idx: int,
                                        last_layer_all_probability_mtx: np.array,
                                        frame_num_prof_matrix_dict: dict,
                                        handling_frame_num: int,
                                        frame_num_cell_slot_idx_occupation_tuple_list_dict: dict,
                                        merge_above_threshold: float,
                                        cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict: dict,
                                        start_frame_num: int):

    total_cell_slot_next_frame: int = last_layer_all_probability_mtx.shape[1]
    index_ab_vec: list = [None] * total_cell_slot_next_frame
    value_ab_vec: list = [None] * total_cell_slot_next_frame
    second_frame: int = start_frame_num + 1

    for next_frame_cell_slot_idx in range(total_cell_slot_next_frame):
        best_idx: int = 0
        best_score: float = 0

        slot_connection_score_list = last_layer_all_probability_mtx[:, next_frame_cell_slot_idx]
        for cell_slot_idx, slot_connection_score in enumerate(slot_connection_score_list):
            is_new_connection_score_higher: bool = (slot_connection_score > best_score)
            if not is_new_connection_score_higher:
                continue

            occupied_cell_idx_tuple: tuple = frame_num_cell_slot_idx_occupation_tuple_list_dict[handling_frame_num][cell_slot_idx]
            has_cell_occupation: bool = (len(occupied_cell_idx_tuple) > 0)

            if (not has_cell_occupation) and is_new_connection_score_higher:
                best_idx = cell_slot_idx
                best_score = slot_connection_score

            elif has_cell_occupation:
                # start_list_value_idx: int = handling_frame_num - 3
                start_list_value_idx: int = handling_frame_num


                if handling_frame_num == second_frame:     handling_cell_probability: float = frame_num_prof_matrix_dict[start_frame_num][handling_cell_idx][cell_slot_idx]
                elif handling_frame_num > second_frame:    handling_cell_probability: float = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][start_list_value_idx][cell_slot_idx]

                for occupied_cell_idx in occupied_cell_idx_tuple:
                    if handling_frame_num == second_frame:     occupied_cell_probability: float = frame_num_prof_matrix_dict[start_frame_num][occupied_cell_idx][cell_slot_idx]
                    elif handling_frame_num > second_frame:    occupied_cell_probability: float = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[occupied_cell_idx][start_list_value_idx][cell_slot_idx]



                    if handling_cell_probability > merge_above_threshold and occupied_cell_probability > merge_above_threshold:
                        # print(f"let both cell share the same cell slot; {merge_above_threshold}; {np.round(slot_connection_score, 20)}, {np.round(occupied_cell_probability, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")
                        if not is_new_connection_score_higher:
                            raise Exception("not is_new_connection_score_higher")

                        best_idx = cell_slot_idx
                        best_score = slot_connection_score

                    elif handling_cell_probability < merge_above_threshold and occupied_cell_probability > merge_above_threshold:
                        pass
                        # print(f"handling_cell_probability merge to other cell; {merge_above_threshold}; {np.round(slot_connection_score, 20)}, {np.round(occupied_cell_probability, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")

                    elif handling_cell_probability > merge_above_threshold and occupied_cell_probability < merge_above_threshold:
                        # print(f"redo trajectory of occupied_cell_idx {occupied_cell_idx}; {merge_above_threshold}; {np.round(slot_connection_score, 20)}, {np.round(occupied_cell_probability, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")
                        # to_redo_cell_idx_set.add(occupied_cell_idx)
                        # may not be final track, handle at find track

                        best_idx = cell_slot_idx
                        best_score = slot_connection_score

                        # time.sleep(2)

                    elif handling_cell_probability < merge_above_threshold and occupied_cell_probability < merge_above_threshold:
                        # print(f"??? have to define what to do (For now, let both cell share the same cell slot ). {merge_above_threshold}; {np.round(slot_connection_score, 20)}, {np.round(occupied_cell_probability, 20)} ; {handling_cell_idx}vs{occupied_cell_idx}")

                        best_idx = cell_slot_idx
                        best_score = slot_connection_score

                        # time.sleep(2)

                    else:
                        print("sdgberb")
                        print(handling_cell_probability, occupied_cell_probability, merge_above_threshold)
                        raise Exception("else")


        index_ab_vec[next_frame_cell_slot_idx] = best_idx
        value_ab_vec[next_frame_cell_slot_idx] = best_score

    return index_ab_vec, np.array(value_ab_vec)


def __________object_start_label():
    raise Exception("for labeling only")


class CellId():
    def __init__(self, frame_num: int, cell_idx: int):
        self.frame_num = frame_num
        self.cell_idx = cell_idx




def __________common_function_start_label():
    raise Exception("for labeling only")


def save_track_dictionary(dictionary, save_file):
    if not os.path.exists(save_file):
        with open(save_file, 'w'):
            pass
    pickle_out = open(save_file, "wb")
    pickle.dump(dictionary, pickle_out)
    pickle_out.close()





if __name__ == '__main__':
    main()