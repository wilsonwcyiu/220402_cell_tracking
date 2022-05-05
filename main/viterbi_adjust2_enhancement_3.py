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



#find the best track start from first frame based on dict which returned from _process
def _find(start_list_index, start_list_value):
    store_dict = defaultdict(list)
    for k, v in start_list_value.items():
        #print(f'start from {k}-th sample:')
        current_step = len(v) - 1
        current_maximize_index = np.argmax(v[current_step])
        previous_maximize_index = start_list_index[k][current_step][current_maximize_index]
        store_dict[k].append((current_maximize_index, current_step + 2, previous_maximize_index))
        for i in range(len(v)-1):
            # print(current_maximize_index)
            current_maximize_index_ = previous_maximize_index
            # previous_maximize_index = np.argmax(v[current_step - 1])
            previous_maximize_index_ = start_list_index[k][current_step - i - 1][current_maximize_index_]
            #print(f'current_maximize_index: {current_maximize_index}, '
            #      f'current_step: {current_step}, '
            #      f'previous_maximize_index: {previous_maximize_index}')
            store_dict[k].append((current_maximize_index_, current_step + 1 - i, previous_maximize_index_))
            previous_maximize_index = previous_maximize_index_

        if len(v) != 1:
            store_dict[k].append((previous_maximize_index_, 1, k))
            store_dict[k].append((k, 0, -1))
        else:
            store_dict[k].append((previous_maximize_index, 0, -1))

    for values in store_dict.values():
        values = list.reverse(values)

    return store_dict




# #find the best track start from current frame, and current node based on dict which returned from _process_iter
# def _find_iter_one_track(frame_num_cell_slot_idx_best_index_vec_dict: dict,
#                          frame_num_cell_slot_idx_best_value_vec_dict: dict,
#                          start_frame_idx: int,
#                          track_cell_idx: int):
#
#     track_data_list: list = []
#
#     max_frame_num: int = np.max(list(frame_num_cell_slot_idx_best_value_vec_dict.keys()))
#     min_frame_num: int = np.min(list(frame_num_cell_slot_idx_best_value_vec_dict.keys()))
#     total_frame: int = (max_frame_num - min_frame_num) + 1
#
#     last_frame_num: int = max_frame_num
#     last_frame_idx: int = max_frame_num - 1
#
#     current_maximize_index = np.argmax(frame_num_cell_slot_idx_best_value_vec_dict[last_frame_num])
#     previous_maximize_index = frame_num_cell_slot_idx_best_index_vec_dict[last_frame_num][current_maximize_index]
#
#     track_data_list.append((current_maximize_index, last_frame_idx, previous_maximize_index))
#
#     # for _ in range(total_frame - 1):
#     for reverse_frame_num in range(3, last_frame_num + 1, -1):
#         reverse_frame_idx: int = reverse_frame_num - 1
#
#         current_maximize_index_ = previous_maximize_index
#         last_frame_idx: int = last_frame_idx - 1
#         last_frame_num = last_frame_idx + 1
#         previous_maximize_index_1 = frame_num_cell_slot_idx_best_index_vec_dict[last_frame_num][current_maximize_index_]
#         track_data_list.append((current_maximize_index_, last_frame_idx, previous_maximize_index_1))
#         previous_maximize_index = previous_maximize_index_1
#
#
#     if len(frame_num_cell_slot_idx_best_value_vec_dict) > 1:
#         track_data_list.append((previous_maximize_index_1, start_frame_idx + 1, track_cell_idx))
#         track_data_list.append((track_cell_idx, start_frame_idx + 0, -1))
#
#     elif len(frame_num_cell_slot_idx_best_value_vec_dict) == 1:
#         track_data_list.append((previous_maximize_index, start_frame_idx + 1, track_cell_idx))
#         track_data_list.append((track_cell_idx, start_frame_idx + 0, -1))
#
#
#     # for values in store_dict.values():
#     list.reverse(track_data_list)
#
#     return track_data_list




#find the best track start from current frame, and current node based on dict which returned from _process_iter
def _find_iter_one_track(frame_num_cell_slot_idx_best_index_vec_dict: dict,
                         frame_num_cell_slot_idx_best_value_vec_dict: dict,
                         start_frame_idx: int,
                         track_cell_idx: int):

    track_data_list: list = []

    last_frame_num: int = np.max(list(frame_num_cell_slot_idx_best_value_vec_dict.keys()))
    first_frame_num: int = np.min(list(frame_num_cell_slot_idx_best_value_vec_dict.keys()))

    last_frame_idx: int = last_frame_num - 1




    current_maximize_index: int = np.argmax(frame_num_cell_slot_idx_best_value_vec_dict[last_frame_num])




    previous_maximize_index: int = frame_num_cell_slot_idx_best_index_vec_dict[last_frame_num][current_maximize_index]

    track_data_list.append((current_maximize_index, last_frame_idx, previous_maximize_index))


    for reversed_frame_num in range(last_frame_num-1, first_frame_num-1, -1): #119 to 3
        reversed_frame_idx = reversed_frame_num - 1

        current_maximize_index = previous_maximize_index
        previous_maximize_index = frame_num_cell_slot_idx_best_index_vec_dict[reversed_frame_num][current_maximize_index]

        track_data_list.append((current_maximize_index, reversed_frame_idx, previous_maximize_index))


    if len(frame_num_cell_slot_idx_best_value_vec_dict) > 1:
        track_data_list.append((previous_maximize_index, start_frame_idx + 1, track_cell_idx))
        track_data_list.append((track_cell_idx, start_frame_idx + 0, -1))

    elif len(frame_num_cell_slot_idx_best_value_vec_dict) == 1:
        track_data_list.append((previous_maximize_index, start_frame_idx + 1, track_cell_idx))
        track_data_list.append((track_cell_idx, start_frame_idx + 0, -1))


    list.reverse(track_data_list)

    return track_data_list



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


# #find the best track start from current frame, and current node based on dict which returned from _process_iter
# def _find_iter_one_cell_1(start_list_index_vec_list: list, start_list_value_vec_list: list, start_frame_idx: int, track_idx: int):
#
#     # global previous_maximize_index_1
#     # store_dict = defaultdict(list)
#
#     track_data_list: list = []
#     # for cell_idx, start_list_value_vec_list in start_list_value_vec_dict.items():
#     current_step = len(start_list_value_vec_list) - 1
#
#     current_maximize_index = np.argmax(start_list_value_vec_list[current_step])
#     previous_maximize_index = start_list_index_vec_list[current_step][current_maximize_index]
#
#     frame_num: int = start_frame_idx + current_step + 2
#     track_data_list.append((current_maximize_index, frame_num, previous_maximize_index))
#
#     for frame_idx in range(len(start_list_value_vec_list)-1):
#         current_maximize_index_ = previous_maximize_index
#         previous_maximize_index_1 = start_list_index_vec_list[current_step - frame_idx - 1][current_maximize_index_]
#         track_data_list.append((current_maximize_index_, start_frame_idx + current_step + 1 - frame_idx, previous_maximize_index_1))
#         previous_maximize_index = previous_maximize_index_1
#
#
#     if len(start_list_value_vec_list) > 1:
#         track_data_list.append((previous_maximize_index_1, start_frame_idx + 1, track_idx))
#         track_data_list.append((track_idx, start_frame_idx + 0, -1))
#     elif len(start_list_value_vec_list) == 1:
#         track_data_list.append((previous_maximize_index, start_frame_idx + 1, track_idx))
#         track_data_list.append((track_idx, start_frame_idx + 0, -1))
#     else:
#         raise Exception()
#
#     # for values in store_dict.values():
#     #     list.reverse(values)
#     list.reverse(track_data_list)
#
#     return track_data_list






#find the best track start from current frame, and current node based on dict which returned from _process_iter
def _find_iter_1(start_list_index_vec_dict: dict, start_list_value_vec_dict: dict, start_frame_idx: int, track_idx: int):

    # global previous_maximize_index_1
    store_dict = defaultdict(list)
    for cell_idx, start_list_value_vec_list in start_list_value_vec_dict.items():
        current_step = len(start_list_value_vec_list) - 1

        current_maximize_index = np.argmax(start_list_value_vec_list[current_step])
        previous_maximize_index = start_list_index_vec_dict[cell_idx][current_step][current_maximize_index]

        frame_num: int = start_frame_idx + current_step + 2
        store_dict[cell_idx].append((current_maximize_index, frame_num, previous_maximize_index))

        for frame_idx in range(len(start_list_value_vec_list)-1):
            current_maximize_index_ = previous_maximize_index
            previous_maximize_index_1 = start_list_index_vec_dict[cell_idx][current_step - frame_idx - 1][current_maximize_index_]
            store_dict[cell_idx].append((current_maximize_index_, start_frame_idx + current_step + 1 - frame_idx, previous_maximize_index_1))
            previous_maximize_index = previous_maximize_index_1


        if len(start_list_value_vec_list) > 1:
            store_dict[cell_idx].append((previous_maximize_index_1, start_frame_idx + 1, track_idx))
            store_dict[cell_idx].append((track_idx, start_frame_idx + 0, -1))
        elif len(start_list_value_vec_list) == 1:
            store_dict[cell_idx].append((previous_maximize_index, start_frame_idx + 1, track_idx))
            store_dict[cell_idx].append((track_idx, start_frame_idx + 0, -1))
        else:
            raise Exception()

    for values in store_dict.values():
        list.reverse(values)

    return store_dict



#find the best track start from current frame, and current node based on dict which returned from _process_iter
def _find_iter(start_list_index_vec_dict: dict, start_list_value_vec_dict: dict, start_frame_idx: int, track_idx: int):

    global previous_maximize_index_1
    store_dict = defaultdict(list)
    for k, v in start_list_value_vec_dict.items():
        #print(f'start from {k}-th sample:')
        current_step = len(v) - 1
        current_maximize_index = np.argmax(v[current_step])
        previous_maximize_index = start_list_index_vec_dict[k][current_step][current_maximize_index]
        frame_num: int = start_frame_idx + current_step + 2
        store_dict[k].append((current_maximize_index, frame_num, previous_maximize_index))

        for i in range(len(v)-1):
            # print(current_maximize_index)
            current_maximize_index_ = previous_maximize_index
            # previous_maximize_index = np.argmax(v[current_step - 1])
            previous_maximize_index_1 = start_list_index_vec_dict[k][current_step - i - 1][current_maximize_index_]
            #print(f'current_maximize_index: {current_maximize_index}, '
            #      f'current_step: {current_step}, '
            #      f'previous_maximize_index: {previous_maximize_index}')
            store_dict[k].append((current_maximize_index_, start_frame_idx + current_step + 1 - i, previous_maximize_index_1))
            previous_maximize_index = previous_maximize_index_1
        if len(v)!=1:
            store_dict[k].append((previous_maximize_index_1, start_frame_idx + 1, track_idx))
            store_dict[k].append((track_idx, start_frame_idx + 0, -1))
        else:
            store_dict[k].append((previous_maximize_index, start_frame_idx + 1, track_idx))
            store_dict[k].append((track_idx, start_frame_idx + 0, -1))

    for values in store_dict.values():
        values = list.reverse(values)

    return store_dict



# after got tracks which started from first frame, check if there are very lower prob between each two cells, then truncate it.
def _cut(longTracks, threshold, transition_group):
    short_Tracks = {}

    if list(longTracks.keys())[0] != 0:
        for miss_key in range(list(longTracks.keys())[0]):
            short_tracks = []
            short_tracks.append((miss_key,0,-1))
            short_Tracks[miss_key] = short_tracks

    for key, track in longTracks.items():
        short_tracks = []
        for index in range(len(longTracks[key])-1):
            current_frame = longTracks[key][index][1]
            next_frame = longTracks[key][index+1][1]
            current_node = longTracks[key][index][0]
            next_node = longTracks[key][index+1][0]
            weight_between_nodes = transition_group[current_frame][current_node][next_node]
            if (weight_between_nodes > threshold):
                short_tracks.append(longTracks[key][index])
            else:
                short_tracks = copy.deepcopy(longTracks[key][0:index])
                break
        if (len(short_tracks)==len(longTracks[key])-1):
            short_tracks.append(longTracks[key][-1])
            short_Tracks[key] = short_tracks
        else:
            short_tracks.append(longTracks[key][len(short_tracks)])
            short_Tracks[key] = short_tracks
    return short_Tracks



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



#start from first frame and loop the unvisited nodes in the other frames
def _iteration(transition_group: list):
    all_track_dict = {}

    start_list_index, start_list_value = _process(transition_group) # 2D array list

    store_dict = _find(start_list_index, start_list_value)

    short_Tracks = _cut(store_dict, 0.01, transition_group)   # filter out cells that does not make sense (e.g. too low probability)

    all_track_dict.update(short_Tracks)


    ##
    ## handle new cells that enter the image
    ##
    length = len(all_track_dict)
    mask_transition_group = _mask(short_Tracks, transition_group)
    for p_matrix in range(1, len(transition_group)):
        #print(p_matrix)
        #print(transition_group[p_matrix].shape)
        #print(transition_group[p_matrix].shape[0])
        for node in range(transition_group[p_matrix].shape[0]):  #skip all nodes which are already passed
            #print(node)
            if (mask_transition_group[p_matrix][node] == True):
                continue
            else:
                new_transition_group = []
                new_transition_group_ = transition_group[p_matrix:]
                new_transition_group.append(new_transition_group_[0][node])
                new_transition_group[1:] = transition_group[p_matrix+1:]
                next_list_index, next_list_value = _process_iter(new_transition_group)
                new_store_dict = _find_iter(next_list_index, next_list_value, p_matrix, node)
                new_short_Tracks = _cut_iter(new_store_dict, 0.01, new_transition_group, p_matrix)

            mask_transition_group = _mask_update(new_short_Tracks, mask_transition_group)
            for ke, val in new_short_Tracks.items():
                all_track_dict[length + ke + 1] = val
            length = len(all_track_dict)

    return all_track_dict



#loop each node on first frame to find the optimal path using probabilty multiply
def _process(transition_group: list):
    step = len(transition_group)
    start_list_index = defaultdict(list)
    start_list_value = defaultdict(list)
    #loop each row on first prob matrix. return the maximum value and index through the whole frames
    #the first prob matrix in transition_group is a matrix (2D array)
    for ii, item in enumerate(transition_group[0]):
        for i in range(1, step):
            item = item[:, np.newaxis]   #https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
            item = np.repeat(item, transition_group[i].shape[1], axis=1)
            index_ab = np.argmax(item * transition_group[i], axis=0)
            value_ab = np.max(item * transition_group[i], axis=0)
            if (np.all(value_ab == 0)):
                break

            start_list_index[ii].append(index_ab)
            start_list_value[ii].append(value_ab)
            item = value_ab

    return start_list_index, start_list_value



#loop each node on the other frames which is not passed by the tracks we've got.
def _process_iter(transition_group):
    step = len(transition_group)
    start_list_index = defaultdict(list)
    start_list_value = defaultdict(list)
    #transition_group[0] is the node, shape is (n,), convert it to (1,n)
    #the first prob matrix in transition_group is a vector (1D array)

    # print("shape", transition_group[0].shape)
    for ii, item in enumerate(transition_group[0].reshape(1, transition_group[0].shape[0])):
        # print("shape1", transition_group[0].reshape(1, transition_group[0].shape[0]))
        for i in range(1, step):
            item = item[:, np.newaxis]
            item = np.repeat(item, transition_group[i].shape[1], 1)
            index_ab = np.argmax(item * transition_group[i], 0)
            value_ab = np.max(item * transition_group[i], 0)
            if (np.all(value_ab==0)):
                break
            start_list_index[ii].append(index_ab)
            start_list_value[ii].append(value_ab)
            item = value_ab
    return start_list_index, start_list_value



'''save track dictionary'''
def save_track_dictionary(dictionary, save_file):
    if not os.path.exists(save_file):
        with open(save_file, 'w'):
            pass
    pickle_out = open(save_file,"wb")
    pickle.dump(dictionary,pickle_out)
    pickle_out.close()



def find_existing_series_list(input_series_list: list, series_dir_list: list):
    existing_series_list:list = []
    for input_series in input_series_list:
        if input_series in series_dir_list:
            existing_series_list.append(input_series)

    return existing_series_list;



def find_segmented_filename_list_by_series(series: str, segmented_filename_list: list):
    result_segmented_filename_list: list = []

    for segmented_filename in segmented_filename_list:
        if series in segmented_filename:
            result_segmented_filename_list.append(segmented_filename)

    return result_segmented_filename_list



def derive_prof_matrix_list(segmentation_folder_path: str, output_folder_path: str, series: str, segmented_filename_list):
    prof_mat_list = []

    #get the first image (frame 0) and label the cells:
    img = plt.imread(segmentation_folder_path + segmented_filename_list[0])

    label_img = measure.label(img, background=0, connectivity=1)
    cellnb_img = np.max(label_img)

    # print(series, len(segmented_filename_list))

    # bug fix? need output file     #FileNotFoundError: [Errno 2] No such file or directory: 'D:/viterbi linkage/dataset/output_unet_seg_finetune//S01/mother_190621_++1_S01_frame001_Cell01.png'
    # it is actually a prediction

    for framenb in range(1, len(segmented_filename_list)):
        # for framenb in range(0, len(segmented_filename_list)):
        #get next frame and number of cells next frame
        img_next = plt.imread(segmentation_folder_path + '/' + segmented_filename_list[framenb])

        label_img_next = measure.label(img_next, background=0, connectivity=1)
        cellnb_img_next = np.max(label_img_next)

        #create empty dataframe for element of profit matrix C
        prof_mat = np.zeros( (cellnb_img, cellnb_img_next), dtype=float)

        #loop through all combinations of cells in this and the next frame
        for cellnb_i in range(cellnb_img):
            #cellnb i + 1 because cellnumbering in output files starts from 1
            cell_i_filename = "mother_" + segmented_filename_list[framenb][:-4] + "_Cell" + str(cellnb_i + 1).zfill(2) + ".png"
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



def create_prof_matrix_excel(series: str, one_series_img_list, excel_output_dir_path: str):
    import pandas as pd
    # existing_series_list: list = all_prof_mat_list_dict.keys()
    # for series in existing_series_list:
    # one_series_img_list: list = all_prof_mat_list_dict[series]
    num_of_segementation_img: int = len(one_series_img_list)

    file_name: str = f"series_{series}.xlsx"
    filepath = excel_output_dir_path + file_name;
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter') #pip install xlsxwriter


    for seg_img_idx in range(0, num_of_segementation_img):
        tmp_array: np.arrays = one_series_img_list[seg_img_idx]

        df = pd.DataFrame (tmp_array)
        sheet_name: str = "frame_1" if seg_img_idx == 0 else str(seg_img_idx+1)
        df.to_excel(writer, sheet_name=sheet_name, index=True)

    writer.save()



def derive_frame_num_cell_slot_id_occupation_tuple_vec_dict(profit_matrix_list: np.array, track_tuple_list_dict: dict):
    frame_num_cell_slot_idx_occupation_tuple_vec_dict: dict = {}

    # initiate frame_num_cell_slot_idx_occupation_tuple_vec_dict
    for idx, profit_matrix in enumerate(profit_matrix_list):
        frame_num: int = idx + 2
        total_cell: int = profit_matrix.shape[1]
        frame_num_cell_slot_idx_occupation_tuple_vec_dict[frame_num] = [()] * total_cell


    # assign True to cell that is occupied
    for cell_idx, track_tuple_list in track_tuple_list_dict.items():
        for track_idx, track_tuple in enumerate(track_tuple_list):
            frame_num: int = track_tuple[1] + 1

            # if frame_num > 120:
            #     print("debug")

            if frame_num == 1:
                continue

            occupied_cell_slot_idx: int = track_tuple[0]

            # syntax to add cell_idx to tuple
            frame_num_cell_slot_idx_occupation_tuple_vec_dict[frame_num][occupied_cell_slot_idx] += (cell_idx,)

    return frame_num_cell_slot_idx_occupation_tuple_vec_dict



#start from first frame and loop the unvisited nodes in the other frames
def _iteration_create_viterbi_track_data(profit_matrix_list: list):
    all_cell_track_dict: dict = {}

    # track_index_vec_list_dict, track_value_vec_list_dict = _process_calculate_best_cell_track(profit_matrix_list) # 2D array list
    store_dict: dict = _process_and_find_best_cell_track(profit_matrix_list) # 2D array list

    cut_threshold: float = 0.01
    short_track_list_dict = _cut_1(store_dict, cut_threshold, profit_matrix_list)   # filter out cells that does not make sense (e.g. too low probability)

    all_cell_track_dict.update(short_track_list_dict)


    # return all_cell_track_dict


    count_dict = {}
    for key, value_tuple_list in all_cell_track_dict.items():
        seq_length = len(value_tuple_list)
        if seq_length not in count_dict:
            count_dict[seq_length] = 0

        count_dict[seq_length] += 1



    ##
    ## handle new cells that enter the image
    ##
    max_id_in_dict: int = len(all_cell_track_dict)           # dict key is cell idx
    mask_transition_group_mtx = _mask_1(short_track_list_dict, profit_matrix_list)

    total_step: int = len(profit_matrix_list)
    for profit_matrix_idx in range(1, total_step):
        for cell_row_idx in range(profit_matrix_list[profit_matrix_idx].shape[0]):  #skip all nodes which are already passed

            is_old_call: bool = (mask_transition_group_mtx[profit_matrix_idx][cell_row_idx] == True)

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

            mask_transition_group_mtx = _mask_update(new_short_Tracks, mask_transition_group_mtx)
            for ke, val in new_short_Tracks.items():
                all_cell_track_dict[max_id_in_dict + ke + 1] = val

            max_id_in_dict = len(all_cell_track_dict)



    # print("len(track_index_vec_list_dict), len(store_dict), len(short_track_list_dict), len(all_track_dict)",
    #       len(track_index_vec_list_dict), len(store_dict), len(short_track_list_dict), len(all_track_dict))

    return all_cell_track_dict







#loop each node on first frame to find the optimal path using probabilty multiply
def _process_and_find_best_cell_track(profit_mtx_list: list, merge_above_threshold:Decimal=Decimal(0)):
    store_dict: dict = {}

    cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict: dict = defaultdict(dict)
    cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict: dict = defaultdict(dict)

    #loop each row on first prob matrix. return the maximum value and index through all the frames, the first prob matrix in profit_matrix_list is a matrix (2D array)
    first_frame_mtx: np.array = profit_mtx_list[0]
    total_cell_in_first_frame: int = first_frame_mtx.shape[0]


    to_skip_cell_idx_list: list = []
    total_frame: int = len(profit_mtx_list) + 1
    frame_cell_occupation_vec_list_dict: dict = derive_frame_num_cell_slot_id_occupation_tuple_vec_dict(profit_mtx_list, store_dict)
    to_handle_cell_idx_list: list = [cell_idx for cell_idx in range(0, total_cell_in_first_frame)]

    print(f"handling_cell_idx:", end='')
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


            # start_list_value_idx: int = frame_num - 3
            start_list_value_idx: int = frame_num
            profit_mtx_idx: int = frame_num - 1

            if  frame_num == 2:  last_layer_cell_vec = first_frame_mtx[handling_cell_idx]
            elif frame_num > 2:  last_layer_cell_vec = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][start_list_value_idx]

            last_layer_cell_vec = last_layer_cell_vec.reshape(last_layer_cell_vec.shape[0], 1)

            total_cell_in_next_frame: int = profit_mtx_list[profit_mtx_idx].shape[1]
            last_layer_cell_mtx: np.array = np.repeat(last_layer_cell_vec, total_cell_in_next_frame, axis=1)

            last_layer_all_probability_mtx: np.array = last_layer_cell_mtx * profit_mtx_list[profit_mtx_idx]


            # index_ab_vec = np.argmax(last_layer_all_probability_mtx, axis=0)
            # value_ab_vec = np.max(last_layer_all_probability_mtx, axis=0)
            adjusted_merge_above_threshold: Decimal = Decimal(merge_above_threshold) ** Decimal(frame_num)
            index_ab_vec, value_ab_vec, to_redo_cell_idx_list = find_best_track(handling_cell_idx,
                                                                                   last_layer_all_probability_mtx,
                                                                                   profit_mtx_list,
                                                                                   frame_num,
                                                                                   frame_cell_occupation_vec_list_dict,
                                                                                   adjusted_merge_above_threshold,
                                                                                   cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict)

            # print("after find_best_track")
            # print("index_ab_vec", index_ab_vec)
            # value_ab_vec = obtain_matrix_value_by_index_list(last_layer_all_probability_mtx, index_ab_vec)

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

            track_list: list = _find_iter_one_track(cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[handling_cell_idx],
                                                    cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx],
                                                    start_frame_idx,
                                                    handling_cell_idx)



            # debug
            track_value_list: list = []
            for track_tuple in track_list:
                frame_idx: int = track_tuple[1]

                if frame_idx < 2: continue

                cell_path_idx: int = track_tuple[0]
                frame_num: int = frame_idx + 1
                # start_list_value_idx: int = frame_num - 3
                start_list_value_idx: int = frame_num

                if len(cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx]) > total_frame:
                    print(len(cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx]))
                    print("handling_cell_idx", handling_cell_idx)
                    print("total_frame", total_frame)
                    raise Exception("len(cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx]) > total_frame")

                cell_idx_frame_num_cell_slot_idx_best_value_vec_dict = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx]

                # print("qwf", cell_idx_frame_num_cell_slot_idx_best_value_vec_dict.keys())
                if start_list_value_idx > total_frame:
                    print("debug")
                    raise Exception("debug")


                value = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict[start_list_value_idx][cell_path_idx]
                track_value_list.append(value)
# removed round(20 and see if value<=0 still appears)

                if value <= 0:
                    print(value <= 0)
                    print(len(track_value_list))
                    print(track_value_list)
                    print("handling_cell_idx", handling_cell_idx)
                    print("track_tuple", track_tuple)
                    print("cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][start_list_value_idx]", cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][start_list_value_idx])

                    time.sleep(2)
                    raise Exception("value <= 0")
                    # inspect why 0 value is being chosen


            # print(f"{handling_cell_idx}: {track_value_list}")
            # time.sleep(2)





            store_dict[handling_cell_idx] = track_list

            # to_redo_trajectory_cell_idx_list = derive_to_redo_track_list()
            for to_redo_cell_idx in to_redo_cell_idx_list:
                del store_dict[to_redo_cell_idx]
                del cell_idx_frame_num_cell_slot_idx_best_index_vec_dict_dict[to_redo_cell_idx]
                del cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[to_redo_cell_idx]
                to_handle_cell_idx_list.append(to_redo_cell_idx)

                # print("to_redo_cell_idx", to_redo_cell_idx)

            to_handle_cell_idx_list.sort()

            frame_cell_occupation_vec_list_dict = derive_frame_num_cell_slot_id_occupation_tuple_vec_dict(profit_mtx_list, store_dict)

    print("  --> finish")

    return store_dict



def derive_to_redo_track_list():
    return []

#
# #loop each node on first frame to find the optimal path using probabilty multiply
# def _process_and_find_best_cell_track(profit_mtx_list: list, merge_above_threshold:float=1.0, split_below_threshold:float=1.0):   # former method _process
#     # print("_process_1")
#     store_dict: dict = {}
#
#     start_list_index_vec_list_dict: int = defaultdict(list)
#     start_list_value_vec_list_dict: int = defaultdict(list)
#
#     #loop each row on first prob matrix. return the maximum value and index through all the frames, the first prob matrix in profit_matrix_list is a matrix (2D array)
#     first_frame_mtx: np.array = profit_mtx_list[0]
#     total_cell_in_first_frame: int = first_frame_mtx.shape[0]
#
#
#
#     to_skip_cell_idx_list: list = []
#     total_frame: int = len(profit_mtx_list)
#     frame_cell_occupation_vec_list_dict: dict = derive_frame_cell_occupation_vec_list(profit_mtx_list, store_dict)
#     for cell_idx in range(0, total_cell_in_first_frame):
#         for frame_num in range(1, total_frame):
#
#             if frame_num == 1:
#                 single_cell_vec = first_frame_mtx[cell_idx]
#             elif frame_num > 1:
#                 frame_idx: int = frame_num - 2
#                 single_cell_vec = start_list_value_vec_list_dict[cell_idx][frame_idx]
#             else:
#                 raise Exception()
#
#
#             single_cell_mtx: np.array = single_cell_vec.reshape(single_cell_vec.shape[0], 1)
#
#             total_cell_in_next_frame: int = profit_mtx_list[frame_num].shape[1]
#             single_cell_mtx = np.repeat(single_cell_mtx, total_cell_in_next_frame, axis=1)
#
#             last_layer_all_probability_mtx: np.array = single_cell_mtx * profit_mtx_list[frame_num]
#
#
#             index_ab_vec = np.argmax(last_layer_all_probability_mtx, axis=0)
#             # value_ab_vec = np.max(last_layer_all_probability_mtx, axis=0)
#             value_ab_vec = obtain_matrix_value_by_index_list(last_layer_all_probability_mtx, index_ab_vec)
#
#             if ( np.all(value_ab_vec == 0) ):
#                 to_skip_cell_idx_list.append(cell_idx)
#                 break
#
#             start_list_index_vec_list_dict[cell_idx].append(index_ab_vec)
#             start_list_value_vec_list_dict[cell_idx].append(value_ab_vec)
#
#
#         if cell_idx not in to_skip_cell_idx_list:
#             start_frame_idx: int = 0
#             track_list: list = _find_iter_one_track(start_list_index_vec_list_dict[cell_idx], start_list_value_vec_list_dict[cell_idx], start_frame_idx, cell_idx)
#             store_dict[cell_idx] = track_list
#
#             to_remove_trajectory_cell_idx_list = [] # to be completed
#             for to_remove_trajectory_cell_idx in to_remove_trajectory_cell_idx_list:
#                 if to_remove_trajectory_cell_idx in store_dict:
#                     del store_dict[to_remove_trajectory_cell_idx]
#
#             frame_cell_occupation_vec_list_dict = derive_frame_cell_occupation_vec_list(profit_mtx_list, store_dict)
#
#     return store_dict




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
        value_ab_vec = obtain_matrix_value_by_index_list(last_layer_all_probability_mtx, index_ab_vec)

        if ( np.all(value_ab_vec == 0) ):
            to_skip_cell_idx_list.append(cell_idx)
            continue

        start_list_index_vec_dict[cell_idx].append(index_ab_vec)
        start_list_value_vec_dict[cell_idx].append(value_ab_vec)




    return start_list_index_vec_dict, start_list_value_vec_dict









# #loop each node on first frame to find the optimal path using probabilty multiply
# def _process_and_find_calculate_best_cell_track(profit_mtx_list: list, merge_above_threshold:float=1.0, split_below_threshold:float=1.0):   # former method _process
#     # print("_process_1")
#     store_dict = defaultdict(list)
#
#     start_list_index_vec_dict: int = defaultdict(list)
#     start_list_value_vec_dict: int = defaultdict(list)
#
#     #loop each row on first prob matrix. return the maximum value and index through all the frames, the first prob matrix in profit_matrix_list is a matrix (2D array)
#     first_frame_mtx: np.array = profit_mtx_list[0]
#     total_cell_in_first_frame: int = first_frame_mtx.shape[0]
#
#     cell_idx_frame_num_tuple_list: list = []
#     total_frame: int = len(profit_mtx_list)
#     for cell_idx in range(0, total_cell_in_first_frame):
#         for frame_num in range(1, total_frame):
#             cell_idx_frame_idx_tuple: tuple = (cell_idx, frame_num)
#             cell_idx_frame_num_tuple_list.append(cell_idx_frame_idx_tuple)
#
#
#
#     frame_num_cell_idx_occupation_vec_dict: dict = {}
#     for frame_num in range(1, total_frame):
#         total_cell_in_frame_num: int = profit_mtx_list[frame_num].shape[0]
#         frame_num_cell_idx_occupation_vec_dict[frame_num] = [False for idx in range(0, total_cell_in_frame_num)]
#
#
#     to_skip_cell_idx_list: list = []
#     for cell_idx_frame_idx_tuple in cell_idx_frame_num_tuple_list:
#         cell_idx = cell_idx_frame_idx_tuple[0]
#         frame_num = cell_idx_frame_idx_tuple[1]
#
#         if cell_idx in to_skip_cell_idx_list:
#             continue
#
#         if frame_num == 1:
#             single_cell_vec = first_frame_mtx[cell_idx]
#         elif frame_num > 1:
#             # print("next_frame_num", next_frame_num)
#             frame_idx: int = frame_num-2
#             single_cell_vec = start_list_value_vec_dict[cell_idx][frame_idx]
#         else:
#             raise Exception()
#
#
#         single_cell_mtx: np.array = single_cell_vec.reshape(single_cell_vec.shape[0], 1)
#
#         total_cell_in_next_frame: int = profit_mtx_list[frame_num].shape[1]
#         single_cell_mtx = np.repeat(single_cell_mtx, total_cell_in_next_frame, axis=1)
#
#         last_layer_all_probability_mtx: np.array = single_cell_mtx * profit_mtx_list[frame_num]
#
#         index_ab_vec = np.argmax(last_layer_all_probability_mtx, axis=0)
#         # value_ab_vec = np.max(last_layer_all_probability_mtx, axis=0)
#         value_ab_vec = obtain_matrix_value_by_index_list(last_layer_all_probability_mtx, index_ab_vec)
#
#         if ( np.all(value_ab_vec == 0) ):
#             to_skip_cell_idx_list.append(cell_idx)
#             continue
#
#         start_list_index_vec_dict[cell_idx].append(index_ab_vec)
#         start_list_value_vec_dict[cell_idx].append(value_ab_vec)
#
#     return start_list_index_vec_dict, start_list_value_vec_dict




def depricated_process_calculate_single_best_cell_track(profit_mtx_list: list, merge_above_threshold:float=1.0, split_below_threshold:float=1.0):   # former method _process
    # print("_process_1")

    start_list_index_vec_list_dict: int = defaultdict(list)
    start_list_value_vec_list_dict: int = defaultdict(list)

    start_list_index_vec_list: list = []
    start_list_value_vec_list: list = []


    linkage_strategy: str = "viterbi"

    #loop each row on first prob matrix. return the maximum value and index through all the frames, the first prob matrix in profit_matrix_list is a matrix (2D array)
    first_frame_mtx: np.array = profit_mtx_list[0]
    total_cell_in_first_frame: int = first_frame_mtx.shape[0]

    cell_idx_frame_num_tuple_list: list = []
    total_frame: int = len(profit_mtx_list)
    for frame_num in range(1, total_frame):
        for cell_idx in range(0, total_cell_in_first_frame):
            cell_idx_frame_idx_tuple: tuple = (cell_idx, frame_num)
            cell_idx_frame_num_tuple_list.append(cell_idx_frame_idx_tuple)



    # frame_num_cell_idx_occupation_vec_dict: dict = {}
    # for frame_num in range(1, total_frame):
    #     total_cell_in_frame_num: int = profit_mtx_list[frame_num].shape[0]
    #     frame_num_cell_idx_occupation_vec_dict[frame_num] = [False for idx in range(0, total_cell_in_frame_num)]


    to_skip_cell_idx_list: list = []
    for cell_idx_frame_idx_tuple in cell_idx_frame_num_tuple_list:
        cell_idx = cell_idx_frame_idx_tuple[0]
        frame_num = cell_idx_frame_idx_tuple[1]

        if cell_idx in to_skip_cell_idx_list:
            continue

        if frame_num == 1:
            single_cell_vec = first_frame_mtx[cell_idx]
        elif frame_num > 1:
            # print("next_frame_num", next_frame_num)
            frame_idx: int = frame_num-2
            single_cell_vec = start_list_value_vec_list_dict[cell_idx][frame_idx]
        else:
            raise Exception()


        single_cell_mtx: np.array = single_cell_vec.reshape(single_cell_vec.shape[0], 1)

        total_cell_in_next_frame: int = profit_mtx_list[frame_num].shape[1]
        single_cell_mtx = np.repeat(single_cell_mtx, total_cell_in_next_frame, axis=1)

        last_layer_all_probability_mtx: np.array = single_cell_mtx * profit_mtx_list[frame_num]


        index_ab_vec = np.argmax(last_layer_all_probability_mtx, axis=0)
        # value_ab_vec = np.max(last_layer_all_probability_mtx, axis=0)
        value_ab_vec = obtain_matrix_value_by_index_list(last_layer_all_probability_mtx, index_ab_vec)

        if ( np.all(value_ab_vec == 0) ):
            # print(">> np.all(value_ab_vec == 0); break")
            to_skip_cell_idx_list.append(cell_idx)
            continue
            # break

        start_list_index_vec_list_dict[cell_idx].append(index_ab_vec)
        start_list_value_vec_list_dict[cell_idx].append(value_ab_vec)




    return start_list_index_vec_list_dict, start_list_value_vec_list_dict



def depricated_derive_one_cell_track(track_idx: int, index_ab_vec_list: np.array, value_ab_vec_list: list):

    total_step: int = len(value_ab_vec_list)
    total_frame: int = total_step + 2
    best_track_list: list = [None] * total_frame
    last_step_idx: int = total_step - 1;    print("last_step_idx", last_step_idx)


    # print("new value_ab_vec", np.round(value_ab_vec_list[last_step_idx], 20))
    last_step_best_result_idx: int = np.argmax(value_ab_vec_list[last_step_idx])

    end_idx: int = 0
    for reversed_idx in range(last_step_idx, end_idx-1, -1):
        frame_num: int = reversed_idx + 2
        # next_step_best_result_idx: int = index_ab_vec_list[previous_step_idx][last_step_best_result_idx]
        previous_frame_cell_idx: int = index_ab_vec_list[reversed_idx][last_step_best_result_idx]
        best_track_list[frame_num] = (last_step_best_result_idx, frame_num, previous_frame_cell_idx)

        last_step_best_result_idx = previous_frame_cell_idx


    if len(index_ab_vec_list) > 1:
        frame_num = 1
        best_track_list[frame_num] = (last_step_best_result_idx, 1, track_idx)
        frame_num = 0
        best_track_list[frame_num] = (track_idx, 0, -1)

    elif len(index_ab_vec_list) == 1:
        frame_num = 0
        best_track_list[frame_num] = (track_idx, 0, -1)

    else:
        raise Exception(len(index_ab_vec_list))



    return best_track_list


def depricated_find_one_cell_1(index_ab_vec_list, value_ab_vec_list, track_idx: int):

    # store_dict = defaultdict(list)

    # print("len(start_list_value_vec_dict.items())", len(start_list_value_vec_dict.items()))
    # for track_idx, value_ab_vec_list in start_list_value_vec_dict.items():
    #     index_ab_vec_list: list = start_list_index_vec_dict[track_idx]

    track_list: list = []
    last_step: int = len(value_ab_vec_list) - 1
    frame_num: int = last_step + 2
    value_ab_vec: np.array = value_ab_vec_list[last_step]

    # print("original value_ab_vec", np.round(value_ab_vec, 20))
    current_maximize_idx: int = np.argmax(value_ab_vec)

    current_maximize_index_value: int = index_ab_vec_list[last_step][current_maximize_idx]

    track_list.append( (current_maximize_idx, frame_num, current_maximize_index_value) )

    for reverse_step_i in generate_reverse_range_list( len(value_ab_vec_list)-1, end=0):    #https://realpython.com/python-range/#decrementing-with-range
        current_maximize_idx = current_maximize_index_value

        previous_maximize_index_value = index_ab_vec_list[reverse_step_i - 1][current_maximize_idx]

        track_list.append((current_maximize_idx, reverse_step_i + 1, previous_maximize_index_value))

        current_maximize_index_value = previous_maximize_index_value

    # print("len(store_dict[track_idx])", len(store_dict[track_idx]))

    # for i in range( len(value_ab_vec_list)-1 ):
    #     current_maximize_index_ = previous_maximize_index
    #     previous_maximize_index_ = index_ab_vec_list[current_step - i - 1][current_maximize_index_]
    #     store_dict[track_idx].append((current_maximize_index_, current_step + 1 - i, previous_maximize_index_))
    #     previous_maximize_index = previous_maximize_index_

    ## code walkthrough
    if len(value_ab_vec_list) > 1:
        track_list.append( (previous_maximize_index_value, 1, track_idx) )
        track_list.append( (track_idx, 0, -1) )

    elif len(value_ab_vec_list) == 1:
        track_list.append( (current_maximize_index_value, 0, -1) )

    else:
        raise Exception(len(value_ab_vec_list))


    # for value_list in store_dict.values():
    list.reverse(track_list)

    return track_list




#find the best track start from first frame based on dict which returned from _process
def _find_1(start_list_index_vec_dict, start_list_value_vec_dict):

    store_dict = defaultdict(list)

    # print("len(start_list_value_vec_dict.items())", len(start_list_value_vec_dict.items()))
    for track_idx, value_ab_vec_list in start_list_value_vec_dict.items():
        index_ab_vec_list: list = start_list_index_vec_dict[track_idx]

        last_step: int = len(value_ab_vec_list) - 1
        frame_num: int = last_step + 2
        value_ab_vec: np.array = value_ab_vec_list[last_step]

        # print("original value_ab_vec", np.round(value_ab_vec, 20))
        current_maximize_idx: int = np.argmax(value_ab_vec)

        current_maximize_index_value: int = index_ab_vec_list[last_step][current_maximize_idx]

        store_dict[track_idx].append( (current_maximize_idx, frame_num, current_maximize_index_value) )

        for reverse_step_i in generate_reverse_range_list( len(value_ab_vec_list)-1, end=0):    #https://realpython.com/python-range/#decrementing-with-range
            current_maximize_idx = current_maximize_index_value

            previous_maximize_index_value = index_ab_vec_list[reverse_step_i - 1][current_maximize_idx]

            frame_idx: int = reverse_step_i + 1
            store_dict[track_idx].append((current_maximize_idx, frame_idx, previous_maximize_index_value))

            current_maximize_index_value = previous_maximize_index_value

        # print("len(store_dict[track_idx])", len(store_dict[track_idx]))

        # for i in range( len(value_ab_vec_list)-1 ):
        #     current_maximize_index_ = previous_maximize_index
        #     previous_maximize_index_ = index_ab_vec_list[current_step - i - 1][current_maximize_index_]
        #     store_dict[track_idx].append((current_maximize_index_, current_step + 1 - i, previous_maximize_index_))
        #     previous_maximize_index = previous_maximize_index_

        ## code walkthrough
        if len(value_ab_vec_list) > 1:
            store_dict[track_idx].append( (previous_maximize_index_value, 1, track_idx) )
            store_dict[track_idx].append( (track_idx, 0, -1) )

        elif len(value_ab_vec_list) == 1:
            store_dict[track_idx].append( (current_maximize_index_value, 0, -1) )

        else:
            raise Exception(len(value_ab_vec_list))


    for value_list in store_dict.values():
        list.reverse(value_list)

    return store_dict



def generate_reverse_range_list(start: int, end: int = 0):
    reverse_range_list: list = []
    for i in range(start, end, -1):
        reverse_range_list.append(i)

    return reverse_range_list




# after got tracks which started from first frame, check if there are very lower prob between each two cells, then truncate it.
# store_dict, threshold, profit_matrix_list
def _cut_1(original_track_dict: dict, threshold: float, profit_matrix_list: list):
    short_track_list_dict: dict = {}


    is_first_cell_id_not_zero: bool = list(original_track_dict.keys())[0] != 0
    if is_first_cell_id_not_zero:
        # print("is_first_cell_id_not_zero", is_first_cell_id_not_zero)
        # raise Exception("investigate")
        for miss_key in range(list(original_track_dict.keys())[0]):
            short_track_list = []
            short_track_list.append((miss_key,0, -1))
            short_track_list_dict[miss_key] = short_track_list


    for cell_id, track_content_list in original_track_dict.items():
        short_track_list = []
        for index in range(len(original_track_dict[cell_id]) - 1):
            current_frame = original_track_dict[cell_id][index][1]

            current_node = original_track_dict[cell_id][index][0]
            next_node = original_track_dict[cell_id][index + 1][0]

            weight_between_nodes = profit_matrix_list[current_frame][current_node][next_node]
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
def _mask_1(short_track_list_dict: dict, profit_matrix_list: list):
    # print("_mask_1")
    mask_frame_cell_id_list: list = []      #list list that stores [frame_id][cell_id]

    # initialize the transition group with all False
    for profit_matrix in profit_matrix_list:
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



def obtain_matrix_value_by_index_list(last_layer_all_probability_mtx: np.array, index_value_list: list, axis=0):
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


def viterbi_flow(series: str, segmentation_folder: str, all_segmented_filename_list: list, output_folder: str):

    segmented_filename_list: list = find_segmented_filename_list_by_series(series, all_segmented_filename_list)

    prof_mat_list: list = derive_prof_matrix_list(segmentation_folder, output_folder, series, segmented_filename_list)

    is_create_excel: bool = False
    if is_create_excel:
        excel_output_dir_path = "d:/tmp/"
        create_prof_matrix_excel(series, prof_mat_list, excel_output_dir_path)


    all_track_dict = _iteration_create_viterbi_track_data(prof_mat_list)



    count_dict = {}
    for key, value_list in all_track_dict.items():
        seq_length = len(value_list)
        if seq_length not in count_dict:
            count_dict[seq_length] = 0

        count_dict[seq_length] += 1


    result_list = []
    for i in range(len(all_track_dict)):
        if i not in all_track_dict.keys():
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



def find_best_track(handling_cell_idx: int,
                    last_layer_all_probability_mtx: np.array,
                    profit_mtx_list: list,
                    handling_frame_num: int,
                    frame_num_cell_slot_idx_occupation_tuple_list_dict: dict,
                    merge_above_threshold: float,
                    cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict: dict):

    to_redo_cell_idx_set: set = set()
    total_cell_slot_next_frame: int = last_layer_all_probability_mtx.shape[1]
    index_ab_vec: list = [None] * total_cell_slot_next_frame
    value_ab_vec: list = [None] * total_cell_slot_next_frame

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

                if handling_frame_num == 2:     handling_cell_probability: float = profit_mtx_list[0][handling_cell_idx][cell_slot_idx]
                elif handling_frame_num > 2:    handling_cell_probability: float = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[handling_cell_idx][start_list_value_idx][cell_slot_idx]

                for occupied_cell_idx in occupied_cell_idx_tuple:
                    if handling_frame_num == 2:     occupied_cell_probability: float = profit_mtx_list[0][occupied_cell_idx][cell_slot_idx]
                    elif handling_frame_num > 2:    occupied_cell_probability: float = cell_idx_frame_num_cell_slot_idx_best_value_vec_dict_dict[occupied_cell_idx][start_list_value_idx][cell_slot_idx]



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
                        to_redo_cell_idx_set.add(occupied_cell_idx)

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

    return index_ab_vec, np.array(value_ab_vec), list(to_redo_cell_idx_set)



if __name__ == '__main__':
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

    existing_series_list = find_existing_series_list(input_series_list, listdir(output_folder))


    viterbi_result_dict = {}
    is_use_thread: bool = False

    if is_use_thread:
        for input_series in input_series_list:
            viterbi_result_dict[input_series] = []

        pool = ThreadPool(processes=8)
        thread_list: list = []

        all_prof_mat_list_dict: dict = {}
        for series in existing_series_list:
            print(f"working on series: {series}", end="\t")

            async_result = pool.apply_async(viterbi_flow, (series, segmentation_folder, all_segmented_filename_list, output_folder,)) # tuple of args for foo
            thread_list.append(async_result)

        for thread_idx in range(len(thread_list)):
            final_result_list = thread_list[thread_idx].get()
            viterbi_result_dict[series] = final_result_list
            print(f"Thread {thread_idx} completed")

    else:
        for series in existing_series_list:
            print(f"working on series: {series}", end="\t")
            final_result_list = viterbi_flow(series, segmentation_folder, all_segmented_filename_list, output_folder)
            viterbi_result_dict[series] = final_result_list




    print("save_track_dictionary")
    save_track_dictionary(viterbi_result_dict, save_dir + "viterbi_results_dict.pkl")

    with open(save_dir + "viterbi_results_dict.txt", 'w') as f:
        f.write(str(viterbi_result_dict[series]))


    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {np.round(execution_time, 4)} seconds")