# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:31:51 2021

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
import cv2
#from google.colab.patches import cv2_imshow
import random
from itertools import combinations
import pickle
from collections import defaultdict


import time




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



#find the best track start from current frame, and current node based on dict which returned from _process_iter
def _find_iter(start_list_index, start_list_value, start_frame, node):
    global previous_maximize_index_1
    store_dict = defaultdict(list)
    for k, v in start_list_value.items():
        #print(f'start from {k}-th sample:')
        current_step = len(v) - 1
        current_maximize_index = np.argmax(v[current_step])
        previous_maximize_index = start_list_index[k][current_step][current_maximize_index]
        store_dict[k].append((current_maximize_index, start_frame + current_step + 2, previous_maximize_index))

        for i in range(len(v)-1):
            # print(current_maximize_index)
            current_maximize_index_ = previous_maximize_index
            # previous_maximize_index = np.argmax(v[current_step - 1])
            previous_maximize_index_1 = start_list_index[k][current_step - i - 1][current_maximize_index_]
            #print(f'current_maximize_index: {current_maximize_index}, '
            #      f'current_step: {current_step}, '
            #      f'previous_maximize_index: {previous_maximize_index}')
            store_dict[k].append((current_maximize_index_, start_frame + current_step + 1 - i, previous_maximize_index_1))
            previous_maximize_index = previous_maximize_index_1
        if len(v)!=1:
            store_dict[k].append((previous_maximize_index_1, start_frame + 1, node))
            store_dict[k].append((node, start_frame + 0, -1))
        else:
            store_dict[k].append((previous_maximize_index, start_frame + 1, node))
            store_dict[k].append((node, start_frame + 0, -1))

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
    for ii, item in enumerate(transition_group[0].reshape(1, transition_group[0].shape[0])):
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



def create_prof_matrix_excel(all_prof_mat_list_dict: dict, excel_output_dir_path: str):
    import pandas as pd
    existing_series_list: list = all_prof_mat_list_dict.keys()
    for series in existing_series_list:
        one_series_img_list: list = all_prof_mat_list_dict[series]
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



#start from first frame and loop the unvisited nodes in the other frames
def _iteration_1(profit_matrix_list: list):
    print("_iteration_1")
    all_track_dict: dict = {}
    start_list_index_vec_dict, start_list_value_vec_dict = calculate_best_cell_track(profit_matrix_list) # 2D array list

    store_dict = _find_1(start_list_index_vec_dict, start_list_value_vec_dict)

    short_track_list_dict = _cut_1(store_dict, 0.01, profit_matrix_list)   # filter out cells that does not make sense (e.g. too low probability)

    all_track_dict.update(short_track_list_dict)


    ##
    ## handle new cells that enter the image
    ##
    max_id_in_dict: int = len(all_track_dict)           # dict key is cell idx
    mask_transition_group = _mask_1(short_track_list_dict, profit_matrix_list)

    for profit_matrix in range(1, len(profit_matrix_list)):
        #print(profit_matrix)
        #print(transition_group[profit_matrix].shape)
        #print(transition_group[profit_matrix].shape[0])
        for cell_idx in range(profit_matrix_list[profit_matrix].shape[0]):  #skip all nodes which are already passed
            #print(cell_idx)
            if (mask_transition_group[profit_matrix][cell_idx] == True):
                continue
            else:
                new_transition_group = []
                new_transition_group_ = profit_matrix_list[profit_matrix:]
                new_transition_group.append(new_transition_group_[0][cell_idx])
                new_transition_group[1:] = profit_matrix_list[profit_matrix + 1:]
                next_list_index, next_list_value = _process_iter(new_transition_group)
                new_store_dict = _find_iter(next_list_index, next_list_value, profit_matrix, cell_idx)
                new_short_Tracks = _cut_iter(new_store_dict, 0.01, new_transition_group, profit_matrix)

            mask_transition_group = _mask_update(new_short_Tracks, mask_transition_group)
            for ke, val in new_short_Tracks.items():
                all_track_dict[max_id_in_dict + ke + 1] = val

            max_id_in_dict = len(all_track_dict)

    return all_track_dict



#loop each node on first frame to find the optimal path using probabilty multiply
def calculate_best_cell_track(profit_mtx_list: list):   # former method _process
    print("_process_1")
    total_step: int = len(profit_mtx_list)

    start_list_index_vec_dict: int = defaultdict(list)
    start_list_value_vec_dict: int = defaultdict(list)

    #loop each row on first prob matrix. return the maximum value and index through all the frames, the first prob matrix in profit_matrix_list is a matrix (2D array)
    first_frame_mtx: np.array = profit_mtx_list[0]
    total_cell_in_first_frame: int = first_frame_mtx.shape[0]
    for cell_idx in range(0, total_cell_in_first_frame):

        single_cell_vec = first_frame_mtx[cell_idx]
        for next_frame_idx in range(1, total_step):
            # single_cell_mtx: np.array = single_cell_vec[:, np.newaxis]   #https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
            single_cell_mtx: np.array = single_cell_vec.reshape(single_cell_vec.shape[0], 1)

            ## ?? Is this step attempting to calculate the max probability from 2 steps prof_mtx?
            num_of_cell_in_next_frame: int = profit_mtx_list[next_frame_idx].shape[1]
            single_cell_mtx = np.repeat(single_cell_mtx, num_of_cell_in_next_frame, axis=1)

            ## calculate profit matrix of 2 steps
            probability_mtx: np.array = single_cell_mtx * profit_mtx_list[next_frame_idx]
            index_ab_vec = np.argmax(probability_mtx, axis=0)
            value_ab_vec = np.max(probability_mtx, axis=0)

            if ( np.all(value_ab_vec == 0) ):
                break

            start_list_index_vec_dict[cell_idx].append(index_ab_vec)
            start_list_value_vec_dict[cell_idx].append(value_ab_vec)
            single_cell_vec = value_ab_vec

    return start_list_index_vec_dict, start_list_value_vec_dict




#find the best track start from first frame based on dict which returned from _process
def _find_1(start_list_index_vec_dict, start_list_value_vec_dict):

    store_dict = defaultdict(list)

    for track_idx, value_ab_vec_list in start_list_value_vec_dict.items():
        index_ab_vec_list: list = start_list_index_vec_dict[track_idx]

        last_step: int = len(value_ab_vec_list) - 1

        value_ab_vec: np.array = value_ab_vec_list[last_step]
        current_maximize_idx: int = np.argmax(value_ab_vec)

        current_maximize_index_value: int = index_ab_vec_list[last_step][current_maximize_idx]

        store_dict[track_idx].append( (current_maximize_idx, last_step + 2, current_maximize_index_value) )

        for reverse_step_i in generate_reverse_range_list( len(value_ab_vec_list)-1, end=0):    #https://realpython.com/python-range/#decrementing-with-range
            current_maximize_idx = current_maximize_index_value

            previous_maximize_index_value = index_ab_vec_list[reverse_step_i - 1][current_maximize_idx]

            store_dict[track_idx].append((current_maximize_idx, reverse_step_i + 1, previous_maximize_index_value))

            current_maximize_index_value = previous_maximize_index_value

        # for i in range( len(value_ab_vec_list)-1 ):
        #     current_maximize_index_ = previous_maximize_index
        #     previous_maximize_index_ = index_ab_vec_list[current_step - i - 1][current_maximize_index_]
        #     store_dict[track_idx].append((current_maximize_index_, current_step + 1 - i, previous_maximize_index_))
        #     previous_maximize_index = previous_maximize_index_

        ## code walkthrough
        if len(value_ab_vec_list) != 1:
            store_dict[track_idx].append( (previous_maximize_index_value, 1, track_idx) )
            store_dict[track_idx].append( (track_idx, 0, -1) )

        else:
            store_dict[track_idx].append( (current_maximize_index_value, 0, -1) )


    for value_list in store_dict.values():
        list.reverse(value_list)

    return store_dict



def generate_reverse_range_list(start: int, end: int = 0):
    reverse_range_list: list = []
    for i in range(start, end, -1):
        reverse_range_list.append(i)

    return reverse_range_list




# after got tracks which started from first frame, check if there are very lower prob between each two cells, then truncate it.
def _cut_1(longTracks, threshold, transition_group):
    short_track_list_dict = {}

    if list(longTracks.keys())[0] != 0:
        for miss_key in range(list(longTracks.keys())[0]):
            short_track_list = []
            short_track_list.append((miss_key,0, -1))
            short_track_list_dict[miss_key] = short_track_list

    for key, track in longTracks.items():
        short_track_list = []
        for index in range(len(longTracks[key])-1):
            current_frame = longTracks[key][index][1]
            next_frame = longTracks[key][index+1][1]
            current_node = longTracks[key][index][0]
            next_node = longTracks[key][index+1][0]
            weight_between_nodes = transition_group[current_frame][current_node][next_node]
            if (weight_between_nodes > threshold):
                short_track_list.append(longTracks[key][index])
            else:
                short_track_list = copy.deepcopy(longTracks[key][0: index])
                break


        if (len(short_track_list) == len(longTracks[key])-1 ):
            short_track_list.append( longTracks[key][-1] )
            short_track_list_dict[key] = short_track_list

        else:
            short_track_list.append( longTracks[key][len(short_track_list)] )
            short_track_list_dict[key] = short_track_list

    return short_track_list_dict



#after got all tracks start from first frame, define a mask matrix. all the nodes which are passed by any tracks are labels as True
def _mask_1(short_track_list_dict: dict, profit_matrix_list: list):
    mask_transition_group_list: list = []

    # initialize the transition group with all False
    for prob_mat in profit_matrix_list:
        mask_transition_group_list.append(np.array([False for i in range(prob_mat.shape[0])]))

    mask_transition_group_list.append(np.array([False for i in range(prob_mat.shape[1])]))

    # if the node was passed, lable it to True
    for kk in short_track_list_dict.keys():
        for iindex in range(len(short_track_list_dict[kk])):
            frame = short_track_list_dict[kk][iindex][1]
            node = short_track_list_dict[kk][iindex][0]
            mask_transition_group_list[frame][node] = True

    return mask_transition_group_list



if __name__ == '__main__':
    folder_path: str = 'D:/viterbi linkage/dataset/'

    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    images_folder = folder_path + 'dataset//images//'
    output_folder = folder_path + 'output_unet_seg_finetune//'
    save_dir = folder_path + 'save_directory_enhancement//'


    print("start")
    start_time = time.perf_counter()


    input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                         'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    # input_series_list = ['S01']

    #all tracks shorter than DELTA_TIME are false postives and not included in tracks
    result_list = []

    all_segmented_filename_list = listdir(segmentation_folder)
    all_segmented_filename_list.sort()

    existing_series_list = find_existing_series_list(input_series_list, listdir(output_folder))


    viterbi_result_dict = {}
    for input_series in input_series_list:
        viterbi_result_dict[input_series] = []


    all_prof_mat_list_dict: dict = {}
    for series in existing_series_list:
        print(f"working on series: {series}")

        segmented_filename_list: list = find_segmented_filename_list_by_series(series, all_segmented_filename_list)

        prof_mat_list: list = derive_prof_matrix_list(segmentation_folder, output_folder, series, segmented_filename_list)

        # all_prof_mat_list_dict[series] = tmp_prof_mat_list  # prof = profit matrix


        # is_create_excel: bool = False
        # if is_create_excel:
        #     excel_output_dir_path = "d:/tmp/"
        #     create_prof_matrix_excel(all_prof_mat_list_dict, excel_output_dir_path)


        # for series in existing_series_list:
        #     prof_mat_list = all_prof_mat_list_dict[series]
        # print("a.", prof_mat_list[0][0].shape)
        all_track_dict = _iteration_1(prof_mat_list)

        result_list = []
        for i in range(len(all_track_dict)):
            if i not in all_track_dict.keys():
                continue
            else:
                if (len(all_track_dict[i]) > 5):
                    result_list.append(all_track_dict[i])


        #print(result)
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

        identifier = series
        viterbi_result_dict[identifier] = final_result_list





    print("save_track_dictionary")
    save_track_dictionary(viterbi_result_dict, save_dir + "viterbi_results_dict.pkl")

    with open(save_dir + "viterbi_results_dict.txt", 'w') as f:
        f.write(str(viterbi_result_dict[identifier]))


    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {np.round(execution_time, 4)} seconds")