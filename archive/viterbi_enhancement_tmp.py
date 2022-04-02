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

from skimage.measure._regionprops import RegionProperties


def main():
    print("start")
    dir = "G:/My Drive/leiden_university_course_materials/thesis/cell_tracking/viterbi linkage/dataset/"
    segmentation_folder = dir + 'segmentation_unet_seg/'
    images_folder = dir + 'images/'
    output_folder = dir + 'output_unet_seg_finetune/'
    save_folder = dir + 'save_directory/'



    viterbi_results_dict = {
        "S01": [], "S02": [],  "S03": [], "S04": [], "S05": [], "S06": [], "S07": [], "S08": [], "S09": [], "S10": [],
        "S11": [],  "S12": [], "S13": [], "S14": [], "S15": [], "S16": [], "S17": [], "S18": [], "S19": [], "S20": [],
    }

    series_num_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    #celltypes = ['C1'] # enter all tracked celllines

    #all tracks shorter than DELTA_TIME are false postives and not included in tracks
    # DELTA_TIME = 5
    result_list: list = []

    segmented_file_list: list = listdir(segmentation_folder)
    segmented_file_list.sort()


    # start viterbi

    output_folder_dir_list: list = listdir(output_folder)
    for series_num in series_num_list:
        print(f"series_num: {series_num}")

        is_generate_data: bool = (series_num in output_folder_dir_list)
        if not is_generate_data:
            continue

        segmentation_file_name_list = obtain_file_name_list_by_series_num(series_num, segmentation_folder)

        #get the first image (frame 0) and label the cells:
        img_list = []; label_img_list = [];
        img_arr: np.array = plt.imread(segmentation_folder + segmentation_file_name_list[0])
        img_list.append(img_arr)

        label_img: np.array = measure.label(img_arr, background=0, connectivity=1)
        label_img_list.append(label_img)

        img_cell_sum: int = np.max(label_img)

        C = [];  prof_mat_list = []
        for frame_idx in range(1, len(segmentation_file_name_list)):
            #get next frame and number of cells next frame
            img_next = plt.imread(segmentation_folder +'/' + segmentation_file_name_list[frame_idx])
            img_list.append(img_next)

            label_img_next = measure.label(img_next, background=0, connectivity=1)
            label_img_list.append(label_img_next)

            img_cell_sum_next = np.max(label_img_next)

            #prof_size = max(img_cell_sum, cellnb_img_next)
            #create empty dataframe for element of profit matrix C
            # 2D matrix with dimension "current cell sum" x "next cell sum"
            prof_mtx: np.array = np.zeros((img_cell_sum, img_cell_sum_next), dtype=float)

            #loop through all combinations of cells in this and the next frame
            for cell_idx_i in range(img_cell_sum):
                #cellnb i + 1 because cellnumbering in output files starts from 1
                file_name: str = segmentation_file_name_list[frame_idx].replace(".png", "")
                cell_count_str: str = str(cell_idx_i+1).zfill(2)
                cell_idx_filename = f"mother_{file_name}_Cell{cell_count_str}.png"
                cell_img_i: np.array = plt.imread(output_folder + series_num +'/' + cell_idx_filename)

                #predictions are for each cell in curr img
                cell_i_prop_list: list(RegionProperties) = measure.regionprops(label_img_next, intensity_image=cell_img_i) #label_img_next是二值图像为255，无intensity。需要与output中的预测的细胞一一对应，预测细胞有intensity

                for cell_idx_j in range(img_cell_sum_next):
                    #calculate profit score from mean intensity neural network output in segmented cell area
                    prof_mtx[cell_idx_i, cell_idx_j] = cell_i_prop_list[cell_idx_j].mean_intensity         #得到填充矩阵size = max(img_cell_sum, cellnb_img_next)：先用预测的每一个细胞的mean_intensity填满cellnb_img, cellnb_img_next行和列

            #prof_mtx = prof_mtx/np.max(prof_mtx)    #np.max 矩阵中的最大数值 归一化
            # prof_mtx = prof_mtx
            prof_mat_list.append(prof_mtx)

            #make next frame current frame
            img_cell_sum = img_cell_sum_next
            label_img = label_img_next

        # end viterbi


        all_tracks_dict: dict(int, list) = _iteration(prof_mat_list)
        # print(type(all_tracks_dict))
        # print(all_tracks_dict.keys())
        # for key, value in all_tracks_dict.items():
        #     print(key)
        #     print(type(key))
        #     print(value)
        #     print(type(value))
        # exit()


        result_list = []
        for i in range(len(all_tracks_dict)):
            if i not in all_tracks_dict.keys():
                continue
            else:
                # filter out trackes that is shorter than 5
                min_frame_no = 5
                if len(all_tracks_dict[i]) > min_frame_no:     ## ? why > 5?
                    result_list.append(all_tracks_dict[i])
# <class 'int'> 16
# (current_cell_id, frame_num, from_previous_cell_id)
# <class 'list'> [(16, 0, -1), (16, 1, 16), (19, 2, 16), (19, 3, 19), (18, 4, 19), (17, 5, 18), (18, 6, 17), (18, 7, 18), (18, 8, 18), (16, 9, 18), (17, 10, 16), (16, 11, 17), (15, 12, 16), (16, 13, 15), (15, 14, 16), (15, 15, 15), (19, 16, 15), (17, 17, 19), (19, 18, 17), (14, 19, 19), (16, 20, 14), (16, 21, 16), (17, 22, 16), (15, 23, 17), (21, 24, 15), (24, 25, 21), (19, 26, 24), (22, 27, 19), (22, 28, 22), (23, 29, 22), (25, 30, 23), (25, 31, 25), (24, 32, 25), (19, 33, 24), (19, 34, 19), (21, 35, 19), (23, 36, 21), (20, 37, 23), (24, 38, 20), (24, 39, 24), (25, 40, 24), (23, 41, 25), (23, 42, 23), (21, 43, 23), (23, 44, 21), (23, 45, 23), (23, 46, 23), (21, 47, 23), (23, 48, 21), (21, 49, 23), (21, 50, 21), (24, 51, 21), (26, 52, 24), (25, 53, 26), (21, 54, 25), (22, 55, 21), (30, 56, 22), (25, 57, 30), (22, 58, 25), (26, 59, 22), (24, 60, 26), (23, 61, 24), (23, 62, 23), (23, 63, 23), (22, 64, 23), (27, 65, 22), (23, 66, 27), (23, 67, 23), (19, 68, 23), (20, 69, 19), (23, 70, 20), (19, 71, 23), (22, 72, 19), (20, 73, 22), (22, 74, 20), (21, 75, 22), (18, 76, 21), (25, 77, 18), (24, 78, 25), (22, 79, 24), (21, 80, 22), (22, 81, 21), (19, 82, 22), (17, 83, 19), (19, 84, 17), (23, 85, 19), (23, 86, 23), (21, 87, 23), (21, 88, 21), (21, 89, 21), (21, 90, 21), (23, 91, 21), (23, 92, 23), (25, 93, 23), (24, 94, 25), (25, 95, 24), (24, 96, 25), (21, 97, 24), (23, 98, 21), (23, 99, 23), (25, 100, 23), (26, 101, 25), (25, 102, 26), (24, 103, 25), (20, 104, 24), (23, 105, 20), (24, 106, 23), (22, 107, 24), (19, 108, 22), (22, 109, 19), (22, 110, 22), (19, 111, 22), (17, 112, 19), (17, 113, 17), (18, 114, 17), (19, 115, 18), (15, 116, 19), (19, 117, 15), (21, 118, 19), (18, 119, 21)]


        # post handling: detect problematic cells; handling of problematic cells; remove track;
        ## The section below can be removed when SWAP is implemented
        # print(result_list)
        for j in range(len(result_list)-1):
            for k in range(j + 1, len(result_list)):
                pre_track = result_list[j]
                next_track = result_list[k]

                ## ?? what does this do
                overlap_track: list = sorted(set([i[0:2] for i in pre_track]) & set([i[0:2] for i in next_track]), key = lambda x : (x[1], x[0]))

                if overlap_track == []:
                    continue
                overlap_frame1 = overlap_track[0][1]
                node_combine = overlap_track[0][0]
                pre_frame = overlap_frame1 - 1
                for i, tuples in enumerate(pre_track):
                    if tuples[1] == pre_frame:
                        index_merge1 = i
                        break
                    else:
                        continue
                node_merge1 = pre_track[index_merge1][0]
                for ii, tuples in enumerate(next_track):
                    if tuples[1] == pre_frame:
                        index_merge2 = ii
                        break
                    else:
                        continue
                node_merge2 = next_track[index_merge2][0]
                sub_matrix = prof_mat_list[pre_frame]
                threSh1 = sub_matrix[node_merge1][node_combine]
                threSh2 = sub_matrix[node_merge2][node_combine]
                if threSh1 < threSh2:
                    result_list[k] = next_track
                    pre_track_new = copy.deepcopy(pre_track[0:index_merge1 + 1])
                    result_list[j] = pre_track_new
                else:
                    result_list[j] = pre_track
                    next_track_new = copy.deepcopy(next_track[0:index_merge2 + 1])
                    result_list[k] = next_track_new
        #print(result_list)
        final_result = []
        for i in range(len(result_list)):
            if (len(result_list[i])>5):
                final_result.append(result_list[i])
        identifier = series_num
        viterbi_results_dict[identifier] = final_result


    print(save_folder)
    save_track_dictionary(viterbi_results_dict, save_folder + "viterbi_results_dict.pkl")





def obtain_file_name_list_by_series_num(series_num: int, segmented_file_dir: str):
    segmented_file_name_list: list = listdir(segmented_file_dir)
    segmented_file_name_list.sort()

    filtered_segmented_file_name_list: list = []
    #select all files of the current images series_list and celltype
    for segmented_file in segmented_file_name_list:
        if series_num in segmented_file:
            filtered_segmented_file_name_list.append(segmented_file)

    return filtered_segmented_file_name_list


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
        if len(v)!=1:
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
def _cut(longTracks, Threshold, transition_group):
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
            if (weight_between_nodes > Threshold):
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
def _iteration(transition_group_dict: dict):
    all_tracks_dict: dict = {}

    start_list_index, start_list_value = _process(transition_group_dict)
    store_dict = _find(start_list_index, start_list_value)
    short_Tracks = _cut(store_dict, 0.01, transition_group_dict)
    all_tracks_dict.update(short_Tracks)
    length = len(all_tracks_dict)
    mask_transition_group =  _mask(short_Tracks, transition_group_dict)
    for p_matrix in range(1, len(transition_group_dict)):
        #print(p_matrix)
        #print(transition_group[p_matrix].shape)
        #print(transition_group[p_matrix].shape[0])
        for node in range(transition_group_dict[p_matrix].shape[0]):  #skip all nodes which are already passed
            #print(node)
            if (mask_transition_group[p_matrix][node]==True):
                continue
            else:
                new_transition_group = []
                new_transition_group_ = transition_group_dict[p_matrix:]
                new_transition_group.append(new_transition_group_[0][node])
                new_transition_group[1:] = transition_group_dict[p_matrix + 1:]
                next_list_index, next_list_value = _process_iter(new_transition_group)
                new_store_dict = _find_iter(next_list_index, next_list_value, p_matrix, node)
                new_short_Tracks = _cut_iter(new_store_dict, 0.01, new_transition_group, p_matrix)
            mask_transition_group =  _mask_update(new_short_Tracks, mask_transition_group)
            for ke, val in new_short_Tracks.items():                
                all_tracks_dict[length + ke + 1] = val
            length = len(all_tracks_dict)

    return all_tracks_dict



#loop each node on first frame to find the optimal path using probabilty multiply
def _process(transition_group):
    step = len(transition_group)
    start_list_index = defaultdict(list)
    start_list_value = defaultdict(list)
    #loop each row on first prob matrix. return the maximum value and index through the whole frames 
    #the first prob matrix in transition_group is a matrix (2D array)
    for ii, item in enumerate(transition_group[0]):
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

if __name__ == '__main__':
    main()