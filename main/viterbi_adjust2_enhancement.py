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

start_time = time.perf_counter()


folder_path: str = 'D:/viterbi linkage/dataset/'


segmentation_folder = folder_path + 'segmentation_unet_seg//'
images_folder = folder_path + 'dataset//images//'
output_folder = folder_path + 'output_unet_seg_finetune//'
save_dir = folder_path + 'save_directory_enhancement//'




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
def _iteration(transition_group):
    all_tracks = {}
    start_list_index, start_list_value = _process(transition_group)
    store_dict = _find(start_list_index, start_list_value)
    short_Tracks = _cut(store_dict, 0.01, transition_group)
    all_tracks.update(short_Tracks)
    length = len(all_tracks)
    mask_transition_group =  _mask(short_Tracks, transition_group)
    for p_matrix in range(1,len(transition_group)):
        #print(p_matrix)
        #print(transition_group[p_matrix].shape)
        #print(transition_group[p_matrix].shape[0])
        for node in range(transition_group[p_matrix].shape[0]):  #skip all nodes which are already passed 
            #print(node)
            if (mask_transition_group[p_matrix][node]==True):
                continue
            else:
                new_transition_group = []
                new_transition_group_ = transition_group[p_matrix:]
                new_transition_group.append(new_transition_group_[0][node])
                new_transition_group[1:] = transition_group[p_matrix+1:]
                next_list_index, next_list_value = _process_iter(new_transition_group)
                new_store_dict = _find_iter(next_list_index, next_list_value, p_matrix, node)
                new_short_Tracks = _cut_iter(new_store_dict, 0.01, new_transition_group, p_matrix)
            mask_transition_group =  _mask_update(new_short_Tracks, mask_transition_group)
            for ke, val in new_short_Tracks.items():                
                all_tracks[length + ke + 1] = val
            length = len(all_tracks)
    return all_tracks



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




if __name__ == '__main__':
    print("start")

    # viterbi_result_dict = {
    #     "S01": [], "S02": [], "S03": [], "S04": [], "S05": [], "S06": [], "S07": [], "S08": [], "S09": [], "S10": [],
    #     "S11": [], "S12": [], "S13": [], "S14": [], "S15": [], "S16": [], "S17": [], "S18": [], "S19": [], "S20": []
    # }

    input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                         'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']


    #celltypes = ['C1'] # enter all tracked celllines

    #all tracks shorter than DELTA_TIME are false postives and not included in tracks
    # DELTA_TIME = 5
    result_list = []

    segmented_filename_list = listdir(segmentation_folder)
    segmented_filename_list.sort()

    existing_series_list = find_existing_series_list(input_series_list, listdir(output_folder))

    viterbi_result_dict = {}
    for input_series in input_series_list:
        viterbi_result_dict[input_series] = []

    for series in existing_series_list:
        print(f"working on series: {series}")

        file_list = find_segmented_filename_list_by_series(series, segmented_filename_list)


        img_list = []
        prof_mat_list = []

        #get the first image (frame 0) and label the cells:
        img = plt.imread(segmentation_folder + file_list[0])
        img_list.append(img)

        label_img = measure.label(img, background=0, connectivity=1)
        cellnb_img = np.max(label_img)

        for framenb in range(1, len(file_list)):
            #get next frame and number of cells next frame
            img_next = plt.imread(segmentation_folder + '/' + file_list[framenb])
            img_list.append(img_next)

            label_img_next = measure.label(img_next, background=0, connectivity=1)
            cellnb_img_next = np.max(label_img_next)

            #create empty dataframe for element of profit matrix C
            prof_mat = np.zeros( (cellnb_img, cellnb_img_next), dtype=float)

            #loop through all combinations of cells in this and the next frame
            for cellnb_i in range(cellnb_img):
                #cellnb i + 1 because cellnumbering in output files starts from 1
                cell_i_filename = "mother_" + file_list[framenb][:-4] + "_Cell" + str(cellnb_i + 1).zfill(2) + ".png"
                cell_i = plt.imread(output_folder + series + '/' + cell_i_filename)
                #predictions are for each cell in curr img
                cell_i_props = measure.regionprops(label_img_next, intensity_image=cell_i) #label_img_next是二值图像为255，无intensity。需要与output中的预测的细胞一一对应，预测细胞有intensity
                for cellnb_j in range(cellnb_img_next):
                    #calculate profit score from mean intensity neural network output in segmented cell area
                    prof_mat[cellnb_i, cellnb_j] = cell_i_props[cellnb_j].mean_intensity         #得到填充矩阵size = max(cellnb_img, cellnb_img_next)：先用预测的每一个细胞的mean_intensity填满cellnb_img, cellnb_img_next行和列

            prof_mat_list.append(prof_mat)

            #make next frame current frame
            cellnb_img = cellnb_img_next
            label_img = label_img_next

        all_tracks = _iteration(prof_mat_list)


        result_list = []
        for i in range(len(all_tracks)):
            if i not in all_tracks.keys():
                continue
            else:
                if (len(all_tracks[i]) > 5):
                    result_list.append(all_tracks[i])





        #print(result)
        for j in range(len(result_list) - 1):
            for k in range(j + 1, len(result_list)):
                pre_track_list = result_list[j]
                next_track_list = result_list[k]
                overlap_track_list = sorted(set([i[0:2] for i in pre_track_list]) & set([i[0:2] for i in next_track_list]), key = lambda x : (x[1], x[0]))
                if overlap_track_list == []:
                    continue

                overlap_frame1_list =  overlap_track_list[0][1]
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


    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {execution_time: 0.4f} seconds")