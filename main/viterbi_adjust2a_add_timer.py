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




segmentation_folder = 'D://viterbi linkage//dataset//segmentation_unet_seg//'
images_folder = 'D://viterbi linkage//dataset//images//'
output_folder = 'D://viterbi linkage//dataset//output_unet_seg_finetune//'


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

    # for store_list in store_dict.values():
    #     if store_list[0] == (14, 0, -1):
    #         tmp = store_list

    short_Tracks = _cut(store_dict, 0.01, transition_group)

    # for store_list in short_Tracks.values():
    #     if store_list[0] == (14, 0, -1):
    #         tmp1 = store_list

    all_tracks.update(short_Tracks)

    is_apply_fix: bool = False # when len(all_tracks.keys() == [0, 2, 3, 5]) == 4, which '5' will be overwritten in next step
    if is_apply_fix:    length = np.max(list(all_tracks.keys())) - 1
    else:               length = len(all_tracks)

    mask_transition_group =  _mask(short_Tracks, transition_group)



    # bugfix
    short_track_count: int = 0
    for p_matrix in range(1, len(transition_group)):
        #print(p_matrix)
        #print(transition_group[p_matrix].shape)
        #print(transition_group[p_matrix].shape[0])
        for node in range(transition_group[p_matrix].shape[0]):  #skip all nodes which are already passed
            #print(node)
            is_node_occupied: bool = mask_transition_group[p_matrix][node]
            if (is_node_occupied == True):
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
                all_tracks[length + short_track_count + 1] = val
                short_track_count += 1

            length = len(all_tracks)

    return all_tracks

#loop each node on first frame to find the optimal path using probabilty multiply
def _process(transition_group):
    step = len(transition_group)
    start_list_index = defaultdict(list)
    start_list_value = defaultdict(list)
    #loop each row on first prob matrix. return the maximum value and index through the whole frames 
    #the first prob matrix in transition_group is a matrix (2D array)
    for cell_idx, item in enumerate(transition_group[0]):

        for frame_idx in range(1, step):
            item = item[:, np.newaxis]
            item = np.repeat(item, transition_group[frame_idx].shape[1], 1)
            index_ab = np.argmax(item * transition_group[frame_idx], 0)
            value_ab = np.max(item * transition_group[frame_idx], 0)
            if (np.all(value_ab==0)):
                break
            start_list_index[cell_idx].append(index_ab)
            start_list_value[cell_idx].append(value_ab)
            item = value_ab

        # if cell_idx == 14:
        #     tmp1 = start_list_index[cell_idx]
        #     tmp2 = start_list_value[cell_idx]
        #     print("grsdb", "bugfix")

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



def filter_track_list_by_length(track_list_list: list, min_track_length: int = 5):
    result_track_list: list = []
    for track_list in track_list_list:
        if (len(track_list) > min_track_length):
            result_track_list.append(track_list)

    return result_track_list


viterbi_results_dict = {   

    "S01": [],    
    "S02": [],
    "S03": [],    
    "S04": [],
    "S05": [],    
    "S06": [],
    "S07": [],    
    "S08": [],
    "S09": [],    
    "S10": [],
    "S11": [],    
    "S12": [],
    "S13": [],    
    "S14": [],
    "S15": [],    
    "S16": [],
    "S17": [],    
    "S18": [],
    "S19": [],    
    "S20": [],
      
}

series = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
# series = ['S13']
#celltypes = ['C1'] # enter all tracked celllines

#all tracks shorter than DELTA_TIME are false postives and not included in tracks
DELTA_TIME = 5 
result = []

segmented_files = listdir(segmentation_folder)
segmented_files.sort()
for serie in series:
    #if data is not complete
    if not serie in listdir(output_folder):
        continue 
    print(serie)
    filelist = []
    img_list = []
    label_img_list = []
    #select all files of the current images series and celltype
    for filename in segmented_files:
        if serie in filename:
            filelist = filelist + [filename]

    C = []
    prof_mat_list = []
    #get the first image (frame 0) and label the cells:
    img = plt.imread(segmentation_folder + filelist[0])
    img_list.append(img)
    label_img = measure.label(img, background=0, connectivity=1)
    label_img_list.append(label_img)
    cellnb_img = np.max(label_img)
        
    for framenb in range(1,len(filelist)):
        #get next frame and number of cells next frame
        img_next = plt.imread(segmentation_folder +'/' + filelist[framenb])
        img_list.append(img_next)
        label_img_next = measure.label(img_next, background=0, connectivity=1)
        label_img_list.append(label_img_next)
        cellnb_img_next = np.max(label_img_next)

        #prof_size = max(cellnb_img, cellnb_img_next)
        #create empty dataframe for element of profit matrix C
        prof_mat = np.zeros((cellnb_img, cellnb_img_next), dtype=float)

        #loop through all combinations of cells in this and the next frame
        for cellnb_i in range(cellnb_img):
            #cellnb i + 1 because cellnumbering in output files starts from 1
            cell_i_filename = "mother_" + filelist[framenb][:-4] + "_Cell" + str(cellnb_i+1).zfill(2) + ".png"
            cell_i = plt.imread(output_folder + serie +'/' + cell_i_filename)
            #predictions are for each cell in curr img
            cell_i_props = measure.regionprops(label_img_next,intensity_image=cell_i) #label_img_next是二值图像为255，无intensity。需要与output中的预测的细胞一一对应，预测细胞有intensity
            for cellnb_j in range(cellnb_img_next):
                #calculate profit score from mean intensity neural network output in segmented cell area
                prof_mat[cellnb_i,cellnb_j] = cell_i_props[cellnb_j].mean_intensity         #得到填充矩阵size = max(cellnb_img, cellnb_img_next)：先用预测的每一个细胞的mean_intensity填满cellnb_img, cellnb_img_next行和列

        #prof_mat = prof_mat/np.max(prof_mat)    #np.max 矩阵中的最大数值 归一化
        prof_mat = prof_mat
        prof_mat_list.append(prof_mat)

        #make next frame current frame
        cellnb_img = cellnb_img_next
        label_img = label_img_next
    #print(len(C))
    #result = tracking(C, prof_mat_list, DELTA_TIME)
    #prof_mat_list3 = prof_mat_list[0:4]

    all_tracks = _iteration(prof_mat_list)


    # result = []
    # for i in range(len(all_tracks)):
    #     if i not in all_tracks.keys():
    #         continue
    #     else:
    #         if (len(all_tracks[i])>5):
    #             result.append(all_tracks[i])


    # for final_track in all_tracks.values():
    #     if final_track[0] == (14, 0, -1):
    #         print("before", final_track)




    result = filter_track_list_by_length(all_tracks.values())


    #print(result)
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



    for final_track in final_result:
        if final_track[0] == (14, 0, -1):
            print("after", final_track)




    identifier = serie
    viterbi_results_dict[identifier] = final_result      
            
'''save track dictionary'''
def save_track_dictionary(dictionary, save_file):
    if not os.path.exists(save_file):
        with open(save_file, 'w'):
            pass
    pickle_out = open(save_file,"wb")
    pickle.dump(dictionary,pickle_out)
    pickle_out.close()    
save_dir ='D://viterbi linkage//dataset//save_directory//'
print(save_dir)
save_track_dictionary(viterbi_results_dict, save_dir + "viterbi_results_dict.pkl")


# with open(save_dir + "viterbi_results_dict.txt", 'w') as f:
#     f.write(str(viterbi_results_dict[identifier]))




folder_path: str = 'D:/viterbi linkage/dataset/'

segmentation_folder = folder_path + 'segmentation_unet_seg//'
images_folder = folder_path + 'dataset//images//'
output_folder = folder_path + 'output_unet_seg_finetune//'
save_dir = folder_path + 'save_directory_enhancement/'

with open(save_dir + "viterbi_adjust2a_add_timer.txt", 'w') as f:
    for s in series:
        f.write("======================" + str(s) + "================================")
        f.write("\n")
        cell_track_list_list = sorted(viterbi_results_dict[s])
        for cell_track_list in cell_track_list_list:
            f.write(str(cell_track_list))
            f.write("\n")

        f.write("\n\n")


execution_time = time.perf_counter() - start_time
print(f"Execution time: {execution_time: 0.4f} seconds")