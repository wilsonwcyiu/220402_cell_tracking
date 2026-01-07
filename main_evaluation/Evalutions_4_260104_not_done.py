# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:41:40 2021

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
# import cv2
import random
from itertools import combinations
import pickle

def main():
    root_dir: str = 'G:/My Drive/leiden_university_course_materials/thesis/viterbi linkage/dataset'
    root_dir: str = "G:\My Drive\leiden_university_course_materials\thesis\260101_thesis_followup\260106_Dicts"
    # root_dir: str = 'D:/viterbi linkage/dataset/'
    save_dir = root_dir + 'Dicts/'                               #'track_dictory_finetune//track_dictory_unet_seg//'
    segmentation_folder = root_dir + '/segmentation_unet_seg/'    #'Unet_trainingsets//data//segmentation_unet_seg//'
    # output_put_dir: str = root_dir + 'evaluation_output/'

    pkl_dir: str = root_dir + "/save_directory_enhancement/pkl_files/"
    output_put_dir: str =  root_dir + "/save_directory_enhancement/"

    # gt_results_dict = open_track_dictionary(save_dir + "gt_results_dict.pkl")
    gt_results_dict = open_track_dictionary(pkl_dir + "gt_results_dict.pkl")
 

    preload_file_name_list = [
                            #  'ground_truth_results_dict.pkl',
                             # 'modified_ground_truth_results_dict.pkl',
                              'DeLTA.pkl',
                              'Hungarian.pkl',
                              'KDE.pkl',
                              'Viterbi__viterbi_adjust4f_hp010__R(ALL)_M(0)_MIN(5)_CT(0.5)_ADJ(NO)_CS(D)_BB(S).pkl',
                              'Viterbi-MLT__viterbi_adjust4f_a_hp182__R(ALL)_M(0.89)_MIN(5)_CT(0.48)_ADJ(NO)_CS(D)_BB(S).pkl',
                              'Viterbi-SLT__viterbi_adjust4e_hp056__R(ONE)_M(0.95)_MIN(5)_CT(0.45)_ADJ(NO)_CS(D)_BB(S).pkl',
                              'Feature_weighted__cell_tracking_algorithm_bon_1c_hp016__R(ALL)_M(0.4)_MIN(5)_CT(0.5)_ADJ(NO)_CS(A)_BB(S).pkl',
                              'viterbi_adjust4f_hp001__R(ALL)_M(0.875)_MIN(5)_CT(0.445)_ADJ(NO)_CS(D)_BB(S).pkl'
                              # 'viterbi_results_dict_adj2.pkl'
                              ]
    method_name_pkl_dict: dict = {}
    for file_name in preload_file_name_list:
        track_file_path = open_track_dictionary(pkl_dir + file_name)
        file_name = file_name.replace(".pkl", "").replace("_results_dict", "").replace("_allseries_unet", "")

        if "__" in file_name:
            file_name = file_name[0: file_name.index("__")]

        file_name = file_name.replace("_", " ")

        method_name_pkl_dict[file_name] = track_file_path


    min_fit: float = 0.0151
    min_fio: float = 0.2368
    max_tp: float = 0.7465
    max_op: float = 0.2862

    series_list: list[str] = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                                'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20'] # enter all tracked images series_list
    
    segmented_file_list: list = listdir(segmentation_folder)
    segmented_file_list.sort()

    result_log_list: list = []
    complete_tracking_performance_list: list = []
    data_list_list: list = []
    total: int = len(method_name_pkl_dict.keys())
    for idx, (method_name, result_dict) in enumerate(method_name_pkl_dict.items()):
        print(f"{idx+1} / {total}; method_name: {method_name}; ", end='')
        tracking_performance_list_list: list = []
        for series in series_list:
            file_list: list = []
            for filename in segmented_file_list:
                if series in filename:
                    file_list = file_list + [filename]


            estimate_tracks_tub_list: list = result_dict[series]
            ground_truth_tracks_tub_list: list = gt_results_dict[series]

            gt_idmap = create_gt_idmap(ground_truth_tracks_tub_list, estimate_tracks_tub_list)
            estimate_idmap = create_estimate_idmap(ground_truth_tracks_tub_list, estimate_tracks_tub_list)

            FIT_and_FIO_tub_list: list = get_FIT_and_FIO(len(file_list), gt_idmap, estimate_idmap)
            FIT_list: list[float] = [i[0]/i[2] for i in FIT_and_FIO_tub_list]
            FIO_list: list[float] = [i[1]/i[2] for i in FIT_and_FIO_tub_list]

            TP_list: list[float] = None
            OP_list: list[float] = None
            TP_list, OP_list = get_TP_and_OP(len(file_list), gt_idmap, estimate_idmap)


            tracking_performance_list = [FIT_list, FIO_list, TP_list, OP_list, estimate_tracks_tub_list]
            tracking_performance_list_list.append(tracking_performance_list)
        

        #average tracking performance
        all_train_test_val_list_list_list: list = [
                                                    tracking_performance_list_list, #all
                                                    [tracking_performance_list_list[index] for index in [0,1,2,4,5,6,7,8,10,11,12,13,14,16,17,18]], #train[0,1,2,4,5,6,7,8,10,11,12,13,14,16,17,18]
                                                    [tracking_performance_list_list[index] for index in [15,19]],   # test[15,19]
                                                    [tracking_performance_list_list[index] for index in [3,9]]      # val[3,9]
                                                   ] 


        for idx, all_train_test_val_list_list in enumerate(all_train_test_val_list_list_list):
            FIT_complete_list: list[float] = [val for sublist in [performance[0] for performance in all_train_test_val_list_list] for val in sublist]
            FIO_complete_list: list[float] = [val for sublist in [performance[1] for performance in all_train_test_val_list_list] for val in sublist]
            TP_complete_list: list[float] = [val for sublist in [performance[2] for performance in all_train_test_val_list_list] for val in sublist]
            OP_complete_list: list[float] = [val for sublist in [performance[3] for performance in all_train_test_val_list_list] for val in sublist]
            estimate_tub_list: list = [val for sublist in [performance[4] for performance in all_train_test_val_list_list] for val in sublist]

            norm_FIT: float = np.mean(FIT_complete_list)
            norm_FIO: float = np.mean(FIO_complete_list)
            norm_TP: float = np.mean(TP_complete_list)
            norm_OP: float = np.mean(OP_complete_list)
            sd_FIT: float = np.std(FIT_complete_list)
            sd_FIO: float = np.std(FIO_complete_list)
            sd_TP: float = np.std(TP_complete_list)
            sd_OP: float = np.std(OP_complete_list)

            track_length_list: list[int] = [len(set([cell[1] for cell in track])) for track in estimate_tub_list]

            average_track_length: float = sum(track_length_list)/len(estimate_tub_list)
            track_length_sd: float = np.std(track_length_list)

            if idx == 2:
                data_list_list.append([FIT_complete_list, FIO_complete_list, TP_complete_list, OP_complete_list, track_length_list])
                better_count = 0
                if norm_FIT < min_fit: better_count += 1
                if norm_FIO < min_fio: better_count += 1
                if norm_TP > max_tp:   better_count += 1
                if norm_OP > max_op:   better_count += 1

                prefix = "\t\t" * better_count
                log_msg = f"{prefix}better_count: {better_count}; {method_name}, {round(norm_FIT, 4)}, {round(norm_FIO, 4)}, {round(norm_TP, 4)}, {round(norm_OP, 4)}"
                result_log_list.append(log_msg)
                print(log_msg)


            complete_tracking_performance_list.extend([[norm_FIT, sd_FIT, norm_FIO, sd_FIO,
                                                   norm_TP, sd_TP, norm_OP, sd_OP,
                                                   average_track_length, track_length_sd]])

    result_log_list.sort()
    for result_log in result_log_list:
        print(result_log)


    # https://www.google.com/search?q=The+requested+array+has+an+inhomogeneous+shape+after+2+dimensions.+The+detected+shape+was+(1%2C+5)+%2B+inhomogeneous+part.&rlz=1C1ONGR_enCA1072CA1072&oq=The+requested+array+has+an+inhomogeneous+shape+after+2+dimensions.+The+detected+shape+was+(1%2C+5)+%2B+inhomogeneous+part.&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRiPAjIHCAIQIRiPAjIHCAMQIRiPAtIBBzE2NmowajeoAgCwAgA&sourceid=chrome&ie=UTF-8
    data_list_list = np.array(data_list_list, dtype=object).T
    np.savetxt(output_put_dir + "tracking_performance.csv", complete_tracking_performance_list, delimiter=",")


    axlist = []
    width = len(method_name_pkl_dict.keys())
    height = 25
    total_chart_num = 5

    fig = plt.figure(figsize=(width, height))
    for idx in range(total_chart_num):
        ax = plt.subplot2grid(shape=(total_chart_num, 1), loc=(idx, 0))
        axlist.append(ax)

    value = ['Falsely Identified Tracker (FIT)', 'Falsely Identified Object (FIO)', 'Track Purity (TP)', 'Object Purity (OP)', 'average track length']

    methods = []
    for method_name in method_name_pkl_dict.keys():
        methods.append(method_name)


    for i, ax in enumerate(axlist):
        ax.boxplot(data_list_list[i], showmeans=True)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_title(value[i], fontsize=25)
        ax.grid(axis = 'y')
        ax.set_xticklabels(methods, rotation=20,  horizontalalignment="center", fontsize=15)

    # Save the figure and show
    plt.tight_layout(pad=0.4, w_pad=5, h_pad=1.0)
    output_file_path = output_put_dir + 'performance_metrics.png'
    print(output_file_path)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.show()







'''opening a track dictionary from file'''
def open_track_dictionary(save_file):
    pickle_in = open(save_file,"rb")
    dictionary = pickle.load(pickle_in)
    return dictionary



#create identification map w.r.t. GT
def create_gt_idmap(ground_truth_tracks, estimate):
    gt_idmap = []
    for gt_track in ground_truth_tracks:

        #get most similar track to ground truth track
        sim = []
        for track in estimate:
            gt_track_adj = [i[0:2] for i in gt_track]
            track_adj = [i[0:2] for i in track]
            res = len(set(gt_track_adj) & set(track_adj))
            sim.append(res)

        index_max = np.argmax(sim)

        if max(sim) > 0:
            gt_idmap.append((gt_track, estimate[index_max]))

    return gt_idmap



#create identification map w.r.t. estimate
def create_estimate_idmap(ground_truth_tracks, estimate):
    estimate_idmap = []
    for track in estimate:
        #get most similar track to ground truth track
        sim = []
        for gt_track in ground_truth_tracks:
            gt_track_adj = [i[0:2] for i in gt_track]
            track_adj = [i[0:2] for i in track]
            res = len(set(gt_track_adj) & set(track_adj))
            sim.append(res)
        index_max = np.argmax(sim)
        if max(sim) > 0:
            estimate_idmap.append((track, ground_truth_tracks[index_max]))
    return estimate_idmap



# Count falsely identified tracker error (FIT), 
# falsely identified object error (FIO) per frame
def get_FIT_and_FIO(framenbs, gt_idmap, estimate_idmap):
    #find the FIT and FIO errors per frame
    FF_list = []
    for framenb in range(framenbs):
        gt_in_frame = 0
        FIT = 0
        #Count the FITs in this frame
        for gt_track, estimate in gt_idmap:
            #get the gt cell number in this frame
            curr_gt_cell = [ i[0] for i in gt_track if i[1] == framenb]
            # if a ground truth track is present in this frame, 
            # check if estimate is the same, otherwise count FIT
            if curr_gt_cell:
                gt_in_frame += 1
                curr_est_cells = [ i[0] for i in estimate if i[1] == framenb]
                if curr_est_cells:
                    for curr_est_cell in curr_est_cells:
                        if not int(curr_est_cell) == int(curr_gt_cell[0]):
                            FIT = FIT + 1

        FIO = 0
        #count the FIOs in this frame
        for estimate, gt_track in estimate_idmap:
            #get the estimate cell numbers in this frame
            curr_est_cells = [ i[0] for i in estimate if i[1] == framenb]
            # if the ground truth track is present in this frame, 
            # check if estimate is the same, otherwise count FIO
            curr_gt_cell = [ i[0] for i in gt_track if i[1] == framenb]
            if curr_gt_cell:
                if curr_est_cells:
                    for curr_est_cell in curr_est_cells:
                        if not int(curr_est_cell) == int(curr_gt_cell[0]):
                            FIO = FIO + 1
        
        if gt_in_frame > 0:
            FF_list.append((FIT,FIO,gt_in_frame))
    
    return FF_list

#get Tracker Purity (TP) and Object Purity (OP) for all ground truths and estimates
def get_TP_and_OP(framenbs, gt_idmap, estimate_idmap):
    TP_list = []
    OP_list = []
    OP_min, TP_min = 1, 1
    OP_max, TP_max = 0, 0

    #calculate OP for each object (ground truth)
    for gt_track, estimate in gt_idmap:
        corr_gt = 0 # number of correctly identified ground truths
        frames_gt = 0 # number of frames the ground truth is present
        for framenb in range(framenbs):
            #get the gt cell number in this frame
            curr_gt_cell = [ i[0] for i in gt_track if i[1] == framenb]
            # if the ground truth track is present in this frame, 
            # check if estimate is the same
            if curr_gt_cell:
                frames_gt += 1
                curr_est_cells = [ i[0] for i in estimate if i[1] == framenb]
                if curr_est_cells:
                    correctly_identified = False
                    for curr_est_cell in curr_est_cells:
                        if int(curr_est_cell) == int(curr_gt_cell[0]):
                            correctly_identified = True
                    if correctly_identified:
                        corr_gt += 1
        OP = corr_gt/frames_gt
        OP_list.append(OP)
    
    #calculate TP for each tracker (estimate)
    for estimate, gt_track in estimate_idmap:
        corr_est = 0 # number of correctly identified estimates
        frames_est = 0 # number of frames the estimate is present
        for framenb in range(framenbs):
            #get the estimate cell numbers in this frame
            curr_est_cells = [ i[0] for i in estimate if i[1] == framenb]
            # if the ground truth track is present in this frame, 
            # check if estimate is the same, otherwise count FIO
            curr_gt_cell = [ i[0] for i in gt_track if i[1] == framenb]
            if curr_gt_cell: # skip this frame if no gt is present
                if curr_est_cells:
                    frames_est += 1
                    correctly_identified = False
                    for curr_est_cell in curr_est_cells:
                        if int(curr_est_cell) == int(curr_gt_cell[0]):
                            correctly_identified = True
                    if correctly_identified:
                            corr_est += 1
        TP = corr_est/frames_est
        TP_list.append(TP)
    return TP_list, OP_list




if __name__ == '__main__':
    main()