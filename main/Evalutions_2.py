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
import cv2
import random
from itertools import combinations
import pickle

root_dir: str = 'D:/viterbi linkage/dataset/'
save_dir = root_dir + 'Dicts/'                               #'track_dictory_finetune//track_dictory_unet_seg//'
segmentation_folder = root_dir + 'segmentation_unet_seg/'    #'Unet_trainingsets//data//segmentation_unet_seg//'
output_put_dir: str = root_dir + 'evaluation_output/'

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





kuan_tracks_allseries_unet = open_track_dictionary(save_dir + "kuan_tracks_allseries_unet.pkl")  
delta_results_dict = open_track_dictionary(save_dir + "delta_results_dict.pkl")
hungarian_results_dict = open_track_dictionary(save_dir + "hungarian_results_dict.pkl")
viterbi_results_dict = open_track_dictionary(save_dir + "viterbi_results_dict_adj2.pkl")
viterbi_2_results_dict = open_track_dictionary(save_dir + "viterbi_results_dict_2.pkl")
gt_results_dict = open_track_dictionary(save_dir + "gt_results_dict.pkl")

result_dicts = [('kuan_tracks_allseries_unet',kuan_tracks_allseries_unet),
                ('delta_results_dict',delta_results_dict),
                ('hungarian_results_dict',hungarian_results_dict),
                ('viterbi_results_dict',viterbi_results_dict),
                ('viterbi_2_results_dict',viterbi_2_results_dict)]

series = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20'] # enter all tracked images series
#celltypes = ['C1'] # enter all tracked celllines

segmented_files = listdir(segmentation_folder)
segmented_files.sort()

complete_tracking_performance = []
data = []
for name, result_dict in result_dicts:
    print(name)
    tracking_performance_list = []
    for serie in series:
        #for celltype in celltypes:
        print(serie)
        #no ground truth tracks for C29 C1
        identifier = serie
        filelist = []
        #select all files of the current images series and celltype
        for filename in segmented_files:
            if serie in filename:
                filelist = filelist + [filename]
            

        estimate = result_dict[identifier]
        ground_truth_tracks = gt_results_dict[identifier]

        gt_idmap = create_gt_idmap(ground_truth_tracks, estimate)
        estimate_idmap = create_estimate_idmap(ground_truth_tracks, estimate)

        FIT_and_FIO = get_FIT_and_FIO(len(filelist), gt_idmap, estimate_idmap)
        FIT_list = [i[0]/i[2] for i in FIT_and_FIO]
        FIO_list = [i[1]/i[2] for i in FIT_and_FIO]

        TP_list, OP_list = get_TP_and_OP(len(filelist), gt_idmap, estimate_idmap)

        tracking_performance_list.append([FIT_list,FIO_list,TP_list,OP_list,estimate])


    #average tracking performance
    alltraintestval_list = [tracking_performance_list, #all
                            [tracking_performance_list[index] for index in [0,1,2,4,5,6,7,8,10,11,12,13,14,16,17,18]], #train[0,1,2,4,5,6,7,8,10,11,12,13,14,16,17,18]
                            [tracking_performance_list[index] for index in [15,19]], # test[15,19]
                            [tracking_performance_list[index] for index in [3,9]]] # val[3,9]


    for idx, alltraintestval in enumerate(alltraintestval_list):
        FIT_complete = [val for sublist in [performance[0] for performance in alltraintestval] for val in sublist]
        FIO_complete = [val for sublist in [performance[1] for performance in alltraintestval] for val in sublist]
        TP_complete = [val for sublist in [performance[2] for performance in alltraintestval] for val in sublist]
        OP_complete = [val for sublist in [performance[3] for performance in alltraintestval] for val in sublist]
        estimate_list = [val for sublist in [performance[4] for performance in alltraintestval] for val in sublist]

        norm_FIT = np.mean(FIT_complete)
        norm_FIO = np.mean(FIO_complete)
        norm_TP = np.mean(TP_complete)
        norm_OP = np.mean(OP_complete)
        sd_FIT = np.std(FIT_complete)
        sd_FIO = np.std(FIO_complete)
        sd_TP = np.std(TP_complete)
        sd_OP = np.std(OP_complete)
        track_lengths = [len(set([cell[1] for cell in track])) for track in estimate_list]
        
        average_track_length = sum(track_lengths)/len(estimate_list)
        track_length_sd = np.std(track_lengths)

        if idx == 2:
            data.append([FIT_complete,FIO_complete,TP_complete,OP_complete,track_lengths])
        complete_tracking_performance.extend([[norm_FIT, sd_FIT, norm_FIO, sd_FIO,
                                               norm_TP, sd_TP, norm_OP, sd_OP,
                                               average_track_length, track_length_sd]])


data = np.array(data).T
np.savetxt(output_put_dir + "tracking_performance.csv", complete_tracking_performance, delimiter=",")

# Create plot of performance metrics
fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

value = ['normalized FIT','normalized FIO','normalized TP','normalized OP','average track length']
methods = ['KDE method', 'DeLTA method', 'Hungarian method', 'Viterbi method', 'Viterbi 2 method']

meanvalue = []
medianvalue = []
for i in range(5):
    aaa = data[i]
    bbb = data[i][0]
    ccc = data[i][1]
    ddd = data[i][2]
    eee = data[i][3]

    mbb = np.mean(bbb)
    medianbb = np.median(bbb)

    mcc = np.mean(ccc)
    mediancc = np.median(ccc)

    mdd = np.mean(ddd)
    mediandd = np.median(ddd)

    mee = np.mean(eee)
    medianee = np.median(eee)

    meanvalue.append(mbb)
    meanvalue.append(mcc)
    meanvalue.append(mdd)
    meanvalue.append(mee)

    medianvalue.append(medianbb)
    medianvalue.append(mediancc)
    medianvalue.append(mediandd)
    medianvalue.append(medianee)
print(meanvalue) 
print(medianvalue)   

axlist = [ax1,ax2,ax3,ax4,ax5]
for i, ax in enumerate(axlist):
    ax.boxplot(data[i], showmeans=True)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title(value[i], fontsize=25)
    ax.grid(axis = 'y')
    ax.set_xticklabels(methods, rotation=10,  horizontalalignment="center", fontsize=15)

# Save the figure and show
plt.tight_layout(pad=0.4, w_pad=5, h_pad=1.0)
plt.savefig(output_put_dir + 'performance_metrics.png', bbox_inches='tight')
plt.show()