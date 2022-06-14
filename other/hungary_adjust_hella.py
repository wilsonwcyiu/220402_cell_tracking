
''' Algorithm: macrophage tracking adjusted for merge and oversegmentation'''
# from google.colab import drive
# from google.colab.patches import cv2_imshow

import os
from collections import defaultdict
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

def tracking_adj(associations, prof_mat_list, DELTA_TIME):
    tracks = []
    #create tracks for all cells in the first frame
    for i in range(len(prof_mat_list[0])):
        track = []
        track.append((i,0,-1))
        tracks.append(track)
    #loop through the frames
    for t in range(len(associations)):
        for tracknb, track in enumerate(tracks):
            #get all cells in current track in current frame
            cells = [cell for cell in track if cell[1] == t]
            for cell in cells:
                #get all tracks going through the current cell, 
                # which have not been extended yet
                if tracknb < len(tracks)-1:
                    count = 1 #number of tracks going through current cell
                    tracks_through_same_cell = []
                    for track_idx in range(tracknb+1, len(tracks)):
                        curr_track = tracks[track_idx]
                        cells_in_track = [i[0] for i in curr_track if i[1] == t]
                        if cell[0] in cells_in_track:
                            tracks_through_same_cell.append(curr_track)
                            count += 1

                #get associations indexes of all cells in the next frame of the current cell
                pair_idxs = np.where(associations[t][0] == cell[0])[0]
                
                #end of track
                if len(pair_idxs) == 0:
                    continue

                # extend curr track with extra outgoing tracks if
                # no. of outgoing tracks len(pair_idxs) > no. of incoming tracks (count)
                no_cells_to_add = len(pair_idxs)-count
                
                for i in range(no_cells_to_add):
                    #retrieve the pair and add them to the track
                    pair = np.array(associations[t]).transpose()[int(pair_idxs[i])]
                    if not prof_mat_list[t][pair[0], pair[1]] < 0.001:
                        track.append((pair[1],t+1,cell[0]))
                        C[t] = np.delete(associations[t], pair_idxs[i], axis=1)
                
                #get associations indexes again after deletions
                pair_idxs = np.where(associations[t][0] == cell[0])[0]

                # extend curr track with 1 outgoing tracks
                pair = np.array(associations[t]).transpose()[int(pair_idxs[0])]
                if not prof_mat_list[t][pair[0], pair[1]] < 0.001:
                    if pair[0] != cell[0]:
                        print(t)
                        print(cell[0])
                        print(pair[0])
                        print(track)
                    track.append((pair[1],t+1,cell[0]))
                if len(pair_idxs) >= count:
                    #delete the appended pair if not in another track
                    C[t] = np.delete(associations[t], pair_idxs[0], axis=1)

        #create new tracks for all left over cells for the current frame
        for pair in np.array(associations[t]).transpose():
            if not prof_mat_list[t][pair[0], pair[1]] < 0.001:
                track = []
                track.append((pair[1],t+1,-1))
                tracks.append(track)
            
    #Add track to track list if longer than DELTA_TIME
    tracks_final = []
    for track in tracks:
        if len(set([cell[1] for cell in track])) >= DELTA_TIME:
            tracks_final.append(track)
    delete_list = []
    #if a track is for more than 80% similar to another track, delete it
    for i, j in combinations(range(len(tracks_final)), 2):
        res = len(set(tracks_final[i]) & set(tracks_final[j])) / min(len(tracks_final[i]), len(tracks_final[j]))
        if res > 0.8:
            if len(tracks_final[i]) > len(tracks_final[j]):
                delete_list.append(j)
            else:
                delete_list.append(i)
    tracks_final = np.delete(tracks_final, delete_list, axis=0)
    return tracks_final



'''save track dictionary'''
def save_track_dictionary(dictionary, save_file):
    if not os.path.exists(save_file):
        with open(save_file, 'w'):
            pass
    pickle_out = open(save_file,"wb")
    pickle.dump(dictionary,pickle_out)
    pickle_out.close()



''' Create profit matrix for hungarian algorithm with adjustments for merge '''



if __name__ == '__main__':

    input_series_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                         'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    # input_series_list = ['S02']


    hungarian_adj_results_dict = defaultdict(list)


    # input_series_list = ['S10'] # enter all tracked images series
    #celltypes = ['C1'] # enter all tracked celllines

    folder_path: str = 'D:/viterbi linkage/dataset/'

    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    output_folder = folder_path + 'output_unet_seg_finetune//'
    PROJECT_PATH = folder_path
    save_dir = PROJECT_PATH + 'save_directory_enhancement/'


#settings
    DELTA_TIME = 5
    SPLIT_THRESHOLD = 0.4
    MERGE_THRESHOLD = 0.4
    MIN_LEN = 10


    segmented_files = listdir(segmentation_folder)
    segmented_files.sort()
    for serie in input_series_list:
        #for celltype in celltypes:
        print(serie)
        filelist = []
        img_list = []
        img_real_list = []
        label_img_list = []
        #select all files of the current images series and celltype
        for filename in segmented_files:
            if serie in filename:
                filelist = filelist + [filename]
        C = []
        prof_mat_list = []
        #get the first image (frame 0) and label the cells:
        img = plt.imread(segmentation_folder +'/' + filelist[0])
        img_list.append(img)
        label_img = measure.label(img, background=0,connectivity=1)
        label_img_list.append(label_img)
        cellnb_img = np.max(label_img)

        for framenb in range(1,len(filelist)):
            #print(framenb)

            #get next frame and number of cells next frame
            img_next = plt.imread(segmentation_folder +'/' + filelist[framenb])
            img_list.append(img_next)
            label_img_next = measure.label(img_next, background=0,connectivity=1)
            label_img_list.append(label_img_next)
            cellnb_img_next = np.max(label_img_next)

            #create empty dataframe for element of profit matrix C
            prof_mat = np.zeros((cellnb_img, cellnb_img_next), dtype=float)


            #loop through all combinations of cells in this and the next frame
            for cellnb_i in range(cellnb_img):
                cell_i_filename = "mother_" + filelist[framenb][:-4] + "_Cell" + str(cellnb_i+1).zfill(2) + ".png"
                cell_i = plt.imread(output_folder + serie +'/'+ cell_i_filename)
                cell_i_props = measure.regionprops(label_img_next,intensity_image=cell_i)
                for cellnb_j in range(cellnb_img_next):
                    #calculate profit score from mean intensity neural network output in segmented cell area
                    prof_mat[cellnb_i,cellnb_j] = cell_i_props[cellnb_j].mean_intensity

            prof_mat = prof_mat/np.max(prof_mat)

            #create separate columns for cells
            #which have a high score for multiple cells in the previous frames
            prof_mat_bool = prof_mat > MERGE_THRESHOLD # every cell combination with a score higher than 0.4

            #Get all columns and rows with two at least two cell combinations with a score higher than 0.4
            double_match_columns = prof_mat_bool.sum(axis=0)>1


            #coordinates of the non-zero cells in these columns
            nonzero_cols = np.argwhere(prof_mat_bool*double_match_columns)
            nonzero_cols = list(nonzero_cols)
            nonzero_cols.sort(key=lambda tup: tup[1])
            nonzero_cols = np.array(nonzero_cols)
            new_cols = len(nonzero_cols) - sum(double_match_columns)

            #create matrix with extra columns
            prof_mat_adj = np.zeros((len(prof_mat),len(prof_mat[0])+new_cols), dtype=float)
            prof_mat_adj[0:len(prof_mat), 0:len(prof_mat[0])] = prof_mat

            if not len(nonzero_cols) == 0:
                col = -1
                row = []
                extra = 0
                #move double elements cols
                for el in nonzero_cols:
                    new_col = el[1]
                    if new_col == col:
                        value = copy.copy(prof_mat_adj[el[0],el[1]])
                        prof_mat_adj[el[0],el[1]] = 0
                        prof_mat_adj[el[0], len(prof_mat[0])+extra] = value
                        extra += 1
                    else:
                        col = new_col

            prof_mat_bool_rows = prof_mat_adj > SPLIT_THRESHOLD
            double_match_rows = prof_mat_bool_rows.sum(axis=1)>1
            nonzero_rows = np.argwhere((prof_mat_bool_rows.T*double_match_rows).T)
            new_rows = len(nonzero_rows) - sum(double_match_rows)
            length = max(len(prof_mat_adj)+new_rows, len(prof_mat_adj[0]))
            prof_mat_final = np.zeros((length,length), dtype=float)

            #add orignal profit matrix with adjusted size to list
            prof_mat_final[0:len(prof_mat), 0:len(prof_mat[0])] = prof_mat
            prof_mat_list.append(copy.deepcopy(prof_mat_final))
            prof_mat_final[0:len(prof_mat_adj),0:len(prof_mat_adj[0])] = prof_mat_adj

            if not len(nonzero_rows) == 0:
                row = -1
                col = []
                extra = 0
                #move double elements rows
                for el in nonzero_rows:
                    new_row = el[0]
                    if new_row == row:
                        value = copy.copy(prof_mat_final[el[0],el[1]])
                        prof_mat_final[el[0],el[1]] = 0
                        prof_mat_final[len(prof_mat)+extra,el[1]] = value
                        extra += 1
                    else:
                        row = new_row

            Associations = linear_sum_assignment(prof_mat_final, maximize=True)


            #change rows in pairs back to original rows
            Associations_mod = copy.deepcopy(Associations)

            if not len(nonzero_rows) == 0:
                i = 0
                for idx,el in enumerate(Associations[0]):
                    if el > len(prof_mat)-1 and el < len(prof_mat) + new_rows:
                        if i < len(nonzero_rows)-1 and (i == 0 or not nonzero_rows[i][0] == nonzero_rows[i-1][0]):
                            i += 1
                        Associations_mod[0][idx] = nonzero_rows[i][0]
                        i += 1

            # change the order so the columns are ordered
            order = np.argsort(Associations[1])
            Associations = (np.take_along_axis(Associations[0], order, axis = 0),
                            np.take_along_axis(Associations[1], order, axis = 0))
            Associations_mod = (np.take_along_axis(Associations_mod[0], order, axis = 0),
                                np.take_along_axis(Associations_mod[1], order, axis = 0))

            #change columns in pairs back to original columns
            if not len(nonzero_cols) == 0:
                i = 0
                for idx, el in enumerate(Associations[1]):
                    #check if column number is from an added column
                    if el > len(prof_mat[0])-1 and el < len(prof_mat[0]) + new_cols:
                        #skip row in nonzero_cols if from unchanged cell
                        if i < len(nonzero_cols)-1 and (i == 0 or not nonzero_cols[i][1] == nonzero_cols[i-1][1]):
                            i += 1
                        Associations_mod[1][idx] = nonzero_cols[i][1]
                        i += 1

            # append element to profit matrix
            C.append(Associations_mod)

            #make next frame current frame
            cellnb_img = cellnb_img_next
            label_img = label_img_next

        result = tracking_adj(C, prof_mat_list, DELTA_TIME)
        #result = tracking_adj_2(C, prof_mat_list, DELTA_TIME MIN_LEN)
        identifier = serie
        hungarian_adj_results_dict[identifier] = result



    print(save_dir)
    save_track_dictionary(hungarian_adj_results_dict, save_dir + "hungarian_adj_results_dict.pkl")

