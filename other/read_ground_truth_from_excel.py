'''read in ground-truth tracks from excel files'''

import os.path
import seaborn as sns
from imutils import paths
sns.set()
#loop through all rows in tracking file and add samples to training file
def CreateTracks(base, img_folder, segall_folder, Tracked):
    tracks = []
    track = []
    framenb = 0
    i = 0
    #go through all rows of the excel file,
    #where each row is a single tracked cell location
    while i < len(Tracked):
        #first frame of track
        if i == 0 or Tracked[i,0] != Tracked[i-1,0]:
            framenb = 0
            if not i == 0:
                #save old track, when starting new track
                tracks.append(track)
            track = []
            #get cellnumber of first cell in track
            cell_nb = 0
            while cell_nb == 0 and i < len(Tracked):
                framenb = framenb + 1
                filename = '{}{:03d}.png'.format(base, framenb)
                segall = plt.imread(segall_folder + filename)
                labeled = measure.label(segall, background=0,connectivity=1)
                cell_nb = labeled[int(Tracked[i,3]),int(Tracked[i,2])]
                i = i + 1
            # a cell was correctly identified, save this cellnb to track
            if cell_nb != 0:
                #start cellnumbering from 0
                cell = (cell_nb-1, framenb-1, -1)
                track.append(cell)
        #daughter cell (not in this dataset, but in case of new dataset)
        elif Tracked[i,1] == Tracked[i-1,1]:
            continue
        else:
            framenb=framenb+1
            cellnb_prev = cell_nb

            #get filename of current and previous slice
            filename = '{}{:03d}.png'.format(base, framenb)

            #retrieve relevant images for current tracked cell
            segall = plt.imread(segall_folder + filename)

            #label the semented images
            labeled = measure.label(segall, background=0,connectivity=1)

            # get the cell label of the selected cell in the current and
            # previous frame from the pixel (x,y) of the excel file
            cell_nb = labeled[int(Tracked[i,3]),int(Tracked[i,2])]
            if cell_nb != 0:
                cell = (cell_nb-1, framenb-1, cellnb_prev-1)
                props = measure.regionprops(labeled)
                track.append(cell)
            else:
                cell_nb = cellnb_prev

            i = i + 1
    tracks.append(track)
    return tracks

segmentation_folder = '/content/drive/MyDrive/Tracking_simpledata/Unet_trainingsets_simpledata/data/segmentation/'
images_folder = '/content/drive/MyDrive/Tracking_simpledata/Unet_trainingsets_simpledata/data/images/'
#The excel files contain the ground truth tracks
excel_path = PROJECT_PATH + '/Tracking_simpledata/manually tracking Ground Truth excel/'
excel_folders = [excel_path + 'neutrophil/long/',excel_path + 'neutrophil/local/']
rawdata_folders = '/content/drive/MyDrive/Tracking_simpledata/raw data Stack/'

gt_results_dict = {
    "S01": [],
    "S02": [],
    "S03": [],
    "S04": [],
    "S05": [],
}

for excel_folder in excel_folders:

    pot_samples = listdir(excel_folder)

    for filename in pot_samples:
        #these segmentations are missing from the dataset
        print(filename)
        Tracked = np.genfromtxt(excel_folder+filename, delimiter=',', encoding='ISO-8859-1', skip_header=1)
        Tracked = Tracked[~np.isnan(Tracked).any(axis=1)]

        if '--' in filename:
            condition = filename[filename.find('--'):filename.find('--')+3]
        else:
            condition = filename[filename.find('++'):filename.find('++')+3]
        print(condition)
        if '190621' in filename:
            celldate = '190621'
        elif '190701' in filename:
            celldate = '190701'
        elif '200716' in filename:
            celldate = '200716'
        elif '200802' in filename:
            celldate = '200802'
        else:
            celldate = '200829'
        print(celldate)
        imgPaths = list(paths.list_images(rawdata_folders))
        for imgfile in imgPaths:
            #print(imgfile)
            if celldate in imgfile and condition in imgfile:
                serie = imgfile[-7:-4]
        print(serie)
        base = celldate + '_' + condition + '_' + serie + '_frame'

        ground_truth_tracks = CreateTracks(base, images_folder, segmentation_folder, Tracked)

        identifier = serie
        print(identifier)
        print(len(ground_truth_tracks), "cells tracked")
        # for track in ground_truth_tracks:
        #     print(track)
        gt_results_dict[identifier] = ground_truth_tracks