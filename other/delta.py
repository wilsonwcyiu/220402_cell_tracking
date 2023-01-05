np.set_printoptions(edgeitems=30, linewidth=100000)
def getAttributions(scores):
    '''
    Get attribution matrix from tracking scores

    Parameters
    ----------
    scores : numpy 3D array of floats
        Tracking scores matrix as produced by the getTrackingScores function.

    Returns
    -------
    attrib : numpy 3D array of bools
        Attribution matrix. Cells from the old frame (axis 0) are attributed to
        cells in the new frame (axis 1). Each new cell can be attributed to an
        old cell as a "mother", ie the same cell (index 0 along axis 2)
        Example:
            attrib[1,3] == True means cell #4 in the new frame (from top to 
            bottom) is attributed to cell #2 in the old frame.
        Each 'old' cell can have only 0 or 1 'new' cell identified as itself 
        (ie mother-mother relationship).
    '''
    attrib = np.zeros(scores.shape)
    for i in range(scores.shape[1]): # Go over cells in the *new* frame
        if np.sum(scores[:,i]>0) == 1: # If only one old cell was attributed to the new cell
            attrib[:,i] = scores[:,i]>0
        elif np.sum(scores[:,i]>0) > 1: # If conflicts
            if np.sum(scores==np.max(scores[:,i])) == 1: # One cell has a higher score than the others
                attrib[:,i] = scores[:,i]==np.max(scores[:,i])
            else: # If only mother-to-mother couplings
                attrib[np.argmax(scores[:,i]),i]=True # keep only the first one (which is also the one higher in the image)
    for o in range(scores.shape[0]): # Go over cells in the *old* frame
        if np.sum(attrib[o,:]>0) > 1: # If one old cell gets attributed to more than one new cell (mother-mother coupling)
            tokeep = np.argmax(attrib[o,:])
            attrib[o,:]=False
            attrib[o,tokeep]=True # keep only the first one
    return attrib

def getSinglecells(seg, maxcellnb):
    '''
    Return masks of single cells

    Parameters
    ----------
    seg : array of uint8/uint16/float/bool
        Mask of cells. Values >0.5 will be considered cell pixels

    Returns
    -------
    singlecellseg : 3D array of uint8
        Stack of single cell masks. Each single-cell mask is stacked along the
        first axis (axis 0)
    '''

    singlecellseg = np.empty([maxcellnb,seg.shape[0],seg.shape[1]])
    for cellnb in range(maxcellnb):
        singlecellseg[cellnb] = seg == cellnb +1
    return singlecellseg

def getOverlap(output,target):
    '''
    Get portion of tracking output overlapping on target cell

    Parameters
    ----------
    output : array of uint8
        Mask of tracking output.
    target : array of uint8
        Mask of target cell.

    Returns
    -------
    float
        Portion of target cell that is covered by the tracking output.

    '''
    return  cv2.sumElems(cv2.bitwise_and(output,target))[0]/   \
            cv2.sumElems(target)[0]

delta_results_dict = {
    "S01": [],
    "S02": [],

      
}

series = ['S01', 'S02'] # enter all tracked images series
#celltypes = ['C1'] # enter all tracked celllines

segmented_files = listdir(segmentation_folder)
segmented_files.sort()
for serie in series:
    #if data is not complete
    if not serie in listdir(output_folder):
        continue 
    #for celltype in celltypes:
    print(serie)
    filelist = []
    img_list = []
    labeled_img_list = []
    #select all files of the current images series and celltype
    for filename in segmented_files:
        if serie in filename:
            filelist = filelist + [filename]

    #get the first image (frame 0) and label the cells:
    img = plt.imread(segmentation_folder + filelist[0])
    img_list.append(img)
    labeled_img = measure.label(img, background=0,connectivity=1)
    labeled_img_list.append(labeled_img)
    cellnb_img = np.max(labeled_img)

    attributions = []
    for framenb in range(1,len(filelist)):
        #get next frame and number of cells next frame
        img_next = plt.imread(segmentation_folder +'/' + filelist[framenb])
        img_list.append(img_next)
        labeled_img_next = measure.label(img_next, background=0,connectivity=1)
        labeled_img_list.append(labeled_img_next)
        cellnb_img_next = np.max(labeled_img_next)
 
        targetcells = getSinglecells(labeled_img_next, cellnb_img_next).astype(np.uint8)
        scores = np.zeros([cellnb_img,cellnb_img_next],dtype=np.float32)
        for cellnb_i in range(cellnb_img):
            cell_i_filename = "mother_" + filelist[framenb][:-4] + "_Cell" + str(cellnb_i+1).zfill(2) + ".png"
            cell_i = plt.imread(output_folder + serie +'/' + cell_i_filename)
            cell_i = cv2.threshold(cell_i,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
            for cellnb_j in range(cellnb_img_next):
                scores[cellnb_i,cellnb_j] = getOverlap(cell_i,targetcells[cellnb_j,:,:])
        attrib = getAttributions(scores)
        attributions.append(attrib)
        #make next frame current frame
        cellnb_img = cellnb_img_next
        labeled_img = labeled_img_next
        
    tracks = []
    for i in range(attributions[0].shape[0]):
        track = []
        track.append((i,0,-1))
        tracks.append(track)
    for framenb, attrib in enumerate(attributions):
        for i in range(attrib.shape[1]):
            if np.any(attrib[:,i]): # If cell is tracked from an old cell: (mother-mother)
                prev_cellnb = int(np.nonzero(attrib[:,i])[0])
                selected_track_idx = [idx for idx, track in enumerate(tracks) if (track[-1][0] == prev_cellnb and track[-1][1] == framenb)]
                tracks[int(selected_track_idx[0])].append((i,framenb+1,prev_cellnb))
            else: # Orphan cell
                track = []
                track.append((i,framenb+1,-1))
                tracks.append(track)
        
    tracks_final = []
                
    #settings
    DELTA_TIME = 5
    for track in tracks:
        if len(track) >= DELTA_TIME:
            tracks_final.append(track)

    identifier = serie
    delta_results_dict[identifier] = tracks_final
print(delta_results_dict)