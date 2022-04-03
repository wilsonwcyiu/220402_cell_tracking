from main.viterbi_adjust2_enhancement import derive_prof_matrix_list
import numpy as np
import sys

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)


    folder_path: str = 'D:/viterbi linkage/dataset/'
    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    output_folder = folder_path + 'output_unet_seg_finetune//'

    segmented_filename_list = ['190621_++1_S01_frame001.png', '190621_++1_S01_frame002.png']
        # , '190621_++1_S01_frame003.png', '190621_++1_S01_frame004.png', '190621_++1_S01_frame005.png', '190621_++1_S01_frame006.png', '190621_++1_S01_frame007.png', '190621_++1_S01_frame008.png', '190621_++1_S01_frame009.png', '190621_++1_S01_frame010.png', '190621_++1_S01_frame011.png', '190621_++1_S01_frame012.png', '190621_++1_S01_frame013.png', '190621_++1_S01_frame014.png', '190621_++1_S01_frame015.png', '190621_++1_S01_frame016.png', '190621_++1_S01_frame017.png', '190621_++1_S01_frame018.png', '190621_++1_S01_frame019.png', '190621_++1_S01_frame020.png', '190621_++1_S01_frame021.png', '190621_++1_S01_frame022.png', '190621_++1_S01_frame023.png', '190621_++1_S01_frame024.png', '190621_++1_S01_frame025.png', '190621_++1_S01_frame026.png', '190621_++1_S01_frame027.png', '190621_++1_S01_frame028.png', '190621_++1_S01_frame029.png', '190621_++1_S01_frame030.png', '190621_++1_S01_frame031.png', '190621_++1_S01_frame032.png', '190621_++1_S01_frame033.png', '190621_++1_S01_frame034.png', '190621_++1_S01_frame035.png', '190621_++1_S01_frame036.png', '190621_++1_S01_frame037.png', '190621_++1_S01_frame038.png', '190621_++1_S01_frame039.png', '190621_++1_S01_frame040.png', '190621_++1_S01_frame041.png', '190621_++1_S01_frame042.png', '190621_++1_S01_frame043.png', '190621_++1_S01_frame044.png', '190621_++1_S01_frame045.png', '190621_++1_S01_frame046.png', '190621_++1_S01_frame047.png', '190621_++1_S01_frame048.png', '190621_++1_S01_frame049.png', '190621_++1_S01_frame050.png', '190621_++1_S01_frame051.png', '190621_++1_S01_frame052.png', '190621_++1_S01_frame053.png', '190621_++1_S01_frame054.png', '190621_++1_S01_frame055.png', '190621_++1_S01_frame056.png', '190621_++1_S01_frame057.png', '190621_++1_S01_frame058.png', '190621_++1_S01_frame059.png', '190621_++1_S01_frame060.png', '190621_++1_S01_frame061.png', '190621_++1_S01_frame062.png', '190621_++1_S01_frame063.png', '190621_++1_S01_frame064.png', '190621_++1_S01_frame065.png', '190621_++1_S01_frame066.png', '190621_++1_S01_frame067.png', '190621_++1_S01_frame068.png', '190621_++1_S01_frame069.png', '190621_++1_S01_frame070.png', '190621_++1_S01_frame071.png', '190621_++1_S01_frame072.png', '190621_++1_S01_frame073.png', '190621_++1_S01_frame074.png', '190621_++1_S01_frame075.png', '190621_++1_S01_frame076.png', '190621_++1_S01_frame077.png', '190621_++1_S01_frame078.png', '190621_++1_S01_frame079.png', '190621_++1_S01_frame080.png', '190621_++1_S01_frame081.png', '190621_++1_S01_frame082.png', '190621_++1_S01_frame083.png', '190621_++1_S01_frame084.png', '190621_++1_S01_frame085.png', '190621_++1_S01_frame086.png', '190621_++1_S01_frame087.png', '190621_++1_S01_frame088.png', '190621_++1_S01_frame089.png', '190621_++1_S01_frame090.png', '190621_++1_S01_frame091.png', '190621_++1_S01_frame092.png', '190621_++1_S01_frame093.png', '190621_++1_S01_frame094.png', '190621_++1_S01_frame095.png', '190621_++1_S01_frame096.png', '190621_++1_S01_frame097.png', '190621_++1_S01_frame098.png', '190621_++1_S01_frame099.png', '190621_++1_S01_frame100.png', '190621_++1_S01_frame101.png', '190621_++1_S01_frame102.png', '190621_++1_S01_frame103.png', '190621_++1_S01_frame104.png', '190621_++1_S01_frame105.png', '190621_++1_S01_frame106.png', '190621_++1_S01_frame107.png', '190621_++1_S01_frame108.png', '190621_++1_S01_frame109.png', '190621_++1_S01_frame110.png', '190621_++1_S01_frame111.png', '190621_++1_S01_frame112.png', '190621_++1_S01_frame113.png', '190621_++1_S01_frame114.png', '190621_++1_S01_frame115.png', '190621_++1_S01_frame116.png', '190621_++1_S01_frame117.png', '190621_++1_S01_frame118.png', '190621_++1_S01_frame119.png', '190621_++1_S01_frame120.png']

    series = "S01"
    prof_matrix_list = derive_prof_matrix_list(segmentation_folder, output_folder, series, segmented_filename_list)

    print(type(prof_matrix_list))
    print(len(prof_matrix_list))

    print(type(prof_matrix_list[0]))
    print(prof_matrix_list[0].shape)

    print(np.round(prof_matrix_list[0], 3))