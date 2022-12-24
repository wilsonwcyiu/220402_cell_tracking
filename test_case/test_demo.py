from os import listdir

from main.viterbi_adjust3 import find_existing_series_list

if __name__ == '__main__':
    folder_path: str = 'D:/viterbi linkage/dataset/'
    output_folder = folder_path + 'output_unet_seg_finetune//'

    # test case 1
    input_series_list = ['S01']
    series_list = find_existing_series_list(input_series_list, listdir(output_folder))
    # print(series_list)
    if series_list == ['S01']:
        print("pass")
    else:
        print("failed")


    # test case 2
    input_series_list = ['S30']
    series_list = find_existing_series_list(input_series_list, listdir(output_folder))

    if series_list == []:
        print("pass")
    else:
        print("failed")


    # test case 3
    input_series_list = ['S02', 'S34']
    series_list = find_existing_series_list(input_series_list, listdir(output_folder))

    if series_list == ['S02']:
        print("pass")
    else:
        print("failed")


    # test case 4
    input_series_list = []
    series_list = find_existing_series_list(input_series_list, listdir(output_folder))

    if series_list == []:
        print("pass")
    else:
        print("failed")



    # test case 5
    input_series_list = None
    series_list = find_existing_series_list(input_series_list, listdir(output_folder))

    if series_list == []:
        print("pass")
    else:
        print("failed")


    # test case 6
    input_series_list = None
    output_folder = None
    series_list = find_existing_series_list(input_series_list, output_folder)

    if series_list == []:
        print("pass")
    else:
        print("failed")