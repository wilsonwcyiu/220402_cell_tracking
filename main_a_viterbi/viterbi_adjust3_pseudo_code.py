import numpy as np

from enum import Enum

class TrackStrategyEnum(Enum):
    TOTAL = 1; FRAME_TO_FRAME = 2


def main():
    # variable summary
    to_handle_cell_idx_list: list = []                                            # (A list that stores all cell_idx to be handled)
    cell_profit_mtx_list: list = None                                             # (list of matrices. list size is equal to "total_frame_num-1", each matrix stores ALL the cell connections probability from previous frame to next frame)
    frame_num_cell_slot_idx_occupation_tuple_list_dict: dict = {}                 # (stores the occupation information of ALL cell slots in ALL frames. e.g.:  frame:5, cell_slot_idx:3.  frame_num_cell_slot_idx_occupation_tuple_list_dict[5][3] = (tuple with cell_idx that occupied this slot)
    cell_idx_frame_num_cell_slot_idx_total_probability_list_list_dict: dict = {}  # (stores total probability of ALL cells in ALL frames based on cell_idx, frame_num, cell_slot_idx. e.g.: cell_idx:1, frame:5, cell_slot:3.  cell_idx_frame_num_cell_slot_total_probability_list_dict[1][5][3] = the total probability of this cell)
    merge_above_threshold: float = 0.8
    result_cell_idx_trajectory_dict: dict = {}                                    # (trajectory result of each cell_idx)



    print("start")
    cell_profit_mtx_list: list = deriveProfitMtxList()

    to_handle_cell_idx_list = [cell_idx for cell_idx in range(len(cell_profit_mtx_list))]


    while to_handle_cell_idx_list != []:
        handling_cell_idx: int = to_handle_cell_idx_list[0]

        best_trajectory_index_ab_list = find_best_track(handling_cell_idx, cell_profit_mtx_list,
                                                        frame_num_cell_slot_idx_occupation_tuple_list_dict,
                                                        cell_idx_frame_num_cell_slot_idx_total_probability_list_list_dict,
                                                        merge_above_threshold,
                                                        TrackStrategyEnum.TOTAL)

        del to_handle_cell_idx_list[handling_cell_idx]

        to_redo_track_list = derive_to_redo_track_list(best_trajectory_index_ab_list, frame_num_cell_slot_idx_occupation_tuple_list_dict, cell_idx_frame_num_cell_slot_idx_total_probability_list_list_dict)
        to_handle_cell_idx_list.extend(to_redo_track_list)
        to_handle_cell_idx_list.sort()

        result_cell_idx_trajectory_dict[handling_cell_idx] = best_trajectory_index_ab_list

    print("end")



def find_best_track(handling_cell_idx: int,
                    cell_profit_mtx_list: list,
                    frame_num_cell_idx_occupation_tuple_list_dict:dict,
                    frame_num_cell_idx_total_probability_list_dict: dict,
                    merge_above_threshold: float,
                    trackStrategyEnum: Enum = TrackStrategyEnum.TOTAL):

    last_layer_all_connection_probability_list: list(float) = cell_profit_mtx_list[0][handling_cell_idx]
    frame_num_all_cell_best_track_idx_list_dict: list = {}                  # frame_num_all_cell_best_track_idx_list_dict[frame_num][next_frame_cell_slot_idx]

    total_frame: int = len(cell_profit_mtx_list) + 1
    for frame_num in range(2, total_frame):

        cell_profit_mtx_idx: int = frame_num - 1
        total_cell_slot_in_next_frame: int = cell_profit_mtx_list[cell_profit_mtx_idx].shape[1]
        frame_num_all_cell_best_track_idx_list_dict[frame_num] = [None] * total_cell_slot_in_next_frame
        for next_frame_cell_slot_idx in range(total_cell_slot_in_next_frame):
            if trackStrategyEnum == TrackStrategyEnum.TOTAL:
                best_idx: int = derive_best_idx_by_total_probabily()
            elif trackStrategyEnum == TrackStrategyEnum.FRAME_TO_FRAME:
                best_idx: int = derive_best_idx_by_frame_to_frame_probabily()


            cell_idx_occupied_tuple: tuple = frame_num_cell_idx_occupation_tuple_list_dict[frame_num][best_idx]
            has_cell_occupation: bool = (cell_idx_occupied_tuple != None)


            if not has_cell_occupation:
                print("connect cell normally")

            else:
                for occupied_cell_idx in cell_idx_occupied_tuple:
                    if trackStrategyEnum == TrackStrategyEnum.TOTAL:
                        occupied_cell_probability: float = frame_num_cell_idx_total_probability_list_dict[next_frame_cell_slot_idx][occupied_cell_idx]
                        handling_cell_probability: float = last_layer_all_connection_probability_list[next_frame_cell_slot_idx]
                    elif trackStrategyEnum == TrackStrategyEnum.FRAME_TO_FRAME:
                        occupied_cell_probability: float = cell_profit_mtx_list[frame_num][occupied_cell_idx]
                        handling_cell_probability: float = cell_profit_mtx_list[frame_num][next_frame_cell_slot_idx]


                    if handling_cell_probability > merge_above_threshold and occupied_cell_probability > merge_above_threshold:
                        print("both cell merge together")
                        frame_num_all_cell_best_track_idx_list_dict[next_frame_cell_slot_idx] = best_idx

                    elif handling_cell_probability < merge_above_threshold and occupied_cell_probability > merge_above_threshold:
                        print("handling_cell_probability merge to other cell")

                    elif handling_cell_probability > merge_above_threshold and occupied_cell_probability < merge_above_threshold:
                        print("potentially redo trajectory for other_cell, to be handled when final track is determined")
                        print("thus, nothing is to be done in this stage")

                    elif handling_cell_probability > merge_above_threshold and occupied_cell_probability < merge_above_threshold:
                        print("??? have to define what to do")


    best_index_ab_list: list = derive_best_track(last_layer_all_connection_probability_list, frame_num_all_cell_best_track_idx_list_dict)

    return best_index_ab_list



def deriveProfitMtxList():
    return None


def derive_best_idx_by_frame_to_frame_probabily():
    return None

def derive_best_idx_by_total_probabily():
    return None

def derive_to_redo_track_list(best_trajectory_index_ab_list: list, frame_num_cell_idx_occupation_tuple_list_dict, frame_num_cell_idx_total_probability_list_dict):
    return None

def derive_best_track():
    return None



if __name__ == '__main__':
    main()