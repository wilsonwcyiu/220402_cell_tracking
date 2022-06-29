import random
import time
from collections import namedtuple

import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

def main():
    total_cell: int = 10
    total_frame: int = 30

    min_move_distance: int = 0
    max_move_distance: int = 35

    max_move_angle: int = 180

    image_width: int = 512
    image_height: int = 512


    start_time = time.perf_counter()


    folder_path: str = 'D:/viterbi linkage/dataset/'
    synthesis_cell_dir_name: str = 'synthesis_cell'

    all_cell_track_list: list = []
    for cell_idx in range(total_cell):
        max_x = image_width; max_y = image_height
        random_start_coord_tuple = generate_random_start_coord_tuple(max_x, max_y)
        # print(random_start_coord_tuple)

        frame_num: int = 1
        cell_track_list = [(random_start_coord_tuple, frame_num)]
        previous_coord = random_start_coord_tuple
        for frame_num in range(2, total_frame+1):
            move_angle: float = int(random.randrange(-max_move_angle, max_move_angle))
            move_distance: int = int(random.randrange(min_move_distance, max_move_distance))
            diff_x, diff_y = derive_x_y_diff(move_angle, move_distance)

            next_coord_tuple = CoordTuple(previous_coord.x + diff_x, previous_coord.y + diff_y)

            cell_track_list.append((next_coord_tuple, frame_num))

            previous_coord = next_coord_tuple

        all_cell_track_list.append(cell_track_list)


    plt.figure(figsize = (20, 20))
    plt.figure(frameon=False)
    plt.rcParams["figure.autolayout"] = False

    for frame_num in range(1, total_frame+1):
        print(f"{frame_num}, ", end='')
        image_array = np.zeros( (image_width, image_height, 1), dtype=np.uint8)

        for cell_track_list in all_cell_track_list:
            cell_track_tuple = cell_track_list[frame_num-1]
            coord_tuple = cell_track_tuple[0]

            cell_radius = 3
            for x_idx in range(coord_tuple.x - cell_radius, coord_tuple.x + cell_radius + 1):
                for y_idx in range(coord_tuple.y - cell_radius, coord_tuple.y + cell_radius + 1):
                    if x_idx >= image_width or y_idx >= image_height:
                        continue

                    image_array[x_idx][y_idx] = 1



        plt.gray()
        plt.imshow(image_array, interpolation='nearest')
        image_full_path = folder_path + synthesis_cell_dir_name + "/" + str(frame_num)
        plt.savefig(image_full_path)

        # plt.show()

    print()


    execution_seconds = time.perf_counter() - start_time
    print(f"Execution time: {np.round(execution_seconds, 4)} seconds")



CoordTuple = namedtuple("CoordTuple", "x y")


from math import sin, cos, radians, pi
def derive_x_y_diff(move_angle: float, move_distance: float):
    is_x_negative: bool = (move_angle < 0)
    is_y_negative: bool = (move_angle > 90)

    if move_angle < 0:        move_angle = abs(move_angle)

    if move_angle > 90:       move_angle = (move_angle - 90)
    else:                     move_angle = (90 - move_angle)

    degree_rad = pi/2 - radians(move_angle)

    diff_x = move_distance * sin(degree_rad)
    diff_y = move_distance * cos(degree_rad)

    if is_x_negative: diff_x = -diff_x
    if is_y_negative: diff_y = -diff_y

    return int(diff_x), int(diff_y)



def generate_random_start_coord_tuple(max_x, max_y, min_x=0, min_y=0):
    x = int(random.randrange(min_x, max_x))
    y = int(random.randrange(min_y, max_y))

    return CoordTuple(x, y)


if __name__ == '__main__':
    main()