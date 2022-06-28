import time

import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

def main():
    minimum_cell: int = 1
    maximum_cell: int = 10

    min_move_distance: int = 0
    max_move_distance: int = 35

    min_move_angle: int = 0
    max_move_angle: int = 180





    start_time = time.perf_counter()

    total_cell: int = maximum_cell

    # data = np.zeros( (512, 512, 1))
    # data[256,256] = 255

    # plt.imshow(data, interpolation='nearest')
    # plt.gray()
    # plt.imshow(data)
    # plt.show()

    # plt.rcParams["figure.figsize"] = [5, 5]
    # plt.rcParams["figure.autolayout"] = True

    plt.figure(figsize = (20, 20))

    arr = np.zeros( (200, 200, 1), dtype=np.uint8)
    for i in range(0, 80):
        arr[40, i] = 1
        arr[i, i] = 1
    # arr = np.random.rand(5, 5)
    # for i in range(len(arr[0])):
    #     for j in range(len(arr[i])):
    #         print(arr[i][j][0], end='')
    #     print()

    # print(arr)
    plt.gray()
    plt.imshow(arr, interpolation='nearest')

    plt.show()



    execution_seconds = time.perf_counter() - start_time
    print(f"Execution time: {np.round(execution_seconds, 4)} seconds")



if __name__ == '__main__':
    main()