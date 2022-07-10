import random
import time
from collections import namedtuple

import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

def main():
    total_cell: int = 10
    total_frame: int = 5

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

            if (next_coord_tuple, frame_num) not in cell_track_list:
                cell_track_list.append((next_coord_tuple, frame_num))

            previous_coord = next_coord_tuple

        all_cell_track_list.append(cell_track_list)


    plt.figure(figsize = (20, 20))
    plt.figure(frameon=False)
    plt.rcParams["figure.autolayout"] = False

    cell_radius = 1
    edgy = 0.9
    for frame_num in range(1, total_frame+1):
        print(f"{frame_num}, ", end='')
        image_array = np.zeros( (image_width, image_height, 1), dtype=np.uint8)

        for cell_track_list in all_cell_track_list:
            cell_track_tuple = cell_track_list[frame_num-1]
            coord_tuple = cell_track_tuple[0]


            a = get_random_points(n=7, scale=1)

            x_arr, y_arr, _ = get_bezier_curve(a, rad=cell_radius, edgy=edgy)
            x_arr = np.round((x_arr - 0.5) * 20)
            y_arr = np.round((y_arr - 0.5) * 20)
            x_arr = x_arr.astype(int)
            y_arr = y_arr.astype(int)

            x_y_coord_list = []
            for idx in range(len(x_arr)):
                x_y_coord_list.append(CoordTuple(x_arr[idx], y_arr[idx]))

            for x_y_coord in x_y_coord_list:
                x = coord_tuple.x + x_y_coord.x
                y = coord_tuple.y + x_y_coord.y

                if x >= image_width or y >= image_height:
                    continue

                image_array[x][y] = 1

            # cell_radius = 3
            # for x_idx in range(coord_tuple.x - cell_radius, coord_tuple.x + cell_radius + 1):
            #     for y_idx in range(coord_tuple.y - cell_radius, coord_tuple.y + cell_radius + 1):
            #         if x_idx >= image_width or y_idx >= image_height:
            #             continue
            #
            #         image_array[x_idx][y_idx] = 1



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

    return int(round(diff_x)), int(round(diff_y))



def generate_random_start_coord_tuple(max_x, max_y, min_x=0, min_y=0):
    x = int(random.randrange(min_x, max_x))
    y = int(random.randrange(min_y, max_y))

    return CoordTuple(x, y)






import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt


bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                          self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                          self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)






if __name__ == '__main__':
    main()