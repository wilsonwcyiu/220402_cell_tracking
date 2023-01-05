import enum
import itertools
from collections import defaultdict, namedtuple
from decimal import Decimal
from enum import Enum
from functools import cmp_to_key
from datetime import datetime

from PIL import Image, ImageDraw

import numpy as np
import sys

from sklearn.metrics import ConfusionMatrixDisplay

from cell_tracking_3d.feature_based_3d import CoordTuple
from main_a_viterbi.viterbi_adjust4d import CellId
from ml_classification.generate_tracks import generate_all_combination_fixed_track_length



def main():
    generate_all_possibilities()

def product_list_list():
    number_list: list = [1, 2, 4]
    result_combination_tuple_list: list = []

    frame_node_id_list_list : list = []
    for idx, total_node_in_frame in enumerate(number_list):
        node_id_list: list = [x for x in range(1, total_node_in_frame + 1)]
        frame_node_id_list_list.append(node_id_list)

    result_combination_tuple_list = list(itertools.product(*frame_node_id_list_list))


    print(frame_node_id_list_list)
    print()
    print(result_combination_tuple_list)






def method_52():
    actual_list =  [1, 2, 3, 4] * 100
    predict_list = [2, 2, 3, 4] * 100

    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(actual_list, predict_list)

    print(cf_matrix)

    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Zebra fish cell image classification\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=['Long\nPlus','Long\nMinus', 'Local\nPlus', 'Local\nMinus'])
    disp.plot(cmap="Blues", values_format='d')

    ax.xaxis.set_ticklabels(['Long Plus','Long Minus', 'Local Plus', 'Local Minus'])
    ax.yaxis.set_ticklabels(['Long Plus','Long Minus', 'Local Plus', 'Local Minus'])



    ## Display the visualization of the Confusion Matrix.
    plt.show()

    #
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                               ...                               display_labels=clf.classes_)
    # >>> disp.plot()



def method_51():
    abs_file_path = "d:/tmp.png"
    point_radius_pixel = 2
    coord_tuple_list = [CoordTuple_2d(4, 6), CoordTuple_2d(5, 16), CoordTuple_2d(12, 17), CoordTuple_2d(100, 100), CoordTuple_2d(512, 512) ]

    generate_track_image(coord_tuple_list, point_radius_pixel, abs_file_path)

def generate_track_image(coord_tuple_list: list, point_radius_pixel: float, abs_file_path: str, line_width = 1):
    img_dimension: tuple = (512, 512)
    background_rgb = (255, 255, 255)
    line_rgb = (0, 0, 0)


    im = Image.new('RGB', img_dimension, background_rgb)
    draw = ImageDraw.Draw(im)

    total_point: int = len(coord_tuple_list)
    second_last_point = total_point - 1
    for track_idx in range(0, second_last_point):
        current_coord = coord_tuple_list[track_idx]
        next_coord = coord_tuple_list[track_idx+1]

        draw.line((current_coord.x, current_coord.y, next_coord.x, next_coord.y), fill=line_rgb, width=line_width)

        left_top_coord = (current_coord.x - point_radius_pixel, current_coord.y - point_radius_pixel)
        right_bottom_coord = (current_coord.x + point_radius_pixel, current_coord.y + point_radius_pixel)
        draw.ellipse([left_top_coord, right_bottom_coord], fill=line_rgb)

        left_top_coord = (next_coord.x - point_radius_pixel, next_coord.y - point_radius_pixel)
        right_bottom_coord = (next_coord.x + point_radius_pixel, next_coord.y + point_radius_pixel)
        draw.ellipse([left_top_coord, right_bottom_coord], fill=line_rgb)

    im.save(abs_file_path)


def method_50():
    dimension = (512, 512)
    background_rgb = (255, 255, 255)
    line_rgb = (0, 0, 0)

    im = Image.new('RGB', dimension, background_rgb)
    draw = ImageDraw.Draw(im)

    x1, y1 = 100, 400
    x2, y2 = 300, 500
    draw.line((x1, y1, x2, y2), fill=line_rgb, width=4)


    left_top_coord = (10, 10)
    right_bottom_coord = (110, 110)
    draw.ellipse([left_top_coord, right_bottom_coord], fill=line_rgb)
    # im.show()
    im.save("d:/tmp.png")




def method_49():
    result_list_list = []

    result_list = [(0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0)]
    result_list_list.append(result_list)
    result_list = [(1, 1, 0), (1, 2, 0), (1, 3, 0), (1, 4, 0)]
    result_list_list.append(result_list)
    result_list = [(2, 1, 0), (2, 2, 0), (2, 3, 0), (2, 4, 0)]
    result_list_list.append(result_list)


    tmp_list = generate_all_combination_fixed_track_length(result_list_list, fixed_track_length_to_generate=3)

    for tmp in tmp_list:
        print(tmp)



def method_48():
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

     angle_between((1, 0, 0), (0, 1, 0))
     1.5707963267948966
     angle_between((1, 0, 0), (1, 0, 0))
     0.0
     angle_between((1, 0, 0), (-1, 0, 0))
     3.141592653589793
    """

    vector_coord_tuple_1 = CoordTuple(1, 0, 0)
    vector_coord_tuple_2 = CoordTuple(0, 1, 0)
    degree = derive_degree_diff_from_two_vectors(vector_coord_tuple_1, vector_coord_tuple_2)
    print(degree)

    vector_coord_tuple_1 = CoordTuple(1, 0, 0)
    vector_coord_tuple_2 = CoordTuple(1, 0, 0)
    degree = derive_degree_diff_from_two_vectors(vector_coord_tuple_1, vector_coord_tuple_2)
    print(degree)

    vector_coord_tuple_1 = CoordTuple(1, 0, 0)
    vector_coord_tuple_2 = CoordTuple(1, 1, 0)
    degree = derive_degree_diff_from_two_vectors(vector_coord_tuple_1, vector_coord_tuple_2)
    print(degree)



def method_47():
    print(int(round(1.1)))
    print(int(round(1.9)))



def method_46():
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    rad = 0.2
    edgy = 0.05

    # for c in np.array([[0,0], [0,1], [1,0], [1,1]]):
    for c in np.array([[0,0]]):
        a = get_random_points(n=7, scale=1) + c

        x,y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        x = (x - 0.5) * 20
        y = (y - 0.5) * 20
        x = x.astype(int)
        y = y.astype(int)

        plt.plot(x,y)

        print("c", c)
        print("a", a)
        print("x", x)
        print("y", y)

    print()


    plt.show()

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





def method_45():
    move_angle = 270
    move_distance = 10
    diff_x, diff_y = derive_x_y_diff(move_angle, move_distance)
    print(diff_x, diff_y)



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



def method_44():
    total_frame_num = 3
    gt_track_adj_list =      [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    predict_track_adj_list = [(1, 1)]

    gt_idmap_tuple_list = []
    gt_idmap_tuple_list.append((gt_track_adj_list, predict_track_adj_list))
    gt_idmap_tuple_list.append((gt_track_adj_list, predict_track_adj_list))
    gt_idmap_tuple_list.append((gt_track_adj_list, predict_track_adj_list))

    FF_list = get_FIT_and_FIO(total_frame_num, gt_idmap_tuple_list, gt_idmap_tuple_list)


    prediction_idmap_list = []
    prediction_idmap_list.append((predict_track_adj_list, gt_track_adj_list))
    prediction_idmap_list.append((predict_track_adj_list, gt_track_adj_list))
    prediction_idmap_list.append((predict_track_adj_list, gt_track_adj_list))

    return FF_list



def get_FIT_and_FIO(total_frame_num, gt_idmap_tuple_list, prediction_idmap_list):
    #find the FIT and FIO errors per frame
    FF_list = []
    for frame_num in range(total_frame_num):
        gt_in_frame = 0
        FIT = 0
        #Count the FITs in this frame
        for gt_track_list, prediction_list in gt_idmap_tuple_list:
            #get the gt cell number in this frame
            curr_gt_cell = [ i[0] for i in gt_track_list if i[1] == frame_num]  # frame
            # if a ground truth track is present in this frame,
            # check if prediction_list is the same, otherwise count FIT
            if curr_gt_cell:
                gt_in_frame += 1
                curr_est_cells = [ i[0] for i in prediction_list if i[1] == frame_num]
                if curr_est_cells:
                    for curr_est_cell in curr_est_cells:
                        if not int(curr_est_cell) == int(curr_gt_cell[0]):
                            FIT = FIT + 1

        FIO = 0
        #count the FIOs in this frame
        for prediction_list, gt_track_list in prediction_idmap_list:
            #get the prediction_list cell numbers in this frame
            curr_est_cells = [ i[0] for i in prediction_list if i[1] == frame_num]
            # if the ground truth track is present in this frame,
            # check if prediction_list is the same, otherwise count FIO
            curr_gt_cell = [ i[0] for i in gt_track_list if i[1] == frame_num]
            if curr_gt_cell:
                if curr_est_cells:
                    for curr_est_cell in curr_est_cells:
                        if not int(curr_est_cell) == int(curr_gt_cell[0]):
                            FIO = FIO + 1

        if gt_in_frame > 0:
            FF_list.append((FIT, FIO, gt_in_frame))

    return FF_list



def method_43():
    gt_track_adj_list = [(0, 0), (0, 1), (0, 2)]
    predict_track_adj_list = [(0, 0), (2, 1), (0, 2)]

    res = len(set(gt_track_adj_list) & set(predict_track_adj_list))
    print(res)


def method_42():
    # print(np.full((3, 5), []), dtype=list)
    a=np.empty(10); a.fill([])
    print(a)


def method_41():
    cord_list = [CoordTuple_2d(1, 1), CoordTuple_2d(2, 2), CoordTuple_2d(3, 2)]
    final_vector_coord_tuple = derive_vector_from_coord_list(cord_list)
    print(final_vector_coord_tuple)

    zero_degree_vec_tuple = CoordTuple_2d(0, 1)
    degree = derive_degree_diff_from_two_vectors(zero_degree_vec_tuple, final_vector_coord_tuple)
    print(degree)


CoordTuple_2d = namedtuple("CoordTuple", "x y")
def derive_vector_from_coord_list(coord_tuple_list: list):
    start_vec: CoordTuple_2d = coord_tuple_list[0]
    end_vec: CoordTuple_2d = coord_tuple_list[-1]

    final_coord_tuple = CoordTuple_2d(end_vec.x - start_vec.x, end_vec.y - start_vec.y)

    return final_coord_tuple


    # updated_coord_tuple = CoordTuple(0, 0)
    #
    # previous_vec = CoordTuple(0, 0)
    # for coord_tuple in coord_tuple_list:
    #     x_diff = coord_tuple.x - previous_vec.x
    #     y_diff = coord_tuple.y - previous_vec.y
    #
    #     new_x = updated_coord_tuple.x + x_diff
    #     new_y = updated_coord_tuple.y + y_diff
    #
    #     updated_coord_tuple = CoordTuple(new_x, new_y)
    #
    #     previous_vec = coord_tuple
    #
    # return updated_coord_tuple


def method_40():
    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(0, 1), CoordTuple_2d(1, 0));    print(degree)
    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(0, 1), CoordTuple_2d(0, -1));    print(degree)
    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(0, 1), CoordTuple_2d(-1, 0));    print(degree)
    print()

    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(1, 0), CoordTuple_2d(0, 1));    print(degree)
    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(1, 0), CoordTuple_2d(0, -1));    print(degree)
    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(1, 0), CoordTuple_2d(-1, 0));    print(degree)
    print()

    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(0, 1), CoordTuple_2d(1, 0));    print(degree)
    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(0, -1), CoordTuple_2d(1, 0));    print(degree)
    degree = derive_degree_diff_from_two_vectors(CoordTuple_2d(-1, 0), CoordTuple_2d(1, 0));    print(degree)


def method_39():
    degree = derive_degree_from_vector(CoordTuple_2d(-1, 0));    print(degree)



from math import atan2, degrees, radians

def derive_degree_diff_from_two_vectors(vector_coord_tuple_1: CoordTuple_2d, vector_coord_tuple_2: CoordTuple_2d): #These can also be four parameters instead of two arrays
    dot = vector_coord_tuple_1.x * vector_coord_tuple_2.x + vector_coord_tuple_1.y * vector_coord_tuple_2.y      # dot product
    det = vector_coord_tuple_1.y * vector_coord_tuple_2.x - vector_coord_tuple_1.x * vector_coord_tuple_2.y      # determinant
    angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    angle = degrees(angle)

    if angle < 0:
        angle = abs(angle)

    return angle


def derive_degree_from_vector(vector_coord_tuple_1: CoordTuple_2d): #These can also be four parameters instead of two arrays
    default_zero_degree_vector = CoordTuple_2d(0, 1)

    dot = default_zero_degree_vector.x * vector_coord_tuple_1.x + default_zero_degree_vector.y * vector_coord_tuple_1.y      # dot product
    det = default_zero_degree_vector.y * vector_coord_tuple_1.x - default_zero_degree_vector.x * vector_coord_tuple_1.y      # determinant
    angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    angle = degrees(angle)

    if angle < 0:
        angle += 360

    return angle



def method_38():
    print(datetime.now().strftime("%H%M%S"))

def test_retrieve_all_dependency_cell_set_recursive():
    cell_dependency_dict = {}
    cell_dependency_dict[CellId(1, 2)] = [CellId(1,3)]
    cell_dependency_dict[CellId(1, 3)] = [CellId(1,4), CellId(1,5)]

    handling_cell_id = CellId(1,4)

    dependency_set = set()
    retrieve_all_dependency_cell_set_recursive(cell_dependency_dict, handling_cell_id, dependency_set)

    for dependency in dependency_set:
        print(dependency)


def retrieve_all_dependency_cell_set_recursive(cell_dependency_dict: dict, handling_cell_id: CellId, dependency_set: set):
    if handling_cell_id in cell_dependency_dict:
        dependent_cell_id_list = cell_dependency_dict[handling_cell_id]
        for dependent_cell_id in dependent_cell_id_list:
            dependency_set.add(dependent_cell_id)
            retrieve_all_dependency_cell_set_recursive(cell_dependency_dict, dependent_cell_id, dependency_set)

    return




def test_find_most_independent_cell_id_recursive():
    cell_dependency_dict = {}
    cell_dependency_dict[CellId(1, 2)] = [CellId(1,3)]
    cell_dependency_dict[CellId(1, 3)] = [CellId(1,4), CellId(1,5)]

    handling_cell_id = CellId(1, 2)
    to_handle_cell_id_list = [CellId(1,2), CellId(1,3), CellId(1,4), CellId(1,5)]

    to_handle_cell_id = find_most_independent_cell_id_recursive(cell_dependency_dict, handling_cell_id, to_handle_cell_id_list)
    print(to_handle_cell_id)


def find_most_independent_cell_id_recursive(cell_dependency_dict: dict, handling_cell_id: CellId, to_handle_cell_id_list: list):
    if handling_cell_id in cell_dependency_dict:
        dependent_cell_id_list = cell_dependency_dict[handling_cell_id]
        for dependent_cell_id in dependent_cell_id_list:
            if dependent_cell_id in to_handle_cell_id_list:
                lower_dependent_cell_id = find_most_independent_cell_id(cell_dependency_dict, dependent_cell_id, to_handle_cell_id_list)

                return lower_dependent_cell_id

    return handling_cell_id


def method_37():

    from datetime import datetime

    now = datetime.now() # current date and time

    date_time = now.strftime("%Y%m%d-%H%M%S")
    print("date and time:",date_time)



def method_36():
    data_list = [CellId(1, 2), CellId(3, 4)]
    data_list.remove(CellId(1, 2))

    for data in data_list:
        print(data)




def method_35():
    data_list = [0,3,7,2]
    print(data_list[-1])



def method_34():
    data_list = [0,3,7,2]
    t = np.argmax(data_list)
    print(t)





def method_33():
    data_dict = {}
    data_dict[CellId(1, 2)] = 1
    data_dict[CellId(1, 3)] = 1
    data_dict[CellId(1, 4)] = 1

    del data_dict[CellId(1, 2)]

    for cell_id in data_dict.keys():
        print(cell_id.__str__())

    data_dict[CellId(1, 2)] = 1
    print()

    for cell_id in data_dict.keys():
        print(cell_id.__str__())

    cell_id_key_list = list(data_dict.keys())
    cell_id_key_list.sort(key=cmp_to_key(compare_cell_id))
    sorted_dict: dict = {}
    for sorted_key in cell_id_key_list:
        sorted_dict[sorted_key] = data_dict[sorted_key]
    data_dict = sorted_dict

    code_validate_track_key_order(data_dict)


def method_32():
    class STRATEGY_ENUM(enum.Enum):
        ALL_LAYER = 1
        ONE_LAYER = 2

    s_enum = STRATEGY_ENUM.ALL_LAYER

    if s_enum == STRATEGY_ENUM.ALL_LAYER:
        print(STRATEGY_ENUM.ALL_LAYER)
    elif s_enum == STRATEGY_ENUM.ONE_LAYER:
        print(STRATEGY_ENUM.ONE_LAYER)
    else:
        raise Exception(s_enum)



def newton_method(input_number: float, max_error: float=0.01):
    sqrRoot = float(input_number)

    while True:
        sqrRoot = 0.5 * (sqrRoot + input_number/sqrRoot)

        check = abs(input_number - sqrRoot*sqrRoot) < max_error
        if check:
            return sqrRoot


def SqrLoop(i1: float, max_err: float = 0.01):
    o1 = i1;
    v: float = i1

    while True:
        i2 = Approx(o1)
        v = Dup(i2)

        Check: bool = o1 - (i2 * i2) < max_err
        if Check:
            o2 = i2
            return o2
        else:
            o1 = v






def Approx(data):
    return None

def Dup(data):
    return None



def method_31():
    data_dict = {}
    data_dict[0] = [10,11,12]
    data_dict[1] = [20,21]
    data_dict[2] = [30,31]
    data_dict[3] = [40,41,42]
    data_dict[4] = [50,51,52]

    # _cut
    del data_dict[1]
    del data_dict[2]

    print("data_dict", data_dict)

    result_list = []
    for i in range(len(data_dict)):
        if i not in data_dict.keys():
            print("skip handle key: ", i)
            continue
        else:
            print("do handle key: ", i)
            min_track_length: int = 2
            if (len(data_dict[i]) > min_track_length):
                result_list.append(data_dict[i])

    print(result_list)





def method_30():
    data_dict = {}
    data_dict[10] = [10, 11, 12]
    data_dict[20] = [20, 21, 22]

    for key, value in enumerate(data_dict.keys()):
        print(key)
        print(value)

def method_29():
    # list defined by reference
    data_list = [[]] * 10
    data_list[0].append(1)

    print(data_list)


def method_28():
    cell_id_1 = CellId(1,2)
    cell_id_2 = CellId(2,3)

    tmp = [(cell_id_1)]
    tmp[0] += (cell_id_2,)
    print(tmp[0][1])


    # data_dict = {}
    # data_dict[cell_id_1] = [1,2,3]
    # data_dict[cell_id_2] = [4,5,6]
    #
    # print(data_dict.__str__())
    #
    # for cell_id, data in data_dict.items():
    #     print(cell_id, data)


def method_27():
    cell_id_1 = CellId(1,2)
    cell_id_2 = CellId(1,2)

    print(cell_id_1 == cell_id_2)


def method_26():
    cell_id_1 = CellId(1,2)
    cell_id_2 = CellId(1,3)
    cell_id_3 = CellId(2,2)
    cell_id_4 = CellId(2,3)
    cell_id_5 = CellId(3,1)



    # tmp = cell_id_1.__cmp__(cell_id_2)
    # print(tmp)
    #
    # tmp = cell_id_2.__cmp__(cell_id_1)
    # print(tmp)

    # print(cell_id_1 < cell_id_2)


    # cell_id_list: CellId = [cell_id_1, cell_id_2, cell_id_3, cell_id_4, cell_id_5]
    cell_id_list: CellId = [cell_id_5, cell_id_4, cell_id_3, cell_id_2, cell_id_1]


    cell_id_list.sort(key=cmp_to_key(compare_cell_id))

    print(len(cell_id_list))
    for cell_id in cell_id_list:
        print(cell_id)



def method_25():
    for i in range(119, 2, -1):
        print(i)



def method_24():
    data_dict = {}
    data_dict[1] = 2
    data_dict[2] = 2
    data_dict[5] = 2
    data_dict[3] = 2

    print(np.max(list(data_dict.keys())))



def method_23():
    data_dict = defaultdict(dict)
    data_dict[1][2] = [10, 11]
    data_dict[1][3] = [30, 31]

    print(data_dict)



def method_22():
    import decimal

    tmp_float: Decimal = decimal.Decimal(0.2)
    for i in range(0, 40):
        tmp_float = tmp_float * tmp_float
        print(i, ":", tmp_float)



def method_21():
    tmp_float: float = 0.0
    print(tmp_float <= 0)



def method_20():
    data_set = set()
    data_set.add(1)
    data_set.add(1)
    data_set.add(1)
    data_set.add(2)

    print(data_set)



def method_19():
    data_list = [(1,2,3)] * 3
    for data in data_list[0]:
        print(data)



def method_18():
    data_list = [()] * 3
    print(data_list[0] == None)
    print(len(data_list[0]))



def method_17():
    a = ('x', 'y')
    a += ('z',)
    print(a)



def method_16():
    data_list = [()] * 3
    data_list[1] += (2,)
    print(data_list)

    tmp_list = list(data_list[0])
    tmp_list.append(123)
    data_list[0] = tuple(tmp_list)

    # tmp_list = list(data_list[0])
    # tmp_list.append(456)
    # data_list[0] = tuple(tmp_list)
    print(data_list)

    print(len(data_list[0]))


def method_15():
    mtx_1 = np.array([ [0, 1],
                       [2, 3],
                       [4, 5]])
    print(mtx_1[0:2, 0])


def method_14():
    test_arr = np.array([[1, 2], [3, 4], [5, 6]])
    col_arr = test_arr[:, 0]
    print(type(col_arr))
    print(col_arr.shape)
    print(col_arr)


c = 1 # global variable
def add():
    # global c
    c = c + 2 # increment c by 2
    print(c)




def method_13():
    data_list = np.array([
        [1, 2, 3],
        [4, 5, 6]])

    print(data_list[0])
    print(data_list[0].shape)

    tmp_list = [1,2,3]
    tmp_arr = np.array(tmp_list)
    print(tmp_arr)
    print(tmp_arr.shape)



def method_12():
    data_list = np.array([
        [1, 2, 3],
        [4, 5, 6]])
    tmp = np.max(data_list, axis=1)
    print(tmp.shape)
    tmp = tmp.tolist()
    print(type(tmp))
    print(tmp)


def method_11():
    data_list = np.array([
                            [1, 2, 3],
                            [4, 5, 6]])

    b = obtain_matrix_value_by_index_list(data_list, [0, 1, 0])
    print(b)


def method_10():
    original_track_list = [0, 1, 2, 3, 4]

    print(original_track_list[-1])



def method_9():
    value_ab_vec = np.array([0, 0, 0, 1])
    if ( np.all(value_ab_vec == 0) ):
        print("np.all(value_ab_vec == 0); break")
    else:
        print("np.all(value_ab_vec != 0); break")



def method_8():
    a_dict = {}
    a_dict["1"] = 1
    a_dict["2"] = 2
    a_dict["3"] = 2

    print(len(a_dict))

def method_7():
    arr1 = np.array([1, 2, 3])
    arr1 = arr1.reshape(3,1)

    # arr1 = arr1[:, np.newaxis]
    print("np.newaxis", arr1)
    repeat_times = 3
    repeated_mtx = np.repeat(arr1, repeat_times, axis=1)

    print(repeated_mtx)



def method_6():
    arr1 = np.array([[1, 2, 3], [4, 5, 6], [4.1, 2.2, 2.3]])
    max_data = np.max(arr1, axis=0)
    print(max_data)


def method_5():
    arr1 = np.array([[1, 2, 3], [4, 5, 6], [4.1, 2.2, 2.3]])
    max_data = np.argmax(arr1, axis=0)
    print(max_data)



def method_4():
    value_list: list = [(1,2,3), (4,5,6)]
    list.reverse(value_list)
    print(value_list)



def method_3():
    dummy_dict = {}
    dummy_dict["a"] = [1, 2, 3]
    dummy_dict["b"] = [4, 5]

    for k, v in dummy_dict.items():
        print(k, v)


def method_2():
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])

    print(arr1 * arr2)



def method_1():
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


if __name__ == '__main__':
    main()
