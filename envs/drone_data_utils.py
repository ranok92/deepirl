import numpy as np 

import sys
sys.path.insert(0, '..')

from envs.gridworld_drone import GridWorldDrone
from featureExtractor.gridworld_featureExtractor import SocialNav,LocalGlobal,FrontBackSideSimple

from scipy.interpolate import splev, splprep

import os
import pdb


'''
information regarding datasets and annotations:
ZARA DATASET:
    Video size : 720 × 576
    fps = 25
    Annotations are done considering the enter of the frame as origin.


UNIVERSITY STUDENTS:
    
    Video size : 720 × 576
    fps = 25
    Annotations are done considering the enter of the frame as origin.


'''

def get_dense_trajectory_from_control_points(list_of_ctrl_pts):
    #given the control points returns a dense array of all the in between points
    #the control points should be of the format 
    #frame number, ped_id, y_coord, x_coord
    x = []
    y = []
    t = []
    interpolate_fps = 1
    ped_id = None
    min_frame = 9999999
    max_frame = -1
    dense_val = []
    for point in list_of_ctrl_pts:

        #point_list = point.split(',')
        point_list = point
        x.append(float(point_list[3]))
        y.append(float(point_list[2]))
        cur_frame = int(point_list[0])
        if cur_frame < min_frame:
            min_frame = cur_frame
        if cur_frame > max_frame:
            max_frame = cur_frame
        t.append(int(point_list[0]))
        ped_id = point_list[1]

    print(x)
    print(y)
    tck, u = splprep([x,y], u=t)
    interpol_vals = splev(np.arange(min_frame, max_frame, interpolate_fps), tck)

    t_dense = np.arange(min_frame, max_frame, interpolate_fps)
    x_dense = interpol_vals[0]
    y_dense = interpol_vals[1]

    for i in range(t_dense.shape[0]):

        dense_val.append([t_dense[i], ped_id, y_dense[i], x_dense[i]])

    return dense_val



def read_data_from_file(annotation_file):
    '''

    given a file_name read the f and convert that into a list.
    
    The current format :
    x, y, frame_id, gaze_direction 
    
    Final format:
    frame_number, ped_id, y_coord, x_coord
    
    '''
    frame_x = 720
    frame_y = 576  
    diff_x = frame_x/2
    diff_y = frame_y/2  
    ped_id = -1
    processed_dict = {}
    if not os.path.isfile(annotation_file):
        print("The file does not exist!")
        return 0

    with open(annotation_file) as f:

        for line in f:
            prob_point_info = line.split(' - ')[0]
            if len(prob_point_info.split(' '))==4:
                point_info = prob_point_info.split(' ')
                processed_dict[ped_id].append([point_info[2], ped_id, float(point_info[1])+diff_y, float(point_info[0])+diff_x])
            else:
                ped_id += 1
                processed_dict[ped_id] = []

    return processed_dict



def preprocess_data(annotation_file):


    file_n = annotation_file+'_processed'+'.txt'

    extracted_dict = read_data_from_file(annotation_file)
    dense_info_list = []
    for ped in extracted_dict.keys():
            
        #for i in range(len(extracted_dict[ped])):
        #    print(extracted_dict[ped][i])

        dense_info = get_dense_trajectory_from_control_points(extracted_dict[ped])

        #for i in range(len(dense_info)):
        #    print(dense_info[i])
        with open(file_n, 'a') as f:

            for info in dense_info:
                dense_info_list.append(info)
                for val in info:
                    f.write("%s " %val)
                f.write("\n")

    return dense_info_list




def extract_trajectory(annotation_file, feature_extractor):

    subject_list = extract_subjects_from_file(annotation_file)
    trajectory_dict = []
    for sub in subject_list:
        print('Starting for subject :',sub)
        world = GridWorldDrone(display=False, is_onehot = False, 
                        seed = 0, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file= annotation_file,
                        subject=sub,                        
                        rows=1088, cols=1424, width=20)

        world.reset()

        while world.current_frame < world.final_frame:
            state,_,_,_ = world.step()

            state = feature_extactor.extract_features(state)

def extract_subjects_from_file(annotation_file):

    sub_list = []
    if not os.path.isfile(annotation_file):
        print("The annotation file does not exist.")
        return 0

    with open(annotation_file) as f:

        for line in f:
            line = line.strip().split(' ')
            sub_list.append(int(line[0]))

    return set(sub_list)

'''
file_name = '/home/thalassa/akonar/Study/deepirl/envs/stanford_drone_subset/annotations/bookstore/video0/annotations.txt'
feature_extactor = LocalGlobal(window_size=3, grid_size=20, fieldList = ['agent_state','goal_state','obstacles'])


print(extract_subjects_from_file(file_name))
extract_trajectory(file_name,feature_extactor)
'''
data = [
        [279.000000, -123.000000, 0, 87.397438], 
        [218.000000, -123.000000, 25, 90.000000], 
        [148.000000, -118.000000, 55, 84.382416],
        [82.000000 ,-135.000000, 86, 106.460014], 
        [15.000000, -151.000000, 113, 91.145760], 
        [-61.000000, -157.000000, 143, 90.000000], 
        [-139.000000, -160.000000, 173, 87.273689], 
        [-224.000000, -175.000000, 209, 93.814072], 
        [-346.000000, -190.000000, 265, 88.602821]] 

intval = preprocess_data('./data_zara/crowds_zara01.vsp')