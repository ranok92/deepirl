import numpy as np 
import torch


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


def preprocess_data_from_stanford_drone_dataset(annotation_file):
    '''
    Final format:
    frame_number, ped_id, y_coord, x_coord
    '''
    file_n = annotation_file+'_processed'+'.txt'

    if not os.path.isfile(annotation_file):
        print("The file does not exist!")
        return 0

    with open(annotation_file) as f:

        for line in f:
            line_list = line.strip().split(' ')

            with open(file_n,'a') as fnew:
                if line_list[6]!=str(1):
                    fnew.write("%s "%line_list[5]) #the frame number
                    fnew.write("%s "%line_list[0]) #the ped id
                    y_coord = (int(line_list[2])+int(line_list[4]))/2
                    #pdb.set_trace()
                    x_coord = (int(line_list[1])+int(line_list[3]))/2
                    fnew.write("%d "%y_coord) #the y_coord
                    fnew.write("%d "%x_coord) #the x-coord
                    fnew.write("\n")


    return 0


def preprocess_data_from_control_points(annotation_file):
    '''
    given a annotation file containing spline control points converts that 
    to a frame-by-frame representation and writes that on a txt file with a easy to read format
    '''

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




def extract_trajectory(annotation_file, feature_extractor, folder_to_save):


    if not os.path.exists(folder_to_save):
        os.mkdir(folder_to_save)

    subject_list = extract_subjects_from_file(annotation_file)
    print(subject_list)
    for sub in subject_list:
        trajectory_info = []
        print('Starting for subject :',sub)
        world = GridWorldDrone(display=True, is_onehot=False, 
                        seed=0, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file=annotation_file,
                        subject=sub,                        
                        rows=1088, cols=1420,
                        width=20)

        world.reset()

        while world.current_frame < world.final_frame:
            state,_,_,_ = world.step()
            #pdb.set_trace()
            state = feature_extactor.extract_features(state)
            state = torch.tensor(state)
            trajectory_info.append(state)
        state_tensors = torch.stack(trajectory_info)
        torch.save(state_tensors, os.path.join(folder_to_save,'traj_of_sub_%s.states' % str(sub)))




def extract_subjects_from_file(annotation_file):

    sub_list = []
    if not os.path.isfile(annotation_file):
        print("The annotation file does not exist!!")
        return 0

    with open(annotation_file) as f:

        for line in f:
            line = line.strip().split(' ')
            sub_list.append(int(line[1]))

    return set(sub_list)


def record_trajectories(num_of_trajs,path):
    '''
    Let user play in an environment simulated from the data taken from a dataset
    '''
    step_size = 20
    agent_size = 20
    grid_size = 20
    obs_size = 20
    window_size = 7
    '''
    feature_extractor = LocalGlobal(window_size=window_size, agent_width=agent_size,
                                    step_size=step_size, 
                                    obs_width=obs_size,
                                    grid_size=grid_size, 
                                    fieldList=['agent_state', 'goal_state', 'obstacles'])
    '''

    feature_extractor = FrontBackSideSimple(thresh1=1, thresh2=2,
                                      thresh3=5, agent_width=agent_size,
                                      step_size=step_size, grid_size=grid_size,
                                      fieldList=['agent_state','goal_state','obstacles'])


    env = GridWorldDrone(display=True, is_onehot=False, agent_width=agent_size,
                         seed=10, obstacles=None, obs_width=obs_size,
                         step_size=step_size, width=grid_size,
                         show_trail=False,
                         is_random=True,
                         annotation_file='./stanford_drone_subset/annotations/bookstore/video0/annotations.txt',
                         subject=None,
                         rows=1088, cols=1424)
    i = 0
    while i < num_of_trajs:
        actions = []
        states = []
        state = env.reset()
        states = [feature_extractor.extract_features(state)]

        done = False
        run_reward = 0
        while not done:

            action = env.take_action_from_user()
            print(action)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            run_reward += reward 
            if not done:
                next_state = feature_extractor.extract_features(next_state)
                #for variants of localglobal
                #print(next_state[-window_size**2:].reshape(window_size,window_size))
                #print(next_state[0:9].reshape(3,3))
                #for fbs simple
                print(next_state[12:].reshape(3,4))
                states.append(next_state)

        if run_reward > 1:

            actions_tensor = torch.tensor(actions)
            states_tensor = torch.stack(states)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            torch.save(actions_tensor,
                       os.path.join(path, 'traj%s.acts' % str(i)))

            torch.save(states_tensor,
                       os.path.join(path, 'traj%s.states' % str(i)))

            i += 1
        else:

            print('Bad example. Discarding!')



file_name = './annotations.txt_processed.txt'
feature_extactor = LocalGlobal(window_size=7, grid_size=20, agent_width=20, obs_width=20,
                               fieldList = ['agent_state','goal_state','obstacles'])


print(extract_subjects_from_file(file_name))
extract_trajectory(file_name,feature_extactor,'temp')
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


preprocess_data_from_stanford_drone_dataset('annotations.txt')
'''