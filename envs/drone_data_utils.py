import numpy as np 

import sys
sys.path.insert(0, '..')

from envs.gridworld_drone import GridWorldDrone
from featureExtractor.gridworld_featureExtractor import SocialNav,LocalGlobal,FrontBackSideSimple

import os

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


file_name = '/home/thalassa/akonar/Study/deepirl/envs/stanford_drone_subset/annotations/bookstore/video0/annotations.txt'
feature_extactor = LocalGlobal(window_size=3, grid_size=20, fieldList = ['agent_state','goal_state','obstacles'])


print(extract_subjects_from_file(file_name))
extract_trajectory(file_name,feature_extactor)