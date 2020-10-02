import numpy as np 
import torch
import glob
import math
import sys
sys.path.insert(0, '..')

from envs.gridworld_drone import GridWorldDrone
from featureExtractor.gridworld_featureExtractor import SocialNav,LocalGlobal,FrontBackSideSimple


from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureRisk, DroneFeatureRisk_v2
from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speed, DroneFeatureRisk_speedv2
from featureExtractor.drone_feature_extractor import VasquezF1, VasquezF2, VasquezF3
from featureExtractor.drone_feature_extractor import Fahad, GoalConditionedFahad
from featureExtractor.drone_feature_extractor import total_angle_between, dist_2d

from scipy.interpolate import splev, splprep

from envs.drone_env_utils import angle_between


from matplotlib import pyplot as plt
import os
import pdb
import pathlib
import copy
import json
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dense_trajectory_from_control_points(list_of_ctrl_pts, frames=1, world_size = [576, 720]):
    '''
    input:
        list_of_ctrl_pts : path to the file containing the spline control points
        frames : int containing the number of frame intervals to maintain
        world_size : the size of the world on which to create the annotation file
                     [rows, cols]
    #given the control points returns a dense array of all the in between points
    #the control points should be of the format 
    #
    #frame number, ped_id, y_coord, x_coord
    '''
    x = []
    y = []
    t = []
    k = 3
    interpolate_fps = frames
    ped_id = None
    min_frame = 9999999
    max_frame = -1
    dense_val = []

    if len(list_of_ctrl_pts) > 0:
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

        #print('x :',x)
        #print('y :',y)
        #print('t :',t)

        if len(x) > k: #condition needed to be satisfied for proper interpolation
            tck, u = splprep([x,y], u=t, k=k)
            interpol_vals = splev(np.arange(min_frame, max_frame, interpolate_fps), tck)

            t_dense = np.arange(min_frame, max_frame, interpolate_fps)
            x_dense = interpol_vals[0]
            y_dense = interpol_vals[1]

            for i in range(t_dense.shape[0]):

                dense_val.append([t_dense[i], ped_id, y_dense[i], x_dense[i]])
        else:
            print('Discarding set of control points due to lack of control points.')
            return None

    return dense_val



def read_data_from_file(annotation_file, speed_multiplier=1,
                        rows=576, cols=720):
    '''

    given a file_name read the f and convert that into a list.
    
    The current format :
    x, y, frame_id, gaze_direction 
    
    Final format:
    frame_number, ped_id, y_coord, x_coord
    
    '''
    frame_x = cols
    frame_y = rows  
    diff_x = frame_x/2
    diff_y = frame_y/2  
    ped_id = -1
    processed_dict = {}
    if not os.path.isfile(annotation_file):
        print("The file does not exist!")
        exit()
        return 0

    with open(annotation_file) as f:

        for line in f:
            prob_point_info = line.split(' - ')[0]
            if len(prob_point_info.split(' '))==4:
                point_info = prob_point_info.split(' ')
                processed_dict[ped_id].append([int(int(point_info[2])/speed_multiplier), ped_id, frame_y - (float(point_info[1])+diff_y), float(point_info[0])+diff_x])
            else:
                ped_id += 1
                processed_dict[ped_id] = []


    print('Reading complete')
    #pdb.set_trace()
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

            with open(file_n, 'a') as fnew:
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


def preprocess_data_from_control_points(annotation_file, speed_multiplier=1,
                                        frame=1, world_size=[720, 576]):
    '''
    given a annotation file containing spline control points converts that 
    to a frame-by-frame representation and writes that on a txt file with a easy to read format
    input:
        annotation_file : the path to the annotation file
        frame : int, denoting the number of frames to skip between each entries of the 
                final frame by frame representation

        world_size : the size of the world [cols, rows]
    '''

    file_name = annotation_file.strip().split('/')[-1].split('.')[0]
    print(file_name)
    dir_name = ''
    for folder in annotation_file.strip().split('/')[0:-1]:
        dir_name += folder + '/'


    print(dir_name) 
    file_n = dir_name + file_name+'_per_frame'+'.txt'

    extracted_dict = read_data_from_file(annotation_file, 
                                         speed_multiplier=speed_multiplier,
                                        rows=world_size[1], cols=world_size[0])
    dense_info_list = []

    #add the world size to the trajectory file
    with open(file_n, 'a') as f:
        for val in world_size:
            f.write("%s " % val)
        f.write("\n")

    for ped in extracted_dict.keys():
            
        #for i in range(len(extracted_dict[ped])):
        #    print(extracted_dict[ped][i])
        dense_info = get_dense_trajectory_from_control_points(extracted_dict[ped], frame, world_size)

        #for i in range(len(dense_info)):
        #    print(dense_info[i])


        with open(file_n, 'a') as f:
            if dense_info is not None:
                for info in dense_info:
                    dense_info_list.append(info)
                    for val in info:
                        f.write("%s " %val)
                    f.write("\n")

    return dense_info_list



def extract_expert_speed_orientation(current_state):
    '''
    get the angle between the goal and the current orientation and the speed
    '''
    ref_vector = np.array([-1, 0])
    agent_orientation = current_state['agent_state']['orientation']
    goal_to_agent_vec = current_state['goal_state'] - current_state['agent_state']['position']
    
    signed_angle_between = (np.arctan2(agent_orientation[0],
                                      agent_orientation[1]) -
                            np.arctan2(goal_to_agent_vec[0],
                                    goal_to_agent_vec[1]))*180/np.pi 

    if signed_angle_between > 180:
        signed_angle_between = signed_angle_between - 360
    elif signed_angle_between < -180:
        signed_angle_between = 360 + signed_angle_between

    if math.isnan(signed_angle_between):
        signed_angle_between = 0

    speed = current_state['agent_state']['speed']

    return np.asarray([signed_angle_between, speed])


def extract_expert_action(next_state, current_state, 
                          orientation_div_size,
                          orientation_array_len,
                          speed_div_size,
                          speed_array_len):
    """
    Given the previous and the current state dictionary of the agent/expert,
    this function calculates the action taken by the expert( in accordance with
    the action space of the agent in the environment.)
    input:
        next_state    : A state dictionary containing the information of the
                        next state.
        current_state : A state dictionary containing the information of the
                        current state.
        orientation_array_len : Length of the orientation array of the environ-
                        ment denoting the number of action choices available for
                        the change in orientation.
        speed_array_len : Length of the speed array of the environment denoting
                          the number of action choices available for the 
                          change in speed.

    output:
        action : a single integer corresponding to the action that best matches
                 with the action that would have caused the agent to move
                from state prev to state current.
    """

    ref_vector = np.array([-1, 0])
    current_orientation = current_state['agent_state']['orientation']
    next_orientation = next_state['agent_state']['orientation']

    signed_angle_between = total_angle_between(next_orientation,
                                                 current_orientation)*180/np.pi

    if signed_angle_between > 0:
        angle_btwn_div = int(signed_angle_between/orientation_div_size)
    else:
        angle_btwn_div = math.ceil(signed_angle_between/orientation_div_size)



    orient_action = min(max(angle_btwn_div, 
                        -int(orientation_array_len/2)),
                        int(orientation_array_len/2)) + int(orientation_array_len/2)
    '''
    print('Current orientation :', current_orientation)
    print('Next orientation :', next_orientation)
    print('Angle between :', signed_angle_between)
    '''
    #pdb.set_trace()
    
    change_in_speed = next_state['agent_state']['speed'] - \
                      current_state['agent_state']['speed']

    change_in_speed_div = int(change_in_speed/speed_div_size)

    speed_action = min(max(int(change_in_speed/speed_div_size), 
                        -int(speed_array_len/2)),
                        int(speed_array_len/2)) + int(speed_array_len/2)

    '''
    print('Change in orientation  :{}, change in speed : {}'.format(angle_btwn,
                                                                        change_in_speed))

    print('Change in orientation division  :{}, change in speed division: {}'.format(angle_btwn_div,
                                                                        change_in_speed_div))

    print('Orientation action  :{}, speed : {}'.format(orient_action,
                                                      speed_action))
    '''

    return (speed_action*orientation_array_len)+orient_action



def extract_trajectory(annotation_file, 
                       folder_to_save, 
                       feature_extractor=None, 
                       display=False, 
                       extract_action=False,
                       show_states=False, subject=None, 
                       trajectory_length_limit=None):


    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    lag_val = 8
    
    tick_speed = 60
    subject_list = extract_subjects_from_file(annotation_file)
    print(subject_list)
    disp = display
    total_path_len = 0

    if show_states:
            tick_speed = 5
            disp = True
            
    #initialize world
    world = GridWorldDrone(display=disp, is_onehot=False, 
                        seed=10, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        show_orientation=True,
                        annotation_file=annotation_file,
                        subject=None,
                        external_control=False,
                        replace_subject=True,      
                        tick_speed=tick_speed,                  
                        rows=576, cols=720,
                        width=10)


    default_action = int(len(world.speed_array)/2)*int(len(world.orientation_array))+int(len(world.orientation_array)/2)
    
    
    default_action = torch.tensor(default_action)

    if subject is not None:
        subject_list = subject
    for sub in subject_list:
        print('Starting for subject :', sub)
        trajectory_info = []

        if extract_action:
            action_info = []
        step_counter_segment = 0

        segment_counter = 1
        world.subject = sub
        old_state = world.reset()
        cur_lag = 0
        print('Path lenghth :',world.final_frame - world.current_frame)
        path_len = world.final_frame - world.current_frame
        cur_subject_final_frame = world.final_frame
        total_path_len += world.final_frame - world.current_frame
        print('Total trajectory information :\nStarting frame: {},final frame: {}'.format(world.current_frame, cur_subject_final_frame))
        print('Total path length :', path_len)                                                                               
        if trajectory_length_limit is not None:

            traj_seg_length = min(trajectory_length_limit, path_len)
            #change the goal position
            world.goal_state = copy.deepcopy(world.return_position(world.cur_ped, world.current_frame + traj_seg_length)['position'])        
            world.state['goal_state'] = copy.deepcopy(world.goal_state) 
    
        print('Segment 1: Start frame :', world.current_frame)    
        while world.current_frame < cur_subject_final_frame:
            state, _, _, _ = world.step()
            step_counter_segment += 1
            #step_counter_trajectory += 1 
            #if disp:
            #    feature_extractor.overlay_bins(state)

            if extract_action:
                
                if cur_lag == lag_val:
                    
                    action = extract_expert_action(state, old_state, 
                                            world.orient_quantization,
                                            len(world.orientation_array),
                                            world.speed_quantization,
                                                len(world.speed_array))
                    '''
                    action = extract_expert_speed_orientation(state)
                    '''
                    old_state = copy.deepcopy(state)
                    action = torch.tensor(action)
                    action_info.append(action)
                    for i in range(cur_lag):
                        action_info.append(default_action)
                    cur_lag = 0
                    #pdb.set_trace()

                else:
                    cur_lag += 1
            if feature_extractor is not None:
                state = feature_extractor.extract_features(state)
                state = torch.tensor(state)
            trajectory_info.append(copy.deepcopy(state))
            if trajectory_length_limit is not None:

                if step_counter_segment%traj_seg_length == 0:
                    print('Segment {} final frame : {}'.format(segment_counter, world.current_frame))
                    path_len = cur_subject_final_frame - world.current_frame
                    traj_seg_length = min(trajectory_length_limit, path_len)
                    print('Length of next path :', traj_seg_length)

                    #change the goal position
                    world.goal_state = copy.deepcopy(world.return_position(world.cur_ped, world.current_frame + traj_seg_length)['position'])        
                    world.state['goal_state'] = copy.deepcopy(world.goal_state) 
                    print('Trajectory length : ', len(trajectory_info))

                    if feature_extractor is not None:
                        state_tensors = torch.stack(trajectory_info)
                        torch.save(state_tensors, 
                                os.path.join(folder_to_save, 
                                        'traj_of_sub_{}_segment{}.states'.format(str(sub), 
                                        str(segment_counter))))
                    else:
                        with open('traj_of_sub_{}_segment{}.states'.format(str(sub), 
                                  str(segment_counter)), 'w') as fout:
                            json.dump(trajectory_info, fout)
                    if extract_action:

                        acton_tensors = torch.stack(action_info)
                        torch.save(action_tensors,
                                os.path.join(folder_to_save, 
                                        'action_of_sub_{}_segment{}.actions'.format(str(sub),
                                        str(segment_counter))))
                    segment_counter += 1 
                    #pdb.set_trace()
                    step_counter_segment = 0 
                    trajectory_info = []
                    print('Segment {}: Start frame : {}'.format(segment_counter, 
                                                                world.current_frame))    

        #add the last bunch of actions

        for i in range(cur_lag):
            action_info.append(default_action)

        if trajectory_length_limit is None:

            if feature_extractor is not None:
                state_tensors = torch.stack(trajectory_info)
                torch.save(state_tensors, os.path.join(folder_to_save, 'traj_of_sub_{}_segment{}.states'.format(str(sub), str(segment_counter))))
            
                if extract_action:
                    #pdb.set_trace()
                    action_tensors = torch.stack(action_info)
                    torch.save(action_tensors,
                            os.path.join(folder_to_save, 
                                    'action_of_sub_{}_segment{}.actions'.format(str(sub),
                                    str(segment_counter))))
            else:
                '''
                with open('traj_of_sub_{}_segment{}.states'.format(str(sub), 
                            str(segment_counter)), 'w') as fout:
                    pdb.set_trace()
                    json.dump(trajectory_info, fout)
                '''
                np.save(os.path.join(folder_to_save, 'traj_of_sub_{}_segment{}.states'.format(str(sub), 
                            str(segment_counter))), trajectory_info)
                
                if extract_action:

                    action_tensors = torch.stack(action_info)
                    torch.save(action_tensors,
                            os.path.join(folder_to_save, 
                                    'action_of_sub_{}_segment{}.actions'.format(str(sub),
                                    str(segment_counter))))
        
    #if feature_extractor.debug_mode:
    #    feature_extractor.print_info()


    print('The average path length :', total_path_len/len(subject_list))




def extract_subjects_from_file(annotation_file):

    sub_list = []
    print('The annotation file :', annotation_file)
    if not os.path.isfile(annotation_file):
        print("The annotation file does not exist!!")
        return []

    with open(annotation_file) as f:

        for line in f:
            line = line.strip().split(' ')
            if len(line)==4:
                sub_list.append(int(line[1]))

    return set(sub_list)


def record_trajectories(num_of_trajs, env, feature_extractor, path, subject_list=None):
    '''
    Let user play in an environment simulated from the data taken from a dataset
    '''

    i = 0
    avg_len = 0
    while i < num_of_trajs:
        actions = []
        states = []
        if subject_list is None:
            state = env.reset()
        else:
            cur_ped_index = np.random.randint(len(subject_list))
            cur_ped = subject_list[cur_ped_index]
            env.subject = cur_ped
            state = env.reset_and_replace()
        states = [torch.from_numpy(feature_extractor.extract_features(state))]

        done = False
        run_reward = 0
        record_flag = False
        while not done:

            action = env.take_action_from_user()
            if action != None:
                record_flag = True
                print('Recording')
            else:
                record_flag = False

            if record_flag:
                actions.append(action)
            next_state, reward, done, _ = env.step(action)
            run_reward += reward 
            '''
            if reward != 0:
                print('current_reward :', reward)
                print('Run reward :', run_reward)
            '''
            next_state = feature_extractor.extract_features(next_state)
            #print('state dim :', next_state.shape)
            #for variants of localglobal
            #print(next_state[-window_size**2:].reshape(window_size,window_size))
            
            #for fbs simple
            #print(next_state[12:].reshape(3,4))
            if record_flag:
                '''
                print(done)
                print(next_state[0:9].reshape(3,3))
                print(next_state[9:18].reshape(3,3))
                print(next_state[18:])
                '''
                #print('Action taken :', action)
                states.append(torch.from_numpy(next_state))
                #print('not recording')

        print('Run reward :',run_reward)


        if run_reward >= .90:

            avg_len += len(states)
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

    print('Avg length :', avg_len/num_of_trajs)


def get_expert_trajectory_info(expert_trajectory_folder):
    '''
    given the expert trajectory folder, this fuctions reads the folder and 
    publishes the following information
    1. The dimension of the states present in the trajectory.
    2. The length of the trajectories.
    3. The avg length of the trajectories.
    '''
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trajectories = glob.glob(os.path.join(expert_trajectory_folder, '*.states'))
    total_avg_speed = np.zeros(6)
    traj_counter = 0
    print('Total number of trajectories :', len(trajectories))
    flag = False
    total_len = 0
    avg_speed = np.zeros(6)
    for idx, trajectory in enumerate(trajectories):
        print('The trajectory filename :', trajectory)
        traj = torch.load(trajectory, map_location=DEVICE)
        traj_np = traj.cpu().numpy()

        if not flag:
            print('State size :', traj[0].shape)
            flag = True

        total_len += len(traj)
        avg_speed = np.sum(traj_np[:,-6:],0)
        total_avg_speed += avg_speed
        print(avg_speed)
        print('Trajectory length :', traj_np.shape[0])
        #pdb.set_trace()

    pdb.set_trace()
    print('Average speed :', total_avg_speed/len(trajectories))

    print('Average len of trajectories :', total_len/len(trajectories))


def get_pedestrians_in_viscinity(state, viscinity):

    counter = 0
    for obs in state['obstacles']:

        dist = np.linalg.norm(obs['position']-state['agent_state']['position'], 1)
        if dist < viscinity:
            counter += 1

    return counter



def classify_pedestrians(annotation_file, viscinity):
    '''
    reads the annotation file and spits out important stats about the data
    '''

    tick_speed = 30
    #initialize world
    env = GridWorldDrone(display=False, is_onehot=False, 
                        seed=10, obstacles=None,
                        show_trail=False,
                        is_random=False,
                        show_orientation=True,
                        annotation_file=annotation_file,
                        subject=None,
                        external_control=False,
                        replace_subject=True,      
                        tick_speed=tick_speed,                  
                        rows=576, cols=720,
                        width=10)    
    
    subject_set = extract_subjects_from_file(annotation_file)
    avg_ped_per_subject = []
    for subject in subject_set:
        print(' Subject :', subject)
        state = env.reset_and_replace(ped=subject)

        nearby_peds_in_frame = 0
        total_frames = env.final_frame - env.current_frame
        while env.current_frame < env.final_frame:

            state, _, _, _ = env.step()
            
            nearby_peds_in_frame += get_pedestrians_in_viscinity(state, viscinity)
        
        avg_peds_per_frame = nearby_peds_in_frame/total_frames
        avg_ped_per_subject.append(avg_peds_per_frame)
        print('Avg peds nearby :', avg_peds_per_frame)

    subject_array = np.asarray(list(subject_set))
    avg_peds_per_subject_arr = np.asarray(avg_ped_per_subject)
    subject_array = subject_array[avg_peds_per_subject_arr.argsort()]
    avg_peds_per_subject_arr.sort()
    total = len(subject_set)

    #easy_cutoff = int(total*.5)
    #med_cutoff = int(total*.85)
    '''
    ##############################
    cutoff values for UCY 003 student dataset : {easy: 248, med: 375} 
    mean for easy = 0.048, med : .295, hard: .73
    ##############################
    cutoff values for zara 01 : {easy :90, med: 134}
    Easy : 0.0008541002858650179, Moderate : 0.07109769250746073, Difficult :0.40787718716023774 
    ##############################
    cutoff values for zara 02 : {easy: 105, med: 180}
    Easy : 0.017, Moderate : 0.206, Difficult: 0.71 
    ##############################
    cutoff values for UCY 001 student dataset : {easy: 250, med: 385}
    Mean value for easy: 0.16 med: 0.71 hard: 1.37 
    '''
    easy_cutoff = 250   
    med_cutoff = 385
   
    plt.plot(avg_peds_per_subject_arr, color='darkgreen')

    plt.vlines(easy_cutoff, 0, 
               avg_peds_per_subject_arr[easy_cutoff], 
               linestyles='dashed', color='lightgreen',
               label='Easy cutoff')
    plt.vlines(med_cutoff, 0, 
               avg_peds_per_subject_arr[med_cutoff],
               linestyles='dashed', color='limegreen',
               label='Med cutoff')
    plt.xlabel('Pedestrians sorted according to their average local density')
    plt.ylabel('Average number of pedestrians in the vicinity')
    print("The mean no. of pedestrians for different classes:")
    print("Easy : {}, Moderate : {}, Difficult :{} ".format(np.mean(avg_peds_per_subject_arr[0:easy_cutoff]),
                                                            np.mean(avg_peds_per_subject_arr[easy_cutoff:med_cutoff]),
                                                            np.mean(avg_peds_per_subject_arr[med_cutoff:])))

    plt.show()

    pdb.set_trace()
    easy_arr = subject_array[0:easy_cutoff]
    medium_arr = subject_array[easy_cutoff:med_cutoff]
    hard_arr = subject_array[med_cutoff:]
    return easy_arr, medium_arr, hard_arr



def read_training_data(parent_folder):

    '''
    Function that reads data from a folder and creates a combined tensor containing
    the x and y of a dataset for supervised learning.
    input:
        parent_folder : path to the parent folder.
    
    output:
        output_tensor : a tensor of size m x n, where m is the 
                        number of samples, which, in this case is the
                        total number of states in all the trajectories 
                        in the parent folder and 
                            n is the sum of the size of the
                            state vector + the size of the action vector.

    '''
    if os.path.isdir(parent_folder):
        print("folder")
    else:
        print("Folder does not exist.")
        sys.exit()

    actions_list = glob.glob(os.path.join(parent_folder, '*.actions'))
    trajectory_list = glob.glob(os.path.join(parent_folder, '*.states'))
    
    actions_list.sort()
    trajectory_list.sort()

    output_tensor =  None

    for i in range(len(actions_list)):
        trajectory = trajectory_list[i]
        trajectory_actions = actions_list[i]

        torch_traj = torch.load(trajectory, map_location=DEVICE)
        traj_np = torch_traj.cpu().numpy()

        torch_actions = torch.load(trajectory_actions, map_location=DEVICE)
        torch_actions = torch_actions.type(torch.DoubleTensor).to(DEVICE)

        if len(torch_actions.shape)==1:
            torch_actions = torch_actions.unsqueeze(1)
        action_np = torch_actions.cpu().numpy()
        joint_info = torch.cat((torch_traj, torch_actions), 1)
        if output_tensor is None:
            output_tensor = joint_info
        else:

            output_tensor = torch.cat((output_tensor, joint_info), 0)

    return output_tensor



def group_pedestrians_from_trajectories(trajectory_folder,
                                         proximity_threshold=45,
                                         temporal_threshold=0.45):
    
    '''
    Returns 2 dictionaries where:
    dictionary 1 stores the group information of each pedestrian.
        (key : value ) - (ped-id : [group])
    
    dictionary 2 stores the group information of each group
        (key : value) - (group-id : [ped id of the peds in the group])

    dictionary 3 stores the classification information of each pedestrian,
    stating if they are solo or not
        (key : value) - (ped-id : 1/0 [based on group or solo])
    '''

    assert os.path.isdir(trajectory_folder), "bad folder path"

    trajectories = glob.glob(os.path.join(trajectory_folder, '*.states.npy'))
    trajectory_group_info_dict = {}

    missing_peds = []
    ped_index = 1
    for trajectory in trajectories:

        file_name = trajectory.split('/')[-1]
        ped_id = file_name.strip().split('_')[3]
        
        if ped_id!=ped_index:
            pdb.set_trace()
            missing_ped.append(ped_index)
            ped_index += 1
        '''
        group_info = find_pedestrian_group(trajectory, 
                                              proximity_threshold=proximity_threshold,
                                              temporal_threshold=temporal_threshold)
        
        trajectory_group_info_dict[ped_id] = group_info
        '''
    
    #create the second dictionary
    classification_info = {}
    for key in trajectory_group_info_dict.keys():
        if len(trajectory_group_info_dict[key])>0:
            classification_info[key] = 1
        else:
            classification_info[key] = 0
    
    return trajectory_group_info_dict, classification_info




def get_index_from_pedid_university_student_dataset(ped_id):
    '''
    Given the ped id will return the serial number of the pedestrian
    (basically taking into account for the missing pedestrians)
    Total number of pedestrian - 430, max pedestrian id 434

    input:
        ped_id : pedestrian id (int)
    
    output:
        ped_index : serial index of the pedestrian
    '''
    #load the precomputed list that contains the difference between index and 
    #id for each of the pedestrians

    with open('../envs/expert_datasets/\
/university_students/data_info/university_students_ped_id_difference_dict.json', 'r') as fp:
        diff_dict = json.load(fp)
    

    return ped_id + diff_dict[str(ped_id)] 


def get_index_from_pedid_zara_02(ped_id):
    #ped_id 148 is missing
    #id indexing start from 1
    #total pedestrians : 203
    pedindex = None
    if int(ped_id) <= 147:
        pedindex = ped_id - 1
    else:
        pedindex = ped_id -2  
    
    return pedindex 


def get_index_from_pedid_zara_01(ped_id):
    #no pedestrians are missing and the ped_id 
    #starts from 0
    #total pedestrians 148
    return ped_id



def find_pedestrian_group(pedestrian_trajectory, 
                          proximity_threshold=40,
                          temporal_threshold=0.7):
    """
    Returns a list containing the id of the pedestrians who are likely to belong to a group
    input:
        trajectory : a filename containing the trajectory of the said pedestrian.
        proximity_threshold : integer containing how close the nearby pedestrians have to
                              be to the subject inorder to be qualified to be in the same group
        
        temporal_threshold : a float mentioning for what fraction of the total trajectory does a
                             nearby pedestrian have to be inside the proximity_threshold of a 
                             subject in order to be classified as a fellow group member

    output:
        group : a list containing the ids of the pedestrians that are in the same group as the
                subject pedestrian
    """

    #read the trajectory from the file

    assert os.path.isfile(pedestrian_trajectory), 'Bad trajectory file!'

    traj = np.load(pedestrian_trajectory, allow_pickle=True)
    total_frames = len(traj)

    #maintain a dictionary of all the pedestrians who pierce the proximity threshold
    #and see who stays inside the proximity threshold for how long.
    proximity_tracker = {}
    for state in traj:

        for obs in state['obstacles']:
            agent_position = state['agent_state']['position']
            ped_dist = dist_2d(agent_position, obs['position'])

            if ped_dist <= proximity_threshold:

                if obs['id'] not in proximity_tracker.keys():
                    proximity_tracker[obs['id']] = 1
                else:
                    proximity_tracker[obs['id']] += 1
    
    #check through the proximity tracker to see which of the nearby pedestrians
    #qualify to be in a group with the pedestrian at hand using the temporal threshold

    temporal_threshold_frames = temporal_threshold * total_frames 
    ped_group = []
    for people in proximity_tracker.keys():

        if proximity_tracker[people] > temporal_threshold_frames:
            ped_group.append(people)
    
    return ped_group





if __name__=='__main__':

    #****************************************************************
    #********section to grouping pedestrians
    '''
    dict1, dict2 = group_pedestrians_from_trajectories('/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/\
university_students/annotation/traj_info/frame_skip_1/students003/Raw_expert_states/', 
                                        proximity_threshold=45, temporal_threshold=0.45)
    pdb.set_trace()
    #find_pedestrian_group('/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/\
#university_students/annotation/traj_info/frame_skip_1/students003/Raw_expert_states/\
#traj_of_sub_13_segment1.states.npy', proximity_threshold=45, temporal_threshold=0.45)

    '''
    #***************************************************************
    '''
    parent_folder = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/university_students/annotation/\
traj_info/frame_skip_1/students003/DroneFeatureRisk_speedv2_with_raw_actions'
    output_tensor = read_training_data(parent_folder)
    pdb.set_trace()
    '''
    #********* section to extract trajectories **********
    
    folder_name = './expert_datasets/'
    dataset_name = 'university_students/annotation/'
    
    #the annotation file to be used to run and extract expert demonstrations
    file_n = 'processed/frame_skip_1/students001_processed_corrected.txt'

    #name of the folder to save the extracted results
    feature_extractor_name = 'Raw_expert_states'

    #path to save the folder
    to_save = 'traj_info/frame_skip_1/students001/'
    
    #complete path to the annotation file for the environment
    file_name = folder_name + dataset_name + file_n

    #complete path to the folder to save the extracted expert
    #demonstrations
    folder_to_save = folder_name + dataset_name + to_save + feature_extractor_name
    

    #parameters for the feature extractors 
    grid_size=agent_width=obs_width = 10
    step_size = 2

    
    #initialize the feature extractor
    feature_extractor = DroneFeatureRisk_speedv2(thresh1=18, thresh2=30,
                                               agent_width=10, obs_width=10,
                                               debug=False,
                                               grid_size=10, step_size=step_size)

    
    #feature_extractor = VasquezF1(agent_width*6, 0, 2)
    #feature_extractor = VasquezF2(agent_width*6, 0, 2)

    #feature_extractor = VasquezF3(agent_width*6)
    #feature_extractor = LocalGlobal(window_size=11, grid_size=grid_size,
    #                                agent_width=agent_width, 
    #                                obs_width=obs_width,
    #                                step_size=step_size,
    #                              )
    
    #feature_extractor = Fahad(36, 60, 0.5, 1.0)
    #feature_extractor = GoalConditionedFahad(36, 60, 0.5, 1.0)
    #print(extract_subjects_from_file(file_name))
    extract_trajectory(file_name, 
                       folder_to_save, 
                       feature_extractor=None, 
                       show_states=False,
                       extract_action=False,
                       display=False, trajectory_length_limit=None)
    
    
    #****************************************************
    #******** section to record trajectories
    '''
    step_size = 2
    agent_size = 10
    grid_size = 10
    obs_size = 10
    window_size = 15
    
    num_trajs = 60
    path_to_save = './DroneFeatureRisk_speed_blank_slate_test'

    
    feature_extractor = LocalGlobal(window_size=window_size, agent_width=agent_size,
                                    step_size=step_size, 
                                    obs_width=obs_size,
                                    grid_size=grid_size, 
                                    )
    
    feature_extractor = DroneFeatureSAM1(step_size=step_size,
                                         thresh1=15,
                                         thresh2=30,
                                         agent_width=agent_size,
                                         grid_size=grid_size,
                                         obs_width=obs_size,
                                         )
    
    
    feature_extractor = DroneFeatureRisk_v2(step_size=step_size,
                                     thresh1=15,
                                     thresh2=30,
                                     agent_width=agent_size,
                                     grid_size=grid_size,
                                     obs_width=obs_size,
                                     )
    
    feature_extractor = DroneFeatureRisk_speed(step_size=step_size,
                                               thresh1=10,
                                               thresh2=15,
                                               agent_width=agent_size,
                                               grid_size=grid_size,
                                               obs_width=obs_size)
    
    env = GridWorldDrone(display=True, is_onehot=False, agent_width=agent_size,
                         seed=10, obstacles=None, obs_width=obs_size,
                         step_size=step_size, width=grid_size,
                         show_trail=False,
                         is_random=True,
                         step_reward=.005,
                         annotation_file=None,
                         subject=None,
                         consider_heading=True,
                         replace_subject=False,
                         show_orientation=True,
                         #rows=576, cols=720
                         rows=100, cols=100) 


    
    record_trajectories(num_trajs, env, feature_extractor, path_to_save)
    '''
    #***************************************************** 

    
    #******** section for preprocessing data ************
    '''
    file_name = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/university_students/annotation/students001.vsp'

    intval = preprocess_data_from_control_points(file_name, speed_multiplier=1, 
                                            frame=1)
    
      
    '''
    #preprocess_data_from_stanford_drone_dataset('annotations.txt')
    #****************************************************
    
   
    #*****************************************************
    #********** getting information of the trajectories
    '''
    expert_traj_folder = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/\
university_students/annotation/traj_info/frame_skip_1/students003/DroneFeatureRisk_speedv2'
    get_expert_trajectory_info(expert_traj_folder)
    
    '''
    #*****************************************************
    #************* classify pedestrians based on presence of nearby obstacles
    
    '''
    annotation_file = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/\
university_students/annotation/processed/frame_skip_1/students001_processed_corrected.txt'
    #annotation_file = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/\
#data_zara/annotation/processed/crowds_zara02_per_frame.txt'
    #get_index_from_pedid_zara_02(annotation_file)
    easy, med, hard = classify_pedestrians(annotation_file, 30)
    pdb.set_trace()
    '''