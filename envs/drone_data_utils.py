import numpy as np 
import torch
import glob

import sys
sys.path.insert(0, '..')

from envs.gridworld_drone import GridWorldDrone
from featureExtractor.gridworld_featureExtractor import SocialNav,LocalGlobal,FrontBackSideSimple


from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureRisk, DroneFeatureRisk_v2
from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speed, DroneFeatureRisk_speedv2
from scipy.interpolate import splev, splprep


from matplotlib import pyplot as plt
import os
import pdb
import pathlib
import copy
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

def get_dense_trajectory_from_control_points(list_of_ctrl_pts, frames):
    #given the control points returns a dense array of all the in between points
    #the control points should be of the format 
    #frame number, ped_id, y_coord, x_coord
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
        exit()
        return 0

    with open(annotation_file) as f:

        for line in f:
            prob_point_info = line.split(' - ')[0]
            if len(prob_point_info.split(' '))==4:
                point_info = prob_point_info.split(' ')
                processed_dict[ped_id].append([point_info[2], ped_id, frame_y - (float(point_info[1])+diff_y), float(point_info[0])+diff_x])
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


def preprocess_data_from_control_points(annotation_file, frame):
    '''
    given a annotation file containing spline control points converts that 
    to a frame-by-frame representation and writes that on a txt file with a easy to read format
    '''

    file_name = annotation_file.strip().split('/')[-1].split('.')[0]
    print(file_name)
    dir_name = ''
    for folder in annotation_file.strip().split('/')[0:-1]:
        dir_name += folder + '/'


    print(dir_name) 
    file_n = dir_name + file_name+'_processed_corrected'+'.txt'

    extracted_dict = read_data_from_file(annotation_file)
    pdb.set_trace()
    dense_info_list = []
    for ped in extracted_dict.keys():
            
        #for i in range(len(extracted_dict[ped])):
        #    print(extracted_dict[ped][i])
        dense_info = get_dense_trajectory_from_control_points(extracted_dict[ped], frame)

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




def extract_trajectory(annotation_file, feature_extractor, 
                       folder_to_save, display=False, 
                       show_states=False, subject=None, 
                       trajectory_length_limit=None):


    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    tick_speed = 30
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

    if subject is not None:
        subject_list = subject
    for sub in subject_list:
        print('Starting for subject :',sub)
        trajectory_info = []
        step_counter_segment = 0

        segment_counter = 1
        world.subject=sub
        world.reset()
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
            state,_,_,_ = world.step()
            step_counter_segment += 1
            #step_counter_trajectory += 1 
            if disp:
                feature_extractor.overlay_bins(state)

            state = feature_extractor.extract_features(state)

            state = torch.tensor(state)
            trajectory_info.append(state)

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
                    state_tensors = torch.stack(trajectory_info)
                    torch.save(state_tensors, os.path.join(folder_to_save, 'traj_of_sub_{}_segment{}.states'.format(str(sub), str(segment_counter))))
                    segment_counter += 1 
                    #pdb.set_trace()
                    step_counter_segment = 0 
                    trajectory_info = []
                    print('Segment {}: Start frame : {}'.format(segment_counter, 
                                                                world.current_frame))    




        if trajectory_length_limit is None:
            state_tensors = torch.stack(trajectory_info)
            torch.save(state_tensors, os.path.join(folder_to_save, 'traj_of_sub_%s.states' % str(sub)))
        
    if feature_extractor.debug_mode:
        feature_extractor.print_info()


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
    easy_arr = subject_array[0:200]
    medium_arr = subject_array[200:380]
    hard_arr = subject_array[380:]

    return easy_arr, medium_arr, hard_arr




if __name__=='__main__':


    
    #********* section to extract trajectories **********
    '''
    folder_name = './expert_datasets/'
    dataset_name = 'university_students/annotation/'
    file_n = 'processed/frame_skip_1/students003_processed_corrected.txt'


    feature_extractor = 'Dronefeature_risk_hit/'
    to_save = 'traj_info/frame_skip_1/students003/'
    file_name = folder_name + dataset_name + file_n

    folder_to_save = folder_name + dataset_name + to_save + feature_extractor
    
    grid_size=agent_width=obs_width = 10
    step_size = 2


    feature_extractor = DroneFeatureRisk_speedv2(thresh1=18, thresh2=30,
                                               agent_width=10, obs_width=10,
                                               debug=True,
                                               grid_size=10, step_size=step_size)

   
    #feature_extractor = LocalGlobal(window_size=11, grid_size=grid_size,
    #                                agent_width=agent_width, 
    #                                obs_width=obs_width,
    #                                step_size=step_size,
    #                              )
    
    
    #print(extract_subjects_from_file(file_name))
    extract_trajectory(file_name, feature_extractor, 
                       folder_to_save, show_states=False,
                       display=False, trajectory_length_limit=None)
    
    '''
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
    
    file_name = './t-junction-3.txt'

    intval = preprocess_data_from_control_points(file_name, 1)
    

    
    #preprocess_data_from_stanford_drone_dataset('annotations.txt')
    #****************************************************
    

    #*****************************************************
    #********** getting information of the trajectories
    '''
    expert_traj_folder = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/university_students/annotation/traj_info/frame_skip_1/students003/DroneFeatureRisk_speed_segments'
    get_expert_trajectory_info(expert_traj_folder)
    ''' 

    #*****************************************************
    #************* classify pedestrians based on presence of nearby obstacles
    '''

    annotation_file = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt'

    classify_pedestrians(annotation_file, 30)
    '''