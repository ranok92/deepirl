import numpy as np 
import matplotlib.pyplot as plt
import pdb

def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)



def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    #this function is for [x, y] coordinates,
    #the input vectors v1 and v2 are [row, col]
    v1_corr = np.array([v1[1], -v1[0]])
    v2_corr = np.array([v2[1], -v2[0]])
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



class InformationCollector():
    '''
    run_info = A string that contains the name/type of agent whose information
    is being collected by the collector
    
    plot_info - plot the information 
    disp_info - display the information in the terminal
    store_info - store the information in a csv file.

    '''
    def __init__(self, 
                 run_info=None,
                 plot_info=False,
                 store_info=False,
                 disp_info=True,
                 save_folder=None,
                 thresh=30):


        #general information
        self.thresh = thresh
        self.run_information = run_info

        #information on the output format
        self.plot_info = plot_info
        self.store_info = store_info
        self.display_info = disp_info
        
        #information that changes per frame
        self.closest_obs_cur_frame = None 
        self.prev_agent_pos = None

        #information that changes per trajectory
        self.agent_start_position = None
        self.agent_cur_position = None
        self.min_dist_list = []
        self.avg_closeness_closest_obs_per_run = None
        self.avg_steps_with_other_obs = None
        self.avg_steps_against_other_obs = None
        self.cur_goal = None
        self.cur_traj_displacement = None
        self.traj_len = 0

        #information across multiple runs
        self.disp_to_steps_ratio = []
        self.min_avg_dist_from_obs_across_runs = []
        self.total_pedestrians_nearby = []
        self.total_pedestrians_against = []
        self.total_pedestrians_along = []

        self.avg_pedestrians_nearby_pf = []
        self.avg_pedestrians_against_pf = []
        self.avg_pedestrians_along_pf = []

        self.feature_extractor = None
        print('Information collector initalized')


    def reset_info(self, state):
        '''
        to reset all the trajectory based information the information collector collects
        '''
        self.min_dist_list = []
        self.obs_nearby_list = []
        self.obs_against_list = []
        self.obs_along_list = []


        self.information_tuple_plot = [('Displacement to steps ratio', self.disp_to_steps_ratio),
                                  ('Min avg distance from obstacle', self.min_avg_dist_from_obs_across_runs),
                                  ('Total pedestrians going against agent per run', self.total_pedestrians_against),
                                  ('Total pedestrians going along agent per run', self.total_pedestrians_along),
                                  ('Total pedestrians nearby', self.total_pedestrians_nearby),
                                  ('Avg pedestrians going against agent per frame', self.avg_pedestrians_against_pf),
                                  ('Avg pedestrians going along agent per frame', self.avg_pedestrians_along_pf),
                                  ('Avg pedestrians nearby', self.avg_pedestrians_nearby_pf)
                                  ]

        self.information_tuple_display = [('Displacement to steps ratio', self.disp_to_steps_ratio),
                                          ('Min avg distance from obstacle', self.min_avg_dist_from_obs_across_runs),
                                          ('Total pedestrians going against agent per run', self.total_pedestrians_against),
                                          ('Total pedestrians going along agent per run', self.total_pedestrians_along),
                                          ('Total pedestrians nearby', self.total_pedestrians_nearby),
                                          ('Avg pedestrians going against agent per frame', self.avg_pedestrians_against_pf),
                                          ('Avg pedestrians going along agent per frame', self.avg_pedestrians_along_pf),
                                          ('Avg pedestrians nearby', self.avg_pedestrians_nearby_pf)
                                          ]



        self.avg_obs_nearby = 0
        #self.cur_traj_displacement = np.linalg.norm(state['agent_state']['position'] - state['goal_state'], 1)
        self.agent_start_position = state['agent_state']['position']
        #print('the current displacement :', self.cur_traj_displacement)
        #self.avg_closeness_closest_obs_per_run = None
        self.cur_goal = state['goal_state']
        self.avg_steps_against_other_obs = 0
        self.avg_steps_with_other_obs = 0
        self.traj_len = 0
        self.prev_agent_pos = self.agent_start_position 


    def collect_information_per_frame(self, state):
        '''
        collects information like closest obstacle, number of pedestrians
        going against/along the agent.
        '''
        #print(self.prev_agent_pos)
        #print(state['agent_state']['position'])
        self.traj_len += np.linalg.norm(self.prev_agent_pos - state['agent_state']['position'])
        #print (self.traj_len)
        #pdb.set_trace()
        min_dist = 9999999999
        obs_list = state['obstacles']
        agent_state = state['agent_state']
        self.agent_cur_position = state['agent_state']['position']
        #if distance less than thresh, consider that for further calculations 
        total_counter = 0
        with_counter = 0
        against_counter = 0

        for obs_state in obs_list:

            distance = np.linalg.norm(obs_state['position'] - agent_state['position'])
            if distance <= min_dist:
                min_dist = distance


            if distance < self.thresh:

                total_counter += 1
                #consider these obstacles
                #check for the obstacles that are going along/against the agent
                #print('Agent state :', agent_state['orientation'])
                #print('Obs state :', obs_state['orientation'])
                if agent_state['orientation'] is not None and obs_state['orientation'] is not None:
                    angle = angle_between(agent_state['orientation'], obs_state['orientation'])
                    #print('The angle  :', angle*180/np.pi)
                    #pdb.set_trace()
                    if angle < np.pi/8:
                        #this is along the agent
                        with_counter += 1
                        #print('along')
                    if angle > np.pi*7/8:
                        #this is against the agent
                        against_counter += 1
                        #print('against')
        
        self.min_dist_list.append(min_dist)

        #print('Total counter :', total_counter)
        #print('with counter :', with_counter)
        #print('against counter :', against_counter)
        #pdb.set_trace()
        self.obs_nearby_list.append(total_counter)
        self.obs_along_list.append(with_counter)
        self.obs_against_list.append(against_counter)
        self.prev_agent_pos = state['agent_state']['position']

    def collab_end_traj_results(self):
        '''
        calculates information like:
            total pedestrians moving against/along the agent in the run.
            avg no. of pedestrians moving against/along the agent per frame
            displacement to steps taken ratio.
        '''
        #calculating displacement to steps taken ratio:
        self.cur_traj_displacement = np.linalg.norm(self.agent_start_position - self.agent_cur_position)

        self.disp_to_steps_ratio.append(self.cur_traj_displacement/(self.traj_len))

        #calculating min avg distance the agent kept across multiple runs
        self.min_avg_dist_from_obs_across_runs.append(sum(self.min_dist_list)/len(self.min_dist_list))

        #calculating the flow of pedestrians around the agent
        self.total_pedestrians_against.append(sum(self.obs_against_list))
        self.total_pedestrians_along.append(sum(self.obs_along_list))
        self.avg_pedestrians_along_pf.append(sum(self.obs_along_list)/len(self.obs_along_list))
        self.avg_pedestrians_against_pf.append(sum(self.obs_against_list)/len(self.obs_against_list))
        self.total_pedestrians_nearby.append(sum(self.obs_nearby_list))
        self.avg_pedestrians_nearby_pf.append(sum(self.obs_nearby_list)/len(self.obs_nearby_list))


    def plot_information(self):
        '''
        takes in a list of tuples where: 
            index 0 has the title 
            index 1 has the list of data 
        '''
        list_of_tuple = self.information_tuple_plot
        for i in range(len(list_of_tuple)):

            plt.figure('{} : {}'.format(self.run_information, list_of_tuple[i][0]))
            plt.plot(list_of_tuple[i][1])

        plt.show()

    def display_information(self):

        list_of_tuple = self.information_tuple_display
        print('For agent :', self.run_information)
        for i in range(len(list_of_tuple)):

            #pdb.set_trace()
            print('{} : {}'.format(list_of_tuple[i][0],
                                   sum(list_of_tuple[i][1])/len(list_of_tuple[i][1])))




    def collab_end_results(self):

        #collab and plot results
        if self.plot_info:
            self.plot_information()

        if self.display_info:
            self.display_information()

        '''
        plt.figure('Displacement to steps ratio')
        plt.plot(self.disp_to_steps_ratio)
        plt.figure('Min avg distance from obstacle')
        plt.plot(self.min_avg_dist_from_obs_across_runs)
        plt.figure('Total pedestrians going against agent per run')
        plt.plot(self.total_pedestrians_against)
        plt.figure('Total pedestrians going along agent per run')
        plt.plot(self.total_pedestrians_along)
        plt.figure('Total pedestrians nearby')
        plt.plot(self.total_pedestrians_nearby)
        plt.figure('Avg pedestrians going against agent per frame')
        plt.plot(self.avg_pedestrians_against_pf)
        plt.figure('Avg pedestrians going along agent per frame')
        plt.plot(self.avg_pedestrians_along_pf)
        plt.figure('Avg pedestrians nearby')
        plt.plot(self.avg_pedestrians_nearby_pf)
        plt.show()
        '''




