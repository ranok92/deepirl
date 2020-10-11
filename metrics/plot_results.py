import argparse
import sys 
import os
import pickle
import pdb
import numpy as np
sys.path.insert(0,"..")
from matplotlib import pyplot as plt 


from metric_utils import read_files_from_directories
parser = argparse.ArgumentParser()

parser.add_argument('--parent-directory', type=str, help="Name of the \
    parent directory containing the metric dictionaries.")

parser.add_argument('--metric-name', type=str, help='Name of the metric to be used for the \
                     plot.')

parser.add_argument('--metric-info', type=str, nargs='*', default=None,
                    help="Description of the values of each of the indexes if \
                        the metrics result is more than one.")
                        
parser.add_argument('--ped-file-list', nargs="*", type=str,
                    default=["../envs/expert_datasets/university_students/data_info/ped_lists_ucy_003/ucy_003_easy.npy", 
                    "../envs/expert_datasets/university_students/data_info/ped_lists_ucy_003/ucy_003_med.npy",
                    "../envs/expert_datasets/university_students/data_info/ped_lists_ucy_003/ucy_003_hard.npy"],
                    help='List of the files containing the pedestrian ids to be \
                        used in the current metric plot.')

parser.add_argument('--x-axis', type=str, default=None)
parser.add_argument('--y-axis', type=str, default=None)

def read_data(list_of_files, ped_file_list):
    """
    Reads list of data files and pedlist files
    """
    master_dictionary_list = []
    for agent_file in list_of_files:
        agent_dictionary_list = []
        for seed_file in agent_file:
            with open(seed_file, 'rb') as pickle_file:
                agent_dictionary_list.append(pickle.load(pickle_file))
                #pdb.set_trace()
        master_dictionary_list.append(agent_dictionary_list)

    ped_list = np.zeros(1)
    for list_name in ped_file_list:
        ped_list = np.concatenate((ped_list, np.load(list_name)), axis=0)
   
    ped_list = ped_list[1:].astype(int)

    ped_list = np.sort(ped_list)

    return master_dictionary_list, ped_list



def plot_histogram(list_of_dictionary, list_of_agent_names, 
                   metric_name, metric_info=None, 
                   ped_list=None, x_axis=None, y_axis=None):
    """
    given a list of dictionary containing run information of different agents
    will return a histogram plot featuring the agent groups and the metrics provided
    against the pedestrians asked for.
    input:
        list_of_dictionary : a 2d list of dictionaries arranged in this format
                             [
                              [dict_1_agent_1, dict_2_agent_1, . . ]
                              [dict_1_agent_2, dict_2_agent_2, . . ]
                             ] 
        list_of_agent_names : a list of strings containing the name of the agents
        metrics : a string containing the name of the metrics 
                  that needs to be retrieved from the dictionary.
                  Names should exactly match the key in the dictonary
        fig_title : String containing the title to be used in the figure.
        ped_list : a sorted numpy array containing the id of the 
                  pedestrians for whom the metric information 
                  needs to be retrieved

    output:
        a mixture of gaussian style plot of the metrics of the different agents
    """

    if metric_info is None:
        metric_value_len = 1
    else:
        metric_value_len = len(metric_info)

    '''
    max_seed_number = 0
    for agent in list_of_dictionary:
        if max_seed_number < len(agent):
            max_seed_number = len(agent)
    
    run_information_array = np.zeros([len(list_of_dictionary),
                                     max_seed_number*len(ped_list),
                                     metric_value_len])
    '''
    run_information_list = []
    total_peds = len(ped_list)
    #agent_counter = 0
    seed_counter = 0
    for agent in list_of_dictionary:
        seed_counter = 0
        run_information_array_agent = np.zeros([len(agent),
                                                len(ped_list),
                                                metric_value_len])

        for run_info in agent:
            #reading data from a single metric dictionary
            i = 0
            for ped in ped_list:
                '''
                run_information_array_agent\
                        [seed_counter][i][:] = run_info['metric_results'][ped]\
                                                             [metric_name][0]
                '''

                if metric_name == 'compute_trajectory_smoothness' or \
                   metric_name == 'compute_distance_displacement_ratio':
                    
                    if run_info['metric_results'][ped]['trajectory_length'][0] >  100:           
                        
                        run_information_array_agent\
                            [seed_counter][i][:] = run_info['metric_results'][ped]\
                                                                    [metric_name][0]
                        #selected += 1
                    else:
                        run_information_array_agent\
                            [seed_counter][i][:] = np.nan
                else:

                    run_information_array_agent\
                            [seed_counter][i][:] = run_info['metric_results'][ped]\
                                                                    [metric_name][0]  

                i += 1
            seed_counter += 1

        run_information_list.append(run_information_array_agent)

        #agent_counter += 1
    pdb.set_trace()
    bins = 200
    alpha = 0.3
    for i in range(len(list_of_dictionary)):
        plt.hist(np.nanmean(run_information_list[i], axis=0),
                 bins=bins,
                 label=list_of_agent_names[i],
                 alpha=alpha)

    if x_axis is not None:
        plt.xlabel(x_axis)
    if y_axis is not None:
        plt.ylabel(y_axis)

    plt.legend()
    plt.show()


def barplots_with_errorbars(list_of_dictionary, list_of_agent_names, 
                            metric_name, metric_info=None,
                            ped_list=None, x_axis=None, y_axis=None):
    """
    given a list of dictionary containing run information of different agents
    will return a histogram plot featuring the agent groups and the metrics provided
    against the pedestrians asked for.

    **The function now stores a dictionary containing all the relevant information
    for simplified future plots.

    The dictionary contains the following keys:

        metric name: name of the metric
        metric_info (optional): addiitonal info on metric
        ped_list : list of pedestrian involved in the calculation 
                    of the metric. 
        agent_list: List of the agents involved.
                    The order of the agent_list, mean_list and stddev_list
                    should be consistent.

        mean_list: List containing the mean values
        stddev_list: List containing the standard deviation of the values

    input:
        list_of_dictionary : a 2d list of dictionaries arranged in this format
                             [
                              [dict_1_agent_1, dict_2_agent_1, . . ]
                              [dict_1_agent_2, dict_2_agent_2, . . ]
                             ] 
        list_of_agent_names : a list of strings containing the name of the agents.
                              This will be used to populate the legend.

        metric_name : a string containing the name of the metric
                  that needs to be retrieved from the dictionary.
                  Names should exactly match the key in the dictonary

        metric_info : If the information for the metric is of size>1, 
                For example: 'compute_trajectory_smoothness' comprises of a 
                    tuple (total change in orientation, 
                            avg change in orientation), 
                    then the metric_info
                    should be a list of length 2 containing the description of the metric
                    eg: 'total smoothness' 'avg smoothness'

        ped_list : a sorted numpy array containing the id of the 
                  pedestrians for whom the metric information 
                  needs to be retrieved

    output:
        a barplot with error bars against each value in the metric


    """
    bar_color_list = ['r','g','m','y','b','k']
    if metric_info is None:
        metric_value_len = 1
    else:
        metric_value_len = len(metric_info)

    max_seed_number = 0
    for agent in list_of_dictionary:
        if max_seed_number < len(agent):
            max_seed_number = len(agent)

    run_information_list = []

    total_peds = len(ped_list)
    agent_counter = 0
    seed_counter = 0
    seed_rejected = 0
    for agent in list_of_dictionary:
        seed_counter = 0
        seed_rejected = 0

        run_information_array_agent = np.zeros([len(agent),
                                                len(ped_list),
                                                metric_value_len])
        #pdb.set_trace()
        selected_total = 0
        for run_info in agent:
            #reading data from a single metric dictionary
            i = 0
            selected = 0
            #pdb.set_trace()
            seed_info_array = np.zeros((total_peds, metric_value_len))
            seed_info_array_NAN = seed_info_array+np.NAN
            for ped in ped_list:

                if metric_name == 'compute_trajectory_smoothness' or \
                   metric_name == 'compute_distance_displacement_ratio':
                    
                    if run_info['metric_results'][ped]['trajectory_length'][0] > 5 or run_info['metric_results'][ped]['goal_reached'][0]:           
                        
                        #run_information_array_agent\
                        #    [seed_counter][i][:] = run_info['metric_results'][ped]\
                        #                                            [metric_name][0]


                        seed_info_array[i, :] = run_info['metric_results'][ped]\
                                                                    [metric_name][0]

                        selected += 1
                        selected_total += 1
                    else:
                        #run_information_array_agent\
                        #    [seed_counter][i][:] = np.nan
                        seed_info_array[i, :] = np.nan
                else:

                    run_information_array_agent\
                            [seed_counter][i][:] = run_info['metric_results'][ped]\
                                                                    [metric_name][0]  

                    seed_info_array[i, :] = run_info['metric_results'][ped]\
                                                                    [metric_name][0]                                                     
                i += 1

            #select those seeds that atleast have success in the above criteria in 
            #25% of all the trajectories
            if metric_name == 'compute_trajectory_smoothness' or \
                   metric_name == 'compute_distance_displacement_ratio':
                if selected > total_peds*0:
                    run_information_array_agent[seed_counter] = seed_info_array[:]
                else:
                    #print('SEED rejected')
                    seed_rejected += 1
                    run_information_array_agent[seed_counter] = seed_info_array_NAN[:]
            else:

                run_information_array_agent[seed_counter] = seed_info_array[:]

            seed_counter += 1
            
            print("For agent :no. of peds that clear the criteria: {}/{}".format(selected, total_peds))
        print("For agent :no. of peds that clear the criteria: {}/{}".format(selected_total, total_peds*seed_counter))
        print("Seeds rejected : {} / {}".format(seed_rejected, seed_counter))
        run_information_list.append(run_information_array_agent)

    #select those pedestrians that have success in all the seeds 
    run_information_array = np.array(run_information_list)
    
    #pdb.set_trace()
    '''
    selected_ped_list_agent = [[[] for i in range(len(list_of_agent_names))] for j in range(metric_value_len)]

    selected_ped_list_all = [[] for j in range(metric_value_len)]
    for k in range(metric_value_len):
        for i in range(total_peds):
            selected_ped_list = []
            ped_score = run_information_array[:, :, i, :]
            select_ped = True
            for j in range(len(list_of_agent_names)):
                ped_score_agent = ped_score[j,:,:]
                #print(ped_score)
                #pdb.set_trace()
                if sum(np.isnan(ped_score_agent))[k] >= int(seed_counter/3):
                    #print("rejected")
                    select_ped = False
                    pass
                else:
                    #print("selected")
                    selected_ped_list_agent[k][j].append(i)
            #pdb.set_trace()
            if select_ped:
                selected_ped_list_all[k].append(i)
        #select those pedestrians that have success in all the agents 

    run_information_list = run_information_array.take(selected_ped_list_all[0], axis=2)
    pdb.set_trace()
    '''    
    #plotting parameters
    bins = 200
    alpha = 0.3
    capsize = 20

    x_axis = np.arange(len(list_of_dictionary))
    #pdb.set_trace()
    for info in range(metric_value_len):
        
        fig, ax = plt.subplots()
        mean_list = []
        std_list = []
        lower_bound_list = []
        upper_bound_list = []
        for i in range(len(list_of_dictionary)):
            mean_over_seeds = np.nanmean(run_information_list[i][:, :, info], axis=1)
            mean_over_seeds_nonzero = mean_over_seeds[mean_over_seeds.nonzero()]
            #clip outliers
            cur_std = np.nanstd(mean_over_seeds_nonzero)
            mean_over_seeds_clipped_up = mean_over_seeds_nonzero[mean_over_seeds_nonzero <= np.nanmean(mean_over_seeds_nonzero) + 2*cur_std]
            mean_over_seeds_clipped = mean_over_seeds_clipped_up[mean_over_seeds_clipped_up >= np.nanmean(mean_over_seeds_nonzero) - 2*cur_std]
            #print("Mean over seeds clipped :", mean_over_seeds_clipped, np.mean(mean_over_seeds_clipped), np.std(mean_over_seeds_clipped))
            #pdb.set_trace() 

            mean_list.append(np.nanmean(mean_over_seeds_clipped))
            std_list.append(np.nanstd(mean_over_seeds_clipped))
            lower_bound_list.append(np.nanmean(mean_over_seeds_nonzero) - np.nanmin(mean_over_seeds_nonzero))
            upper_bound_list.append(np.nanmax(mean_over_seeds_nonzero) - np.nanmean(mean_over_seeds_nonzero))

        print("Mean list :", mean_list)
        print("Std list :", std_list)
        #pdb.set_trace()
        bounds = [lower_bound_list, upper_bound_list]
        #print(np.nanmean(run_information_list[i][:, :, info], axis=1))
        #pdb.set_trace()

        barlist = ax.bar(x_axis, mean_list, yerr=std_list, 
                        alpha=alpha, capsize=capsize, align='center')

        for bar in barlist:
            bar.set_color(bar_color_list[i])
            i = (i+1)%len(bar_color_list)
    
        if y_axis is not None:
            ax.set_ylabel(y_axis)


        #Create the storage dictionary
        storage_dict = {}

        storage_dict['metric_name'] = metric_name

        storage_dict['metric_info'] = metric_info
        storage_dict['ped_list'] = ped_list
        storage_dict['agent_list'] = list_of_agent_names
        storage_dict['mean_list'] = mean_list
        storage_dict['std_list'] = std_list

        comparing_agents = ''
        for agent in list_of_agent_names:
            comparing_agents += agent + '-'

        comparing_agents = comparing_agents[0:-1]
        filename_info_dict = "./numerical_results/"+ metric_name + "_ " +\
                            comparing_agents + "_info_dict.pk"

        with open(filename_info_dict, 'wb') as fp:
            pickle.dump(storage_dict, fp)
            fp.close()


        ax.set_xticks(x_axis)
        ax.set_xticklabels(list_of_agent_names)
        ax.yaxis.grid(True)
        #set the y axis label
        '''
        y_label = ''
        if metric_name=='goal_reached':
            y_label = 'Fraction of runs reaching the goal'

        if metric_name=='compute_trajectory_smoothness':
            y_label = 'Change in orientation (in degrees)'

        if metric_name=='count_collisions':
            y_label = 'Average number of collisions encountered per trajectory'

        plt.ylabel(y_label)
        '''
        #title = fig_title+metric_info[info]
        #ax.set_title(title)
        plt.show()
        '''
        file_name = title+'.fig.pickle'
        file_name_eps = title+'.svg'
        with open(file_name, 'w') as f:
            pickle.dump(fig, f)
        plt.savefig(file_name_eps)
        '''




def plot_information_per_time_frame(list_of_dictionary, list_of_agent_names, 
                                    metric_name, fig_title, metric_info=None,
                                    ped_list=None, x_label=None, y_axis=None):
    """
    Generates a plot of the metric against time for the given pedestrians 
    in the pedestrian list
    input:
        list_of_dictionary : a 2d list of dictionaries arranged in this format
                             [
                              [dict_1_agent_1, dict_2_agent_1, . . ]
                              [dict_1_agent_2, dict_2_agent_2, . . ]
                             ] 
        list_of_agent_names : a list of strings containing the name of the agents.
                              This will be used to populate the legend.

        metrics : a string containing the name of the metric
                  that needs to be retrieved from the dictionary.
                  Names should exactly match the key in the dictonary

        fig_title : Title to be used in the figure. If metric_info is not none,
                    this gets appended with the metric_info to create the final 
                    title.
        metric_info : If the information for the metric is of size>1, 
                        For example: 'compute_trajectory_smoothness' comprises of a 
                            tuple (total change in orientation, 
                                    avg change in orientation), 
                            then the metric_info
                            should be a list of length 2 containing the description of the metric
                            eg: 'total smoothness' 'avg smoothness'
                        
        ped_list : a sorted numpy array containing the id of the
                  pedestrians for whom the metric information
                  needs to be retrieved.
                  **N.B. This method will produce a separate plot for results 
                  against each pedestrian to prevent cluttering.

    output:
        a line plot with errorbars where the xaxis is the time frame and y axis is 
        the value of the metric
    """

    if metric_info is None:
        metric_value_len = 1
    else:
        metric_value_len = len(metric_info)

    total_agents = len(list_of_agent_names)
    total_peds = len(ped_list)
    
    max_traj_len = -1
    for ped in ped_list:
        for agent in list_of_dictionary:
            for run_info in agent:
                if len(run_info['metric_results'][ped][metric_name][0]) > max_traj_len:
                    max_traj_len = len(run_info['metric_results'][ped][metric_name][0])


    x_axis = np.arange(max_traj_len)
    x_ticks = np.arange(max_traj_len, step=int(max_traj_len/10))
    for ped in ped_list:
        agent_counter = 0
        fig, ax = plt.subplots()
        for agent in list_of_dictionary:
            #reading data from a single metric dictionary
            total_seeds = len(agent)
            mask_array = np.ma.empty((total_seeds, max_traj_len))
            mask_array.mask = True
            seed_counter = 0
            for run_info in agent:
                cur_traj_len = len(run_info['metric_results'][ped][metric_name][0])
                mask_array[seed_counter, 0:cur_traj_len] = (run_info['metric_results'][ped]\
                                                                [metric_name][0])
                seed_counter += 1

            mean_arr = mask_array.mean(axis=0)
            std_arr = mask_array.std(axis=0)

            ax.plot(mean_arr, label=list_of_agent_names[agent_counter], alpha=0.8)
            ax.fill_between(x_axis, mean_arr-std_arr, mean_arr+std_arr, alpha=0.3)
            ax.set_xticks(x_ticks)
            ax.yaxis.grid(True)
            if x_axis is not None:
                ax.set_xlabel(x_label)
            if y_axis is not None:
                ax.set_ylabel(y_axis)
            ax.legend()
            #title = fig_title
            ax.set_title(title)
            agent_counter += 1

        plt.show()

    
if __name__=='__main__':

    args = parser.parse_args()


    metric_info_dict = read_files_from_directories(args.parent_directory)
    agent_names = []
    file_list = []
    for key in metric_info_dict.keys():

        if len(metric_info_dict[key]) > 0:
            agent_names.append(key)
            file_list.append(metric_info_dict[key])



    master_dictionary_list, ped_list = read_data(file_list, args.ped_file_list)
        
    #################################################
    #uncomment this function to get historgram plots.
    """
    Sample commandline command:
        python plot_results.py --parent-directory './results' 
                               --fig-title 'Distance displacement ratio'
                               --metric-name 'compute_distance_displacement_ratio'

    
    
    plot_histogram(master_dictionary_list, agent_names, 
                   args.metric_name, 
                   ped_list=ped_list, x_axis=args.x_axis,
                   y_axis=args.y_axis)
    """
    #################################################
    #uncomment this to get barplots with erros
    """
    Sample commandline command:
         python plot_results.py --parent-directory './results'
                                --fig-title 'Trajectory smoothness-'
                                --metric-name 'compute_trajectory_smoothness'
                                --metric-info 'total' 'average'
    
    """  
    barplots_with_errorbars(master_dictionary_list, agent_names, 
                   args.metric_name,
                   metric_info=args.metric_info,
                   ped_list=ped_list, x_axis=args.x_axis,
                   y_axis=args.y_axis) 
     

    #################################################
    #uncomment this to get line plots over time frames.
    """
    Sample commandline command:
        python plot_results.py --parent-directory './results' 
                               --fig-title 'Distance_to_nearest_ped_over_time'  
                               --metric-name 'distance_to_nearest_pedestrian_over_time'
        
        **here the ped_list is modified below.

 
    ped_list = [11, 13, 36, 91, 10, 77]
    plot_information_per_time_frame(master_dictionary_list, 
                                    agent_names, 
                                    args.metric_name, args.fig_title,
                                    metric_info=args.metric_info,
                                    ped_list=ped_list, x_label=args.x_axis,
                                    y_axis=args.y_axis) 
    """