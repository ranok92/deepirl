import argparse
import sys 
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
                        
parser.add_argument('--fig-title', type=str, required=True)

parser.add_argument('--ped-file-list', nargs="*", type=str,
                    default=["./ped_lists/easy.npy", 
                    "./ped_lists/med.npy", "./ped_lists/hard.npy"],
                    help='List of the files containing the pedestrian ids to be \
                        used in the current metric plot.')




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

        master_dictionary_list.append(agent_dictionary_list)

    ped_list = np.zeros(1)
    for list_name in ped_file_list:
        ped_list = np.concatenate((ped_list, np.load(list_name)), axis=0)
   
    ped_list = ped_list[1:].astype(int)

    ped_list = np.sort(ped_list)

    return master_dictionary_list, ped_list



def plot_histogram(list_of_dictionary, list_of_agent_names, 
                   metric_name, fig_title, metric_info=None, 
                   ped_list=None):
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
                #print('ped', ped)
                #print('seed_counter', seed_counter)
                #print('i', i)
                run_information_array_agent\
                        [seed_counter][i][:] = run_info['metric_results'][ped]\
                                                                [metric_name][0]
                i += 1
            seed_counter += 1

        run_information_list.append(run_information_array_agent)

        #agent_counter += 1
    
    bins = 200
    alpha = 0.3
    for i in range(len(list_of_dictionary)):
        plt.hist(np.mean(run_information_list[i], axis=0),
                 bins=bins,
                 label=list_of_agent_names[i],
                 alpha=alpha)
    
    plt.title(fig_title)
    plt.legend()
    plt.show()


def barplots_with_errorbars(list_of_dictionary, list_of_agent_names, 
                            metric_name, fig_title, metric_info=None,
                            ped_list=None):
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
                  needs to be retrieved

    output:
        a barplot with error bars against each value in the metric

    """

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
    for agent in list_of_dictionary:
        seed_counter = 0

        run_information_array_agent = np.zeros([len(agent),
                                                len(ped_list),
                                                metric_value_len])
        for run_info in agent:
            #reading data from a single metric dictionary
            i = 0
            for ped in ped_list:
                run_information_array_agent\
                        [seed_counter][i][:] = run_info['metric_results'][ped]\
                                                                [metric_name][0]
                i += 1
            seed_counter += 1
        agent_counter += 1
        run_information_list.append(run_information_array_agent)
    
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

        for i in range(len(list_of_dictionary)):
            mean_list.append(np.mean(np.mean(run_information_list[i][:, :, info], axis=1)))
            std_list.append(np.std(np.mean(run_information_list[i][:, :, info], axis=1)))
        
        ax.bar(x_axis, mean_list, yerr=std_list, 
            alpha=alpha, capsize=capsize, align='center')
        ax.set_xticks(x_axis)
        ax.set_xticklabels(list_of_agent_names)
        ax.yaxis.grid(True)
        title = fig_title+metric_info[info]
        ax.set_title(title)
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
                                    ped_list=None):
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
            ax.set_xticks(x_axis)
            ax.yaxis.grid(True)
            ax.legend()
            title = fig_title
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

    """
    plot_histogram(master_dictionary_list, agent_names, 
                   args.metric_name, args.fig_title, ped_list=ped_list)
    
    #################################################
    #uncomment this to get barplots with erros
    """
    Sample commandline command:
         python plot_results.py --parent-directory './results'
                                --fig-title 'Trajectory smoothness-'
                                --metric-name 'compute_trajectory_smoothness'
                                --metric-info 'total' 'average'
    

    barplots_with_errorbars(master_dictionary_list, agent_names, 
                   args.metric_name, args.fig_title,
                   metric_info=args.metric_info,
                   ped_list=ped_list) 
    """
    
    #################################################
    #uncomment this to get line plots over time frames.
    """
    Sample commandline command:
        python plot_results.py --parent-directory './results' 
                               --fig-title 'Distance_to_nearest_ped_over_time'  
                               --metric-name 'distance_to_nearest_pedestrian_over_time'
        
        **here the ped_list is modified below.

    
    ped_list = [11, 12, 13]
    plot_information_per_time_frame(master_dictionary_list, 
                                    agent_names, 
                                    args.metric_name, args.fig_title,
                                    metric_info=args.metric_info,
                                    ped_list=ped_list) 
    """