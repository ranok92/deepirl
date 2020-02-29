import argparse
import sys 
import pickle
import pdb
import numpy as np
sys.path.insert(0,"..")
from matplotlib import pyplot as plt 

parser = argparse.ArgumentParser()

parser.add_argument('--dict-files', nargs="*", type=str, help="List of filenames \
                    of the dictionaries to be read.")

parser.add_argument('--agent-names', nargs="*", help='List of agent names.')

parser.add_argument('--metric-name', type=str, help='Name of the metric to be used for the \
                     plot.')

parser.add_argument('--metric-info', type=str, nargs='*', default=None,
                    help="Description of the values of each of the indexes if \
                        the metrics result is more than one.")
                        
parser.add_argument('--fig-title', type=str, required=True)

parser.add_argument('--ped-file-list', nargs="*", type=str,
                    default=["./ped_lists/easy.npy", 
                    "./ped_lists/med.npy", "./ped_lists/hard.npy"] ,
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
                   metric_name, fig_title, metric_info=None, ped_list=None):
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

    run_information_array = np.zeros([len(list_of_dictionary),
                                     len(list_of_dictionary[0])*len(ped_list)],
                                     metric_value_len)

    
    total_peds = len(ped_list)
    agent_counter = 0
    seed_counter = 0
    for agent in list_of_dictionary:
        seed_counter = 0
        for run_info in agent:
            #reading data from a single metric dictionary
            i = 0
            for ped in ped_list:
                run_information_array[agent_counter]\
                        [(seed_counter*total_peds)+i][:] = run_info['metric_results'][ped]\
                                                                [metric_name][0]
                i += 1
            seed_counter+=1
        agent_counter += 1
    
    bins = 200
    alpha = 0.3
    for i in range(len(list_of_dictionary)):
        plt.hist(run_information_array[i,:],
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
        list_of_agent_names : a list of strings containing the name of the agents
        metrics : a string containing the name of the metrics 
                  that needs to be retrieved from the dictionary.
                  Names should exactly match the key in the dictonary
        fig_title : Title to be used in the figure.
        ped_list : a sorted numpy array containing the id of the 
                  pedestrians for whom the metric information 
                  needs to be retrieved

    output:
        a barplot with error bars for the agents and metric

    """

    if metric_info is None:
        metric_value_len = 1
    else:
        metric_value_len = len(metric_info)
    pdb.set_trace()
    run_information_array = np.zeros([len(list_of_dictionary),
                                     len(list_of_dictionary[0])*len(ped_list)],
                                     metric_value_len)

    
    total_peds = len(ped_list)
    agent_counter = 0
    seed_counter = 0
    for agent in list_of_dictionary:
        seed_counter = 0
        for run_info in agent:
            #reading data from a single metric dictionary
            i = 0
            for ped in ped_list:
                print(ped)
                run_information_array[agent_counter]\
                        [(seed_counter*total_peds)+i][:] = run_info['metric_results'][ped]\
                                                                [metric_name][0]
                i += 1
            seed_counter+=1
        agent_counter += 1
    
    #plotting parameters
    bins = 200
    alpha = 0.3
    capsize = 20

    mean_list = []
    std_list = []

    x_axis = np.arange(len(list_of_dictionary))
    fig, ax = plt.subplots()

    for info in range(metric_value_len):
        for i in range(len(list_of_dictionary)):
            mean_list.append(np.mean(run_information_array[i][:][info]))
            std_list.append(np.std(run_information_array[i][:][info]))
        
        ax.bar(x_axis, mean_list, yerr=std_list, 
            alpha=alpha, capsize=capsize, align='center')
        ax.set_xticks(x_axis)
        ax.set_xticklabels(list_of_agent_names)
        ax.yaxis.grid(True)
        title = fig_title+metric_info[info]
        ax.set_title(title)
        plt.show()
    


if __name__=='__main__':

    args = parser.parse_args()


    dict_files = [
                 ['./results/Fahad/Fahad_seed2_highest_score_2020-02-28-15:07'], 
                 ['./results/DroneFeatureRisk_speedv2_Smoothing/Seed9/DroneFeature_smooth_seed9_highest_score_2020-02-28-15:21']
                 ]

    pdb.set_trace()
    master_dictionary_list, ped_list = read_data(dict_files, args.ped_file_list)
    '''
    plot_histogram(master_dictionary_list, args.agent_names, 
                   args.metric_name, args.fig_title, ped_list)
    '''

    barplots_with_errorbars(master_dictionary_list, args.agent_names, 
                   args.metric_name, args.fig_title, ped_list)