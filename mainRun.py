import argparse
import os
import datetime

#import files for the three different modules of the pipeline

import deepirl
import rlmethods
import featureExtractor

'''
this should be the main function that takes in the 3 modules needed to run the pipeline
1.features
2.Rl part
3.IRL part
Also, it should make passing arguments easy and convenient. 

    Possible arguments:
    1. New run/Load?
    2. With or without display
    3. Environment arguments:
    
    4. Features to be used.
        4.1 If local window, size of the window.
    5. RL method to be used
        5.1 Number of iterations till convergence.
    6. IRL method to be used
    

'''
def read_arguments():

    parser = argparse.ArgumentParser(description='Enter arguments to run the pipeline.')

    #arguments for external files that might be necessary to run the program
    parser.add_argument('--cost_network', type=str , help='Enter the location of the file storing the state dictionary of the cost network.')
    parser.add_argument('--policy_network', type=str , help='Enter the location of the file storing the state dictionary of the Policy network.')
    parser.add_argument('--state_dictionary', type=str , help='Enter the type of environment on which to run the algo (obstacle/no obstacle)')
    parser.add_argument('--expert_trajectory_file' , type=str , help='Enter the location of the file containing the exeprt trajectories.')

    #network hyper parameters
    parser.add_argument('--cost_network_input', type=int , default=29, help='Input layer size of cost network.None if you have specified cost network state dict.')
    parser.add_argument('--cost_network_hidden', nargs='+' ,type=int ,default = [256,256], help='Hidden size of cost network.None if you have specified cost network state dict.')
    parser.add_argument('--cost_network_output', type=int , default = 1 ,help='Output layer size of cost network.None if you have specified cost network state dict.')

    parser.add_argument('--policy_network_input', type=int , default=29, help='Input layer size of policy network.None if you have specified policy network state dict.')
    parser.add_argument('--policy_network_hidden', nargs='+' ,type=int ,default = [256,256], help='Hidden layer size of policy network.None if you have specified policy network state dict.')
    parser.add_argument('--policy_network_output', type=int , default = 4 ,help='Output layer size of policy network.None if you have specified policy network state dict.')

    #other run hyper parameters like optimizer and all???


    #run hyperparameters
    parser.add_argument('--irl_iterations', type=int , help='Number of times to iterate over the IRL part.')
    parser.add_argument('--no_of_samples', type=int , help='Number of samples to be taken to create agent state visitation frequency.')
    parser.add_argument('--rl_iterations' , type = int , help='Number of iterations to be performed in the RL section.')

    #arguments for the I/O of the program
    parser.add_argument('--display_board' , type=bool , default=False, help='True/False based on if the program needs to display the environment.')
    parser.add_argument('--on_server' , type=bool , default= True, help='True/False based on if the program is running on server. **False when running on server?')
    parser.add_argument('--store_results' , type=bool , default=True)
    parser.add_argument('--plot interval' , type=int , default= 10 , help='Iterations after which the plot of the loss and the reward curve will be stored.')
    parser.add_argument('--savedict_policy_interval' , type=int , default = 100, help='Iterations after which the policy network will be stored.')
    parser.add_argument('--savedict_cost_interval' , type=int , default = 1,  help='Iterations after which the cost network will be stored.')

    #arguments for the broader pipeLine
    parser.add_argument('--rl_method' , type=str , help='Enter the RL method to be used.')
    parser.add_argument('--feature_space', type=str , help='Enter the type of features to be used to get the state of the agent.')
    parser.add_argument('--irl_method' , type=str , help='Enter the IRL method to be used.')


    args = parser.parse_args()

    return args


def dictToFilename(dict):

    filestr = ''

    for key in dict.keys():

        filestr+=str(dict[key])
        filestr+='_'

    return filestr



def assertargs(args):
    #add assertions later

    return 0

def arrangeDirForStorage(irlMethod, rlMethod , costNNparams , policyNNparams):

    storageDict = {}
    curDay = str(datetime.datetime.now().date())
    curtime = str(datetime.datetime.now().time())

    basePath = 'saved-models-irl/'
    subPathPolicy = curDay+'/'+curtime+'/'+'PolicyNetwork/'
    subPathCost = curDay+'/'+curtime+'/'+'CostNetwork/'
    curDirPolicy = basePath + subPathPolicy
    curDirCost = basePath + subPathCost
    fileNamePolicy = irlMethod+'-'+rlMethod+'-'+dictToFilename(policyNNparams)

    fileNameCost = irlMethod+'-'+rlMethod+'-'+dictToFilename(costNNparams)

    if not os.path.exists(curDirPolicy):
        os.makedirs(curDirPolicy)
    if not os.path.exists(curDirCost):
        os.makedirs(curDirCost)

    storageDict['basepath'] = basePath+curDay+'/'+curtime+'/'
    storageDict['costDir'] = curDirCost
    storageDict['policyDir']= curDirPolicy
    storageDict['costFilename'] = fileNameCost
    storageDict['policyFilename'] = fileNamePolicy

    return storageDict


'''
example running statements:


python mainRun.py --state_dictionary 'no obstacle' --expert_trajectory_file 'expertstateinfolong_50.npy' --irl_iterations 10 --no_of_samples 100 --rl_iterations 200 --rl_method='Actor_Critic' --irl_method='DeepMaxEnt'

'''



if __name__=='__main__':

    args = read_arguments()

    #batch of mandatory arguments
    features = args.feature_space
    rlMethod = args.rl_method
    IRLMethod = args.irl_method
    demofile = args.expert_trajectory_file
    saveInfo = args.store_results
    display = args.display_board
    onServer = args.on_server


    #batch of conditions
    costNetwork = args.cost_network
    policyNetwork = args.policy_network
    costNNparams = {}
    policyNNparams = {}

    costNNparams['input'] = args.cost_network_input
    costNNparams['hidden'] = args.cost_network_hidden
    costNNparams['output'] = args.cost_network_output

    policyNNparams['input'] = args.policy_network_input
    policyNNparams['hidden'] = args.policy_network_hidden
    policyNNparams['output'] = args.policy_network_output


    stateDict,_ = deepirl.getstateDict(args.state_dictionary)
    irlIterations = args.irl_iterations
    sampling_no = args.no_of_samples
    rlIterations = args.rl_iterations

    storageInfoDict = arrangeDirForStorage(IRLMethod , rlMethod , costNNparams , policyNNparams)

    deepirl.deepMaxEntIRL(demofile , rlMethod,  costNNparams , costNetwork , policyNNparams , policyNetwork , irlIterations , sampling_no ,  rlIterations , store=saveInfo , storeInfo=storageInfoDict , render = display, onServer = onServer)


    #deepirl.deepMaxEntIRL() should have provision for taking in costNetwork
    #and policyNetwork as parameter so as to load pretrained network weights
    #if needed












