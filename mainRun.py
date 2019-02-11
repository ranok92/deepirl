import argparse
import datetime
import os

# import files for the three different modules of the pipeline
import matplotlib

# import rlmethods
# import featureExtractor

'''
this should be the main function that takes in the 3 modules needed to run the
pipeline:

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
    """
    Reads argments passed from command line.
    """

    parser = argparse.ArgumentParser(
        description='Enter arguments to run the pipeline.')

    # arguments for external files that might be necessary to run the program
    parser.add_argument(
        '--cost_network', type=str,
        help='file storing the state dictionary of the cost network.'
    )

    parser.add_argument(
        '--policy_network', type=str,
        help='File storing the state dictionary of the Policy network.'
    )

    parser.add_argument(
        '--state_dictionary', type=str,
        help='Environment on which to run the algo (obstacle/no obstacle)'
    )

    parser.add_argument(
        '--expert_trajectory_file', type=str,
        help='Path to file containing the exeprt trajectories.')

    # network hyper parameters
    parser.add_argument(
        '--cost_network_input', type=int, default=29,
        help='layer size of cost network. None if you have specified cost \
        network state dict.')

    parser.add_argument(
        '--cost_network_hidden', nargs='+', type=int, default=[256, 256],
        help='Hidden size of cost network.None if you have specified cost \
        network state dict.')

    parser.add_argument(
        '--cost_network_output', type=int, default=1,
        help='Output layer size of cost network.None if you have specified \
        cost network state dict.')

    parser.add_argument(
        '--policy_network_input', type=int, default=29,
        help='Input layer size of policy network.None if you have specified \
        policy network state dict.')

    parser.add_argument(
        '--policy_network_hidden', nargs='+', type=int, default=[256, 256],
        help='Hidden layer size of policy network.None if you have specified \
        policy network state dict.')

    parser.add_argument(
        '--policy_network_output', type=int, default=4,
        help='Output layer size of policy network.None if you have specified \
        policy network state dict.')

    # other run hyper parameters like optimizer and all???

    # run hyperparameters
    parser.add_argument('--irl_iterations', type=int,
                        help='Number of times to iterate over the IRL part.')

    parser.add_argument(
        '--no_of_samples', type=int,
        help='Number of samples to create agent state visitation frequency.')

    parser.add_argument(
        '--rl_iterations', type=int,
        help='Number of iterations to be performed in the RL section.')

    # arguments for the I/O of the program
    parser.add_argument(
        '--display_board', type=str, default='False',
        help='If True, draw envirnment.')

    parser.add_argument(
        '--on_server', type=str, default='True',
        help='False if program is to run on server.')

    parser.add_argument('--store_results', type=str, default='True')

    parser.add_argument(
        '--plot_interval', type=int, default=10,
        help='Iterations before loss and reward curve plots are stored.')

    parser.add_argument(
        '--savedict_policy_interval', type=int, default=100,
        help='Iterations after which the policy network will be stored.')

    parser.add_argument(
        '--savedict_cost_interval', type=int, default=1,
        help='Iterations after which the cost network will be stored.')

    # arguments for the broader pipeLine
    parser.add_argument(
        '--rl_method', type=str,
        help='Enter the RL method to be used.')

    parser.add_argument(
        '--feature_space', type=str,
        help='Type of features to be used to get the state of the agent.')

    parser.add_argument('--irl_method', type=str,
                        help='Enter the IRL method to be used.')

    parser.add_argument(
        '--run_type', type=str, default='train',
        help='Enter if it is a train run or a test run.(train/test).')

    parser.add_argument(
        '--verbose', type=str, default='False',
        help='Set verbose to "True" to get a myriad of print statements crowd\
        your terminal. Necessary information should be provided with either\
        of the modes.')

    parser.add_argument(
        '--no_of_testRuns', type=int, default=0,
        help='If --run_type set to test, then this denotes the number of test \
        runs you want to conduct.')

    _args = parser.parse_args()

    return _args


def dictToFilename(dict):

    filestr = ''

    for key in dict.keys():

        filestr += str(dict[key])
        filestr += '_'

    return filestr


def assertargs(args):
    # add assertions later

    return 0


def arrangeDirForStorage(irlMethod, rlMethod, costNNparams, policyNNparams):

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
    storageDict['policyDir'] = curDirPolicy
    storageDict['costFilename'] = fileNameCost
    storageDict['policyFilename'] = fileNamePolicy

    return storageDict


def parseBool(stringarg):

    if stringarg == 'True':

        return True
    if stringarg == 'False':

        return False

    return -1


'''
example running statements:

python mainRun.py - -state_dictionary 'no obstacle' - -display_board 'False' - -on_server 'True' - -expert_trajectory_file 'expertstateinfolong_50.npy' - -irl_iterations 10 - -no_of_samples 100 - -rl_iterations 200 - -rl_method = 'Actor_Critic' - -irl_method = 'DeepMaxEnt' - -run_type 'train' - -cost_network '/home/abhisek/Study/Robotics/deepirl/saved-models-irl/2019-01-31/16:40:33.387966/CostNetwork/DeepMaxEnt-Actor_Critic-29_[256, 256]_1_iteration_0.h5' - -policy_network '/home/abhisek/Study/Robotics/deepirl/saved-models-irl/2019-01-31/16:40:33.387966/PolicyNetwork/DeepMaxEnt-Actor_Critic-29_[256, 256]_4_iterEND_1.h5' - -no_of_testRuns 0

'''


if __name__ == '__main__':

    args = read_arguments()

    # batch of mandatory arguments
    features = args.feature_space
    rlMethod = args.rl_method
    IRLMethod = args.irl_method
    demofile = args.expert_trajectory_file
    saveInfo = parseBool(args.store_results)
    display = parseBool(args.display_board)
    onServer = parseBool(args.on_server)

    verbose = parseBool(args.verbose)
    runType = args.run_type
    testRuns = args.no_of_testRuns

    # batch of conditions
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

    irlIterations = args.irl_iterations
    sampling_no = args.no_of_samples
    rlIterations = args.rl_iterations

    plotIntervals = args.plot_interval
    rlModelStoreInterval = args.savedict_policy_interval
    irlModelStoreInterval = args.savedict_cost_interval

    # have to put this in the pipeline of the code, not touching this as of yet
    typeofEnvironment = args.state_dictionary

    print saveInfo
    if saveInfo:

        storageInfoDict = arrangeDirForStorage(
            IRLMethod, rlMethod, costNNparams, policyNNparams)

    else:

        storageInfoDict = None

    if onServer:

        matplotlib.use('Agg')

    import deepirl

    stateDict, _ = deepirl.getstateDict(args.state_dictionary)

    maxEntIrl = deepirl.DeepMaxEntIRL(demofile, rlMethod, costNNparams,
                                      costNetwork, policyNNparams,
                                      policyNetwork, irlIterations,
                                      sampling_no, rlIterations,
                                      store=saveInfo,
                                      storeInfo=storageInfoDict,
                                      render=display, onServer=onServer,
                                      resultPlotIntervals=plotIntervals,
                                      irlModelStoreInterval=irlModelStoreInterval,
                                      rlModelStoreInterval=rlModelStoreInterval,
                                      testIterations=testRuns, verbose=verbose)

    if runType == 'train':

        maxEntIrl.runDeepMaxEntIRL()

    if runType == 'test':

        print 'Starting test branch . . .'
        maxEntIrl.testMaxDeepIRL()

    else:

        print 'I have not coded this option yet.'
