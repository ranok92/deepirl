import numpy as np
from matplotlib import pyplot as plt 
from argparse import ArgumentParser, Namespace 
from evaluate_drift_deep_maxent import plot_drift_results


import pdb
parser = ArgumentParser()

#parser.add_argument('--drift-files', type=str, nargs="*", default=None)

parser.add_argument('--parent-folder', type=str, required=True)

parser.add_argument('--ped-list', type=str, nargs="*", default=["./ped_lists/easy.npy",
                    "./ped_lists/med.npy", "./ped_lists/hard.npy"],
                    help="Pedestrian list to work with.")

parser.add_argument('--start-interval', type=int, default=20, help='The initial number of \
                    frames after which the position of the agent will be reset.')

parser.add_argument('--end-interval', type=int, default=60, help='The final number of \
                    frames after which the position of the agent will be reset.')

parser.add_argument('--increment-interval', type=int, default=30, help='The number of \
                    frames by which the interval should increase.')

parser.add_argument('--dataset', type=str, default='UCY',
                    help='Name of the dataset on which the \
                    current drift files have been calculated.')


args = parser.parse_args()

ped_list = np.zeros(1)
for file in args.ped_list:
    ped_list = np.concatenate((ped_list,np.load(file)), axis=0)

ped_list = ped_list[1:]
ped_list = np.sort(ped_list).astype(int)

'''
master_drift_results = []
for file in args.drift_files:
    drift_info = np.load(file)

    if drift_info.shape[1]!=ped_list.shape[0]:
        print("there is inconsistency in stuff.")

'''


sec_args = Namespace(parent_folder=args.parent_folder,
                 ped_list=ped_list,
                 start_interval=args.start_interval,
                 end_interval=args.end_interval,
                 dataset=args.dataset,
                 increment_interval=args.increment_interval)


plot_drift_results(sec_args)
