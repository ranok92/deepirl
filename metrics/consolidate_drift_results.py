import numpy as np
from matplotlib import pyplot as plt 
import argparse
from evaluate_drift_deep_maxent import plot_drift_results

import pdb
parser = argparse.ArgumentParser()

parser.add_argument('--drift-files', type=str, nargs="*", default=None)


parser.add_argument('--ped-list', type=str, nargs="*", default=["./ped_lists/easy.npy",
                    "./ped_lists/med.npy", "./ped_lists/hard.npy"],
                    help="Pedestrian list to work with.")

parser.add_argument('--start-interval', type=int, default=20, help='The initial number of \
                    frames after which the position of the agent will be reset.')

parser.add_argument('--end-interval', type=int, default=60, help='The final number of \
                    frames after which the position of the agent will be reset.')

parser.add_argument('--increment-interval', type=int, default=30, help='The number of \
                    frames by which the interval should increase.')



args = parser.parse_args()

ped_list = np.zeros(1)
for file in args.ped_list:
    ped_list = np.concatenate((ped_list,np.load(file)), axis=0)

ped_list = np.sort(ped_list).astype(int)


master_drift_results = []
for file in args.drift_files:
    drift_info = np.load(file)

    if drift_info.shape[1]!=ped_list.shape[0]:
        print("there is inconsistency in stuff.")

    master_drift_results.append(drift_info.tolist())
    pdb.set_trace()

plot_drift_results(master_drift_results, args.start_interval, 
                   args.end_interval, args.increment_interval)
