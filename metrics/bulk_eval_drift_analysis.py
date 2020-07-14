import os 
from shutil import copyfile
import argparse
import pdb
import re
import glob
from subprocess import Popen
import pdb

from argparse import Namespace, ArgumentParser

from evaluate_drift_deep_maxent import main as evaluate_drift_maxent_main

parser = ArgumentParser()

parser.add_argument('--parent-directory', 
                    type=str,
                    required=True,
                    help='Base folder containing the policy networks.')

parser.add_argument('--output-directory',
                    type=str,
                    required=True,
                    help='New parent folder containing the evaluation dictionaries.')


parser.add_argument('--annotation-file',
                    type=str,
                    default='../envs/expert_datasets/\
university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt')

parser.add_argument('--max-ep-length',
                    type=int,
                    default=1000,
                    )

parser.add_argument("--disregard-collisions", action="store_true")

parser.add_argument(
                    "--drift-timesteps",
                    type=lambda s: [int(t) for t in s.split(",")],
                    default=['5,10,15,30,45,50,70,90,110'],
                    )

parser.add_argument('--ped-list', type=str, nargs="*", default=["./ped_lists/easy.npy",
                    "./ped_lists/med.npy", "./ped_lists/hard.npy"],
                    help="Pedestrian list to work with.")

parser.add_argument('--start-interval', type=int, default=20, help='The initial number of \
                    frames after which the position of the agent will be reset.')

parser.add_argument('--end-interval', type=int, default=60, help='The final number of \
                    frames after which the position of the agent will be reset.')

parser.add_argument('--increment-interval', type=int, default=30, help='The number of \
                    frames by which the interval should increase.')


def get_appropriate_feature_extractor(dir_name):
    '''
    Given a dir_name returns the feature extractor
    '''
    if dir_name=='Goal-conditioned SAM':
        return 'GoalConditionedFahad'

    if dir_name=='Vasquez F1':
        return 'VasquezF1'

    if dir_name=='Vasquez F3':
        return 'VasquezF3'

    if 'Risk-features' in dir_name:
        return 'DroneFeatureRisk_speedv2'



bulk_args = parser.parse_args()
PARENT_DIR = bulk_args.parent_directory
OUTPUT_DIRECTORY = './results/' + bulk_args.output_directory 


def main(bulk_args):
    '''
    '''
    for root, dirs, files in os.walk(PARENT_DIR):
        #checking the parent directory
        for dir_val in dirs:
            
            output_dir_feature = os.path.join(OUTPUT_DIRECTORY, dir_val)
            feature_path = os.path.join(root, dir_val)
            print(dir_val)
            feature_name = get_appropriate_feature_extractor(dir_val)
            print(feature_name)

            #create directory for feature_extractor
            if os.path.isdir(output_dir_feature):
                pass
            else:
                #pdb.set_trace()
                os.makedirs(output_dir_feature)
            
            #checking the feature directory
            #pdb.set_trace()

            policy_file_list_for_feat = []
            agent_type_for_feat = []
            for root_2, dirs_seeds, files_2 in os.walk(feature_path):


                dirs_seeds.sort()
                for seed_dir in dirs_seeds:

                    #output_dir_seed = os.path.join(output_dir_feature, seed_dir)

                    #create the directories
                    
                   
                    seed_folder_path = os.path.join(root_2, seed_dir)

                    for root_seed, _ , files_seed in os.walk(seed_folder_path):

                        #orig_path_list.append(os.path.realpath(symlink))
                        for policy in files_seed:
                            #copy the file if it is a model
                            policy_filename = os.path.join(seed_folder_path, policy)
                            policy_file_list_for_feat.append(policy_filename)
                            agent_type_for_feat.append('Policy_network')
                            
                        break

                break

            per_feat_args = Namespace(
                    max_ep_length=bulk_args.max_ep_length,
                    feat_extractor=feature_name,
                    annotation_file=bulk_args.annotation_file,
                    ped_list=bulk_args.ped_list,                                    
                    policy_path=policy_file_list_for_feat,
                    agent_type=agent_type_for_feat,
                    save_filename=feature_name, 
                    dont_replace_subject=True,
                    disregard_collisions=bulk_args.disregard_collisions,
                    drift_timesteps=bulk_args.drift_timesteps,
                    render=False,
                    policy_net_hidden_dims=[256],
                    start_interval=bulk_args.start_interval,
                    end_interval=bulk_args.end_interval,
                    increment_interval=bulk_args.increment_interval
                    )
            #print(per_feat_args)
            pdb.set_trace()

            evaluate_drift_maxent_main(per_feat_args)


        break


if __name__=='__main__':

    main(bulk_args)