## Deep IRL

Repo maintained by Abhisek Konar, Bobak H. Baghi.

### Installation

To install the necessary python environment, either globally or in a virtual
env, run:

`pip -r requirements3.txt`

NOTE: For the old python2 files, run `pip -r requirements.txt` in a seperate
python2 installation.

### Running

The code is structured around running experiment scripts, located in the
experiments folder. Make sure you change directory to this folder to ensure
proper function.

For example, to train an actor-critic model on a gridworld, run: 

`python gridworld_ac.py`

To run the trained agent and generate expert trajectories, run: 

`python gridoworld_ac.py --policy-path './saved-models/0.pt' --play`

In any of the above, pass the `--render` flag to display the environment, e.g.:

`python gridworld_ac.p --render`

 `python gridoworld_ac.py --policy-path './saved-models/0.pt' --play --render`

Note: The flags/options passed to experiment files are unique to each
experiment, and users are encouraged to write their own experiments based on
their needs. The above is simply a sample.

###Generating expert demonstrations with a new feature extractor.

The code in the repository is modular and feature extractors can be easily be swapped in and 
out at the start of rl/irl runs. Changing feature extractors while running RL is easy, but
swapping feature extractors while training IRL needs the expert demonstrations of that 
exact feature extractors. Additionally, the parameters of the feature extractor used to create 
the expert demonstrations should be same as the parameters being used in the training.

Steps to generate expert demonstrations:
1. Open /envs/drone_data_utils.py
2. Under the main(), find the section for extracting trajectories.
   Something like this #********* section to extract trajectories*********
3. Uncomment the section. Comment the other sections.
4. Put the name of the desired feature extractor in the variable 'feature_extractor_name'. 
   This will be used to name the folder to store the generated expert demonstrations.
5. Adjust the feature extractor parameters and initialize the feature extractor.
6. Run the file.


