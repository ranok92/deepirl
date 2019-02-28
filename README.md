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


