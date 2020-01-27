import numpy as np 
import pdb
def compute_trajectory_smoothness(trajectory):
    """
    Returns the total and per step change in the orientation (in degrees)
    of the agent during the duration of the trajectory
        input : trajectory (a list of states)
        output : total change in orientation,
                 avg change in orientation
    """

    change_in_orientation = 0
    prev_orientation = None
    for state in trajectory:

        if prev_orientation is None:
            prev_orientation = state['agent_head_dir']

        change_in_orientation += abs(prev_orientation - state['agent_head_dir'])
        prev_orientation = state['agent_head_dir']

    return change_in_orientation, change_in_orientation/len(trajectory)


def compute_distance_displacement_ratio(trajectory, compare_expert=True):
    """
    Returns the ration between the final displacement achieved by the agent and
    the distance travelled by the agent. Meant as an objective
    measure of the goodness of the path taken by the agent
    The value is in the range of [1 - 0], where values close to
    1 indicate that the path taken by the agent are close to
    optimal paths that could have been taken
        input : trajectory
        output : final dispacement/ total distance travelled
    """

    total_displacement = np.linalg.norm(trajectory[0]['agent_state']['position']-
                                    trajectory[-1]['agent_state']['position'])

    total_distance = 0
    prev_pos = trajectory[0]['agent_state']['position']
    for state in trajectory[1:]:

        step_distance = np.linalg.norm(prev_pos-
                                    state['agent_state']['position'])

        total_distance += step_distance
        prev_pos = state['agent_state']['position']

        
    return total_displacement/total_distance
