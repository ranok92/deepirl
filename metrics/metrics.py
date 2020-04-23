""" A collection of metrics to evalaute agents with. """

import numpy as np
from featureExtractor.drone_feature_extractor import dist_2d, angle_between
import warnings


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
            prev_orientation = state["agent_head_dir"]

        change_in_orientation += abs(
            prev_orientation - state["agent_head_dir"]
        )
        prev_orientation = state["agent_head_dir"]

    return change_in_orientation, change_in_orientation / len(trajectory)


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

    total_displacement = np.linalg.norm(
        trajectory[0]["agent_state"]["position"]
        - trajectory[-1]["agent_state"]["position"]
    )

    total_distance = 0
    prev_pos = trajectory[0]["agent_state"]["position"]
    for state in trajectory[1:]:

        step_distance = np.linalg.norm(
            prev_pos - state["agent_state"]["position"]
        )

        total_distance += step_distance
        prev_pos = state["agent_state"]["position"]

    return total_displacement / total_distance


def proxemic_intrusions(trajectory, units_to_meters_ratio):
    """
    Calculate number of proxemic intrusions as defined by the proxemics model:
    "E. Hall, Handbook of Proxemics Research. Society for the Anthropology of
    Visual Communications, 1974."

    Based on Vasquez et. al's use in their paper:
    "IRL algos and features for robot navigation in crowds"

    This is a similified version ignoring public space and near/far phases,
    using thresholds found at:
    "https://en.wikipedia.org/wiki/Proxemics"

    :param trajectory: Trajectory of states generated by environment.
    :type trajectory: List of state_dicts

    :param units_to_meters_ratio: the ratio (env distance unit length) / (1
    meter). e.g. if 1 env unit is 10 cm, this ratio is 0.1.
    :type units_to_meters_ratio: float.

    :return: Number of intrusions in initimate, personal, and social spaces.
    :return type: tuple (int, int, int)
    """

    # Thresholds
    INTIMIATE_DISTANCE = 0.5 * units_to_meters_ratio
    PERSONAL_DISTANCE = 1.2 * units_to_meters_ratio
    SOCIAL_DISTANCE = 3.7 * units_to_meters_ratio

    # intrusion counts
    intimiate_intrusions = 0
    personal_intrusions = 0
    social_intrusions = 0

    for traj in trajectory:
        pedestrians = traj["obstacles"]
        agent_position = traj["agent_state"]["position"]

        for ped in pedestrians:
            ped_position = ped["position"]

            distance = dist_2d(ped_position, agent_position)

            if distance <= INTIMIATE_DISTANCE:
                intimiate_intrusions += 1
            elif INTIMIATE_DISTANCE < distance <= PERSONAL_DISTANCE:
                personal_intrusions += 1
            elif PERSONAL_DISTANCE < distance <= SOCIAL_DISTANCE:
                social_intrusions += 1
            elif distance > SOCIAL_DISTANCE:
                continue
            else:
                raise ValueError("Distance did not fit in any bins.")

    return intimiate_intrusions, personal_intrusions, social_intrusions


def anisotropic_intrusions(trajectory, radius, lambda_param=2.0):
    """
    Measures number of times the anisotropic radius of a pedestrian is violated by an agent.
    This implementation is based on the one presented in Vasquez's paper:
    "IRL algos and features for robot navigation in crowds."

    The anisotrpic radius can be though of a circular radius around
    pedestrians that is stretched to include more of the front direction of
    the pedestrian and less of the back direction. This is because being in
    front of a walking pedestrian is considered worse than being behind or to
    the sides of a walking pedestrian.

    :param trajectory: Trajectory of states generated by environment.
    :type trajectory: List of state_dicts.

    :param radius: (Base) radius to consider around pedestrians. This will be
    stretched according to lambda_param.
    :type radius: float.

    :param lambda_param: factor by which to stretch radius parameter,
    defaults to 2.0
    :type lambda_param: float, optional

    :raises ValueError: If for some reason the angle between (agent_pos -
    pedestrian_pos) and pedestrian_orientation does not fit in [0,2pi]

    :return: Number of times front, side, and back intrusions happen in
    trajectory.
    :rtype: tuple (int, int, int)
    """

    # intrusion counts
    back_anisotropic_intrusion = 0
    side_anisotropic_intrusion = 0
    front_anisotropic_intrusion = 0

    for traj in trajectory:
        agent_position = traj["agent_state"]["position"]
        pedestrians = traj["obstacles"]

        for ped in pedestrians:
            ped_position = ped["position"]
            ped_orientation = ped["orientation"]

            vector_to_agent = agent_position - ped_position

            if ped_orientation is None:
                warnings.warn(
                    "pedestrian orientation is none, setting to (1.0, 0.0)"
                )
                ped_orientation = np.array([1.0, 0.0])

            angle = angle_between(vector_to_agent, ped_orientation)
            distance = dist_2d(agent_position, ped_position)

            # evaluate if agent is in anisotropic radius
            anisotropy_factor = lambda_param - 0.5 * (1 - lambda_param) * (
                1 + np.cos(angle)
            )
            anisotropic_radius = radius * anisotropy_factor

            if distance < anisotropic_radius:
                if angle < 0.25 * np.pi:
                    front_anisotropic_intrusion += 1
                elif 0.25 * np.pi <= angle < 0.75 * np.pi:
                    side_anisotropic_intrusion += 1
                elif 0.75 * np.pi <= angle <= np.pi:
                    back_anisotropic_intrusion += 1
                else:
                    raise ValueError(
                        "Cannot bin angle in any of the thresholds."
                    )

    return (
        front_anisotropic_intrusion,
        side_anisotropic_intrusion,
        back_anisotropic_intrusion,
    )


def count_collisions(trajectory, agent_radius):
    """Counts the number of distinct collisions between agent and
    pedestrians in the provided trajectory.

    :param trajectory: Trajectory of agent as a list of state dictionaries as generated by the
    environment.
    :type trajectory: list of (state) dictionaries

    :param agent_radius: Radius of agents in the environment.
    :type agent_radius: float

    :return: number of collisions with pedestrians in trajectory.
    :rtype: int
    """
    collision_count = 0

    collision_list = []
    for state in trajectory:

        agent_pos = state["agent_state"]["position"]

        for pedestrian in state["obstacles"]:
            ped_position = pedestrian["position"]

            if dist_2d(ped_position, agent_pos) < 2 * agent_radius:
                if pedestrian["id"] not in collision_list:
                    collision_count += 1
                    collision_list.append(pedestrian["id"])
            else:
                if pedestrian["id"] in collision_list:
                    collision_list.remove(pedestrian["id"])
    return collision_count


def goal_reached(trajectory, goal_radius, agent_radius):
    """Returns true if the goal was reached in the last state of this
    trajectory. Goal is considered reached if the distance between agent and
    goal is less than their respective radii (provided it has not hit a
    pedestrian in the way)

    :param trajectory: Trajectory of state dictionaries generated by the
    environment.
    :type trajectory: list of state dictionaries.

    :param goal_radius: radius of goal.
    :type goal_radius: float

    :param agent_radius: radius of agent.
    :type agent_radius: float

    :return: True of distance between goal and agent is less than goal_radius
    + agent_radius.
    :rtype: Boolean.
    """

    if pedestrian_hit(trajectory, agent_radius):
        return False

    else:
        agent_position = trajectory[-1]["agent_state"]["position"]
        goal_position = trajectory[-1]["goal_state"]

        return (
            dist_2d(agent_position, goal_position)
            <= goal_radius + agent_radius
        )


def pedestrian_hit(trajectory, agent_radius):
    """Returns true if pedestrian is hit in any of the state in
    the trajectory.

    :param trajectory: trajecory comprised of states_dicts.
    :type trajectory: list of state_dicts
    :param agent_radius: width of the agent in environment.
    :type agent_radius: float.
    :return: True if pedestrian hit in last state_dict, false otherwise.
    :rtype: Boolean.
    """

    for state in trajectory:

        pedestrians = state["obstacles"]
        agent_position = state["agent_state"]["position"]

        for ped in pedestrians:
            ped_position = ped["position"]

            distance = dist_2d(ped_position, agent_position)

            if distance < 2 * agent_radius:
                return True

    return False


def distance_to_nearest_pedestrian_over_time(trajectory):
    """
    At each timestep in trajectory, calculates the distances to the
    nearest pedestrian.

    :param trajectory: trajectory comprised of states_dicts.
    :type trajectory: list of state_dicts
    :return: list of distances to nearest pedestrian, ordered from t=0 to t=end.
    :rtype: list of floats.
    """
    min_distances = []

    for traj in trajectory:
        agent_position = traj["agent_state"]["position"]
        pedestrians = traj["obstacles"]
        ped_positions = [ped["position"] for ped in pedestrians]
        distances = [
            dist_2d(ped_pos, agent_position) for ped_pos in ped_positions
        ]
        min_distances.append(np.min(distances))

    return min_distances


def trajectory_length(trajectory):
    """
    Returns the length of the trajectory.

    :param trajectory: Trajectory of agent in envrionment.
    :type trajectory: List of state dictionaries.
    :return: length of trajectory by calculating the distance between the position of
             the agent in consequtive frames.
    :rtype: int.
    """
    agent_cur_pos = trajectory[0]["agent_state"]["position"]
    traj_length = 0
    for state in trajectory[1:]:
        traj_length += dist_2d(agent_cur_pos, state["agent_state"]["position"])
        agent_cur_pos = state["agent_state"]["position"]
    return traj_length
