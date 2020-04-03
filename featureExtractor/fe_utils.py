""" Utils for dealing with feature extractors in general. """
import sys

import featureExtractor.drone_feature_extractor as dfe
import featureExtractor.gridworld_featureExtractor as gfe


def load_feature_extractor(
    fe_name,
    step_size=2,
    agent_width=10,
    grid_size=10,
    obs_width=10,
    *args,
    **kwargs
):

    """
    Automagically loads the correct feature extractor class based on their
    name. args and kwargs can be passed for custom initialization of a
    feature extractor, otherwise reasonable defaults are initialized.

    :raises ValueError: If feature extractor name cannot be matched.
    :return: Initialized feature extractor.
    :rtype: Feature extractor class.
    """

    if args or kwargs:
        try:
            fe_class = getattr(dfe, fe_name)
        except AttributeError:
            fe_class = getattr(gfe, fe_name)

        return fe_class(*args, *kwargs)

    elif fe_name == "Onehot":
        feat_ext = gfe.OneHot(grid_rows=10, grid_cols=10)

    elif fe_name == "SocialNav":
        feat_ext = gfe.SocialNav()

    elif fe_name == "FrontBackSideSimple":

        feat_ext = gfe.FrontBackSideSimple(
            thresh1=1,
            thresh2=2,
            thresh3=3,
            thresh4=4,
            step_size=step_size,
            agent_width=agent_width,
            obs_width=obs_width,
        )

    elif fe_name == "LocalGlobal":

        feat_ext = gfe.LocalGlobal(
            window_size=5,
            grid_size=grid_size,
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
        )

    elif fe_name == "DroneFeatureSAM1":

        feat_ext = dfe.DroneFeatureSAM1(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=5,
            thresh2=10,
        )

    elif fe_name == "DroneFeatureRisk":

        feat_ext = dfe.DroneFeatureRisk(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=15,
            thresh2=30,
        )

    elif fe_name == "DroneFeatureRisk_v2":

        feat_ext = dfe.DroneFeatureRisk_v2(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=15,
            thresh2=30,
        )

    elif fe_name == "DroneFeatureRisk_speed":

        feat_ext = dfe.DroneFeatureRisk_speed(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=10,
            thresh2=15,
        )

    elif fe_name == "DroneFeatureRisk_speedv2":

        feat_ext = dfe.DroneFeatureRisk_speedv2(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=18,
            thresh2=30,
        )

    elif fe_name == "VasquezF1":
        feat_ext = dfe.VasquezF1(6 * agent_width, 18, 30)

    elif fe_name == "VasquezF2":
        feat_ext = dfe.VasquezF2(6 * agent_width, 18, 30)

    elif fe_name == "VasquezF3":
        feat_ext = dfe.VasquezF3(agent_width)

    elif fe_name == "Fahad":
        feat_ext = dfe.Fahad(36, 60, 0.5, 1.0)

    elif fe_name == "GoalConditionedFahad":
        feat_ext = dfe.GoalConditionedFahad(36, 60, 0.5, 1.0)

    else:
        raise ValueError(
            "Could not discern feature extractor."
            + " Make sure feature extractor name is valid!"
        )

    return feat_ext
