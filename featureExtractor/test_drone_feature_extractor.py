""" Implements tests for drone_feature_extractor.py """
import unittest
import sys
import numpy as np
from numpy import testing as nptest
from hypothesis import strategies as st
from hypothesis.extra import numpy as npthesis
from hypothesis import given, settings

sys.path.insert(0, "..")

import featureExtractor.drone_feature_extractor as dfe

settings.register_profile("longer-deadline", deadline=500, database=None)
settings.load_profile("longer-deadline")


def is_onehot(arr):
    """
    Helper function to see if numpy array arr is one-hot encoded.

    :param arr: numpy array.
    :type arr: np.array.
    :return: True if array is one-hot, false if not.
    :rtype: Boolean.
    """
    zeros = np.isclose(arr, np.zeros(arr.shape))
    ones = np.isclose(arr, np.ones(arr.shape))
    return np.all(np.logical_or(zeros, ones))


class VectorMathTests(unittest.TestCase):
    """
    Tests vector math functions as defined in drone_feature_extractor.py
    """

    def test_angle_between(self):
        """
        Tests angle_between function.
        """
        ones_vect = np.ones(2)
        north_vect = np.array([0, 1])
        south_vect = np.array([0, -1])
        east_vect = np.array([1, 0])
        west_vect = np.array([-1, 0])
        slightly_above_west = np.array([-1, 0.1])

        self.assertEqual(dfe.angle_between(ones_vect, ones_vect), 0.0)
        self.assertEqual(dfe.angle_between(east_vect, east_vect), 0.0)
        self.assertEqual(dfe.angle_between(west_vect, west_vect), 0.0)
        self.assertEqual(dfe.angle_between(ones_vect, north_vect), np.pi / 4)
        self.assertEqual(dfe.angle_between(north_vect, west_vect), np.pi / 2)
        self.assertEqual(dfe.angle_between(south_vect, west_vect), np.pi / 2)
        self.assertEqual(dfe.angle_between(east_vect, west_vect), np.pi)
        self.assertEqual(dfe.angle_between(west_vect, east_vect), np.pi)
        self.assertLess(
            dfe.angle_between(slightly_above_west, east_vect), np.pi
        )

    def test_total_angle_between(self):
        """
        Tests total_angle_between function.
        """
        ones_vect = np.ones(2)
        north_vect = np.array([0, 1])
        south_vect = np.array([0, -1])
        east_vect = np.array([1, 0])
        west_vect = np.array([-1, 0])
        slightly_above_west = np.array([-1, 0.1])

        self.assertEqual(dfe.total_angle_between(ones_vect, ones_vect), 0.0)
        self.assertEqual(dfe.total_angle_between(east_vect, east_vect), 0.0)
        self.assertEqual(dfe.total_angle_between(west_vect, west_vect), 0.0)
        self.assertEqual(
            dfe.total_angle_between(ones_vect, north_vect), np.pi / 4
        )
        self.assertEqual(
            dfe.total_angle_between(north_vect, ones_vect), -np.pi / 4
        )
        self.assertEqual(
            dfe.total_angle_between(north_vect, west_vect), np.pi / 2
        )
        self.assertEqual(
            dfe.total_angle_between(south_vect, west_vect), -np.pi / 2
        )
        self.assertEqual(dfe.total_angle_between(east_vect, west_vect), np.pi)
        self.assertLess(
            dfe.angle_between(slightly_above_west, east_vect), np.pi
        )

    def test_dist_2d(self):
        """
        Tests dist_2d function.
        """
        p1 = np.array([0, 0])
        p2 = np.array([1, 1])
        p3 = np.array([-1, -1])
        p4 = np.array([1, 0])
        p5 = np.array([1000, 0])
        self.assertEqual(dfe.dist_2d(p1, p2), np.sqrt(2))
        self.assertEqual(dfe.dist_2d(p2, p3), 2 * np.sqrt(2))
        self.assertEqual(dfe.dist_2d(p4, p1), 1.0)
        self.assertEqual(dfe.dist_2d(p1, p4), 1.0)
        self.assertEqual(dfe.dist_2d(p5, p4), 999.0)

    def test_norm_2d(self):
        """
        Tests norm_2d function.
        """
        self.assertEqual(dfe.norm_2d(np.array([0, 0])), 0.0)
        self.assertEqual(dfe.norm_2d(np.array([1, 0])), 1.0)
        self.assertEqual(dfe.norm_2d([-1, -1]), np.sqrt(2))


class FeatureFunctionTests(unittest.TestCase):
    """
    Tests feature extracting functions.
    """

    def test_radial_density(self):
        """
        Tests radial_density_features function.
        """
        agent_pos = (10, 10)
        pedestrian_pos = [
            (0.0, 0.0),
            (100.0, 100.0),
            (1e6, 1e6),
            (5.0, 5.0),
            (3.0, 3.0),
        ]

        nptest.assert_equal(
            dfe.radial_density_features(agent_pos, pedestrian_pos, 1e7),
            np.array([0.0, 0.0, 1.0]),
        )
        nptest.assert_equal(
            dfe.radial_density_features(agent_pos, pedestrian_pos, 0),
            np.array([1.0, 0.0, 0.0]),
        )
        nptest.assert_equal(
            dfe.radial_density_features(agent_pos, pedestrian_pos, 10),
            np.array([0.0, 1.0, 0.0]),
        )

    @given(
        npthesis.arrays(np.float, (2,)),
        npthesis.arrays(
            np.float,
            st.tuples(st.integers(min_value=0, max_value=10000), st.just(2)),
        ),
        st.floats(),
    )
    def test_onehot_radial_density(self, agent_pos, ped_positions, radius):
        """
        Tests if drone_feature_extractor.radial_density_features is one-hot
        encoded or not.
        """
        # import pdb

        # pdb.set_trace()
        radial_feature = dfe.radial_density_features(
            agent_pos, ped_positions, radius
        )
        self.assertTrue(is_onehot(radial_feature))
