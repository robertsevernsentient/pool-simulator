import unittest
from pool_simulator.physics.dynamics import calculate_velocity, update_position

class TestDynamics(unittest.TestCase):

    def test_calculate_velocity(self):
        # Test case for calculating velocity
        initial_velocity = (5, 0)  # 5 units in x direction, 0 in y
        time = 2  # seconds
        expected_velocity = (5, 0)  # should remain the same in this case
        result = calculate_velocity(initial_velocity, time)
        self.assertEqual(result, expected_velocity)

    def test_update_position(self):
        # Test case for updating position
        initial_position = (0, 0)
        velocity = (5, 3)  # 5 units in x direction, 3 in y
        time = 2  # seconds
        expected_position = (10, 6)  # new position after 2 seconds
        result = update_position(initial_position, velocity, time)
        self.assertEqual(result, expected_position)

if __name__ == '__main__':
    unittest.main()