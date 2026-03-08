import unittest
from pool_simulator.physics.collisions import check_collision, resolve_collision
from pool_simulator.geometry.ball import Ball

class TestCollisions(unittest.TestCase):

    def setUp(self):
        self.ball1 = Ball(position=(0, 0), velocity=(1, 0), radius=0.5)
        self.ball2 = Ball(position=(1, 0), velocity=(-1, 0), radius=0.5)

    def test_check_collision(self):
        self.assertTrue(check_collision(self.ball1, self.ball2))

    def test_resolve_collision(self):
        initial_velocity1 = self.ball1.velocity
        initial_velocity2 = self.ball2.velocity
        resolve_collision(self.ball1, self.ball2)
        self.assertNotEqual(self.ball1.velocity, initial_velocity1)
        self.assertNotEqual(self.ball2.velocity, initial_velocity2)

if __name__ == '__main__':
    unittest.main()