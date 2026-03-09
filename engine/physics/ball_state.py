from enum import Enum
from dataclasses import dataclass
import numpy as np

from engine.physics.tuneable_constants import BALL_MASS, BALL_RADIUS, MU_ROLL, MU_SLIDE, SPIN_FRICTION

class MotionState(Enum):
    SLIDING = 1
    ROLLING = 2
    SPINNING = 3
    STOPPED = 4


class BallState:
    radius = BALL_RADIUS
    mass = BALL_MASS

    def __init__(self, pos, vel, omega, motion):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.omega = float(omega)
        self.motion = motion

    def mu(self):
        if self.motion == MotionState.SLIDING:
            return MU_SLIDE
        elif self.motion == MotionState.ROLLING:
            return MU_ROLL
        elif self.motion == MotionState.SPINNING:
            return SPIN_FRICTION
        else:
            return None
