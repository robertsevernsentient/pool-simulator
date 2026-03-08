from enum import Enum
from dataclasses import dataclass
import numpy as np

class MotionState(Enum):
    SLIDING = 1
    ROLLING = 2
    SPINNING = 3
    STOPPED = 4


@dataclass
class BallState:
    pos: np.ndarray
    vel: np.ndarray
    omega: float      # angular velocity around vertical axis
    motion: MotionState
    radius: float
    mass: float