from src.physics.ball_state import BallState, MotionState
from src.physics.motion_models import cue_strike, time_sliding_to_rolling
from src.physics.simulation_state import SimulationState
from src.physics.tuneable_constants import BALL_MASS, BALL_RADIUS, G, MU_SLIDE, STANDARD_9_FOOT
import numpy as np

table = STANDARD_9_FOOT

cue = cue_strike(
    position=[0.5,0.7],
    direction=[1,0],
    speed=2.0
)

object_ball = BallState(
    pos=np.array([1.0,0.7]),
    vel=np.zeros(2),
    omega=0,
    motion=MotionState.STOPPED,
    radius=BALL_RADIUS,
    mass=BALL_MASS
)

state = SimulationState(
    balls=[cue, object_ball],
    time=0
)


def test_initial_state():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=2.0
    )
    assert cue.pos[0] == 0.5
    assert cue.pos[1] == 0.7
    assert cue.vel[0] == 2.0
    assert cue.vel[1] == 0.0
    assert cue.motion == MotionState.SLIDING

def test_time_sliding_to_rolling():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=2.0
    )

    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    assert time_to_rolling > 1
    assert time_to_rolling < 1.5

    # less friction
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE/2, G)
    assert time_to_rolling > 2
    assert time_to_rolling < 2.5

    # softer hit
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=1.0
    )
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    assert time_to_rolling > 0.5
    assert time_to_rolling < 1.0

    # harder hit
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=3.0
    )
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    assert time_to_rolling > 1.5
    assert time_to_rolling < 2.0