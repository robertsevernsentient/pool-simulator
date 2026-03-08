from decimal import Decimal
from engine.physics.ball_state import BallState, MotionState
from engine.physics.motion_models import cue_strike, rolling_motion, sliding_motion, spinning_motion, time_rolling_to_stop, time_sliding_to_rolling, time_spin_to_stop
from engine.physics.simulation_state import SimulationState
from engine.physics.tuneable_constants import BALL_MASS, BALL_RADIUS, G, MU_SLIDE, SPIN_FRICTION, STANDARD_9_FOOT
import numpy as np

THREE_PLACES = Decimal('0.000')

def DEFAULT_CUE_BALL():
    return cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=2.0
    )

def HARD_CUE_BALL():
    return cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=3.0
    )

def SOFT_CUE_BALL():
    return cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=1.0
    )

def ANGLED_CUE_BALL():
    return cue_strike(
        position=[0.5,0.7],
        direction=[1,1],
        speed=2.0
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

########## SLIDING TO ROLLING
def test_default_time_sliding_to_rolling():
    time_to_rolling = time_sliding_to_rolling(DEFAULT_CUE_BALL(), MU_SLIDE, G)
    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.291")

    # less friction
    time_to_rolling = time_sliding_to_rolling(DEFAULT_CUE_BALL(), MU_SLIDE/2, G)
    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.582")

def test_time_sliding_to_rolling_soft():
    time_to_rolling = time_sliding_to_rolling(SOFT_CUE_BALL(), MU_SLIDE, G)
    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.146")

def test_time_sliding_to_rolling_hard():
    time_to_rolling = time_sliding_to_rolling(HARD_CUE_BALL(), MU_SLIDE, G)
    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.437")


def test_state_after_sliding():
    time_to_rolling = time_sliding_to_rolling(DEFAULT_CUE_BALL(), MU_SLIDE, G)
    pos, vel = sliding_motion(DEFAULT_CUE_BALL(), time_to_rolling, MU_SLIDE, G)

    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.999")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")

    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("1.429")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("0.000")


###### ROLLING TO STOP
def test_rolling_to_stop_time_standard():
    cue = DEFAULT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    pos, vel = sliding_motion(cue, time_to_rolling, MU_SLIDE, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, MU_SLIDE, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("0.728")

def test_rolling_to_stop_time_soft():
    cue = SOFT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    pos, vel = sliding_motion(cue, time_to_rolling, MU_SLIDE, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, MU_SLIDE, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("0.364")

def test_rolling_to_stop_time_hard():
    cue = HARD_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    pos, vel = sliding_motion(cue, time_to_rolling, MU_SLIDE, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, MU_SLIDE, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("1.092")

def test_state_after_rolling_standard():
    cue = DEFAULT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    pos, vel = sliding_motion(cue, time_to_rolling, MU_SLIDE, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, MU_SLIDE, G)
    pos, vel = rolling_motion(cue, time_to_stop, MU_SLIDE, G)

    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("1.519")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")

    assert vel[0] == 0.0
    assert vel[1] == 0.0

def test_state_after_rolling_angled():
    cue = ANGLED_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, MU_SLIDE, G)
    pos, vel = sliding_motion(cue, time_to_rolling, MU_SLIDE, G)

    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.291")
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.853")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("1.053")
    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("1.010")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("1.010")

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, MU_SLIDE, G)
    pos, vel = rolling_motion(cue, time_to_stop, MU_SLIDE, G)

    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("1.221")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("1.421")

    assert vel[0] == 0.0
    assert vel[1] == 0.0


def test_time_spin_to_stop_no_spin():
    cue = DEFAULT_CUE_BALL()
    assert time_spin_to_stop(cue, SPIN_FRICTION) == None

def test_time_spin_to_stop_with_spin():
    cue = DEFAULT_CUE_BALL()
    cue.omega = 10.0  # set spin
    time_to_spin_stop = time_spin_to_stop(cue, SPIN_FRICTION)
    assert Decimal(str(time_to_spin_stop)).quantize(THREE_PLACES) == Decimal("1.250")

def test_state_after_spin_ended():
    cue = DEFAULT_CUE_BALL()
    cue.omega = 10.0  # set spin
    time_to_spin_stop = time_spin_to_stop(cue, SPIN_FRICTION)
    pos, omega = spinning_motion(cue, time_to_spin_stop, SPIN_FRICTION)
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.500")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")
    assert Decimal(str(omega)).quantize(THREE_PLACES) == Decimal("0.000")


def test_state_after_spin_ended():
    cue = DEFAULT_CUE_BALL()
    cue.omega = 10.0  # set spin
    time_to_spin_stop = time_spin_to_stop(cue, SPIN_FRICTION)
    pos, omega = spinning_motion(cue, time_to_spin_stop/2, SPIN_FRICTION)
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.500")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")
    assert Decimal(str(omega)).quantize(THREE_PLACES) == Decimal("0.067")