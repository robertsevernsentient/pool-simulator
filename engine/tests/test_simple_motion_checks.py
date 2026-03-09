from decimal import Decimal
from engine.physics.ball_state import BallState, MotionState
from engine.physics.motion_models import cue_strike, rolling_motion, sliding_motion, spinning_motion, time_rolling_to_stop, time_sliding_to_rolling, time_spin_to_stop
from engine.physics.simulation_state import SimulationState
from engine.physics.tuneable_constants import G, SPIN_FRICTION

import numpy as np
import pytest

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


# ── cue_strike: base cases ──

def test_cue_strike_angled_direction():
    # direction [1,1] normalised = [1/√2, 1/√2], vel = 2/√2 ≈ 1.414
    cue = cue_strike(position=[0.5, 0.7], direction=[1, 1], speed=2.0)
    assert Decimal(str(cue.vel[0])).quantize(THREE_PLACES) == Decimal("1.414")
    assert Decimal(str(cue.vel[1])).quantize(THREE_PLACES) == Decimal("1.414")
    assert cue.motion == MotionState.SLIDING

def test_cue_strike_negative_direction():
    cue = cue_strike(position=[0.5, 0.7], direction=[-1, 0], speed=3.0)
    assert cue.vel[0] == -3.0
    assert cue.vel[1] == 0.0
    assert cue.motion == MotionState.SLIDING


# ── cue_strike: edge cases ──

def test_cue_strike_zero_speed():
    cue = cue_strike(position=[0.5, 0.7], direction=[1, 0], speed=0.0)
    assert cue.vel[0] == 0.0
    assert cue.vel[1] == 0.0
    assert cue.motion == MotionState.STOPPED

def test_cue_strike_zero_direction_raises():
    with pytest.raises(ValueError):
        cue_strike(position=[0.5, 0.7], direction=[0, 0], speed=2.0)

def test_cue_strike_unnormalized_direction():
    # direction [3,4] norm=5, unit=[0.6,0.8], vel = 5*[0.6,0.8] = [3.0, 4.0]
    cue = cue_strike(position=[1.0, 1.0], direction=[3, 4], speed=5.0)
    assert Decimal(str(cue.vel[0])).quantize(THREE_PLACES) == Decimal("3.000")
    assert Decimal(str(cue.vel[1])).quantize(THREE_PLACES) == Decimal("4.000")


# ── cue_strike: self-consistency ──

def test_cue_strike_speed_matches_input():
    cue = cue_strike(position=[0.0, 0.0], direction=[3, 4], speed=5.0)
    actual_speed = np.linalg.norm(cue.vel)
    assert Decimal(str(actual_speed)).quantize(THREE_PLACES) == Decimal("5.000")

def test_cue_strike_direction_preserved():
    cue = cue_strike(position=[0.0, 0.0], direction=[1, 2], speed=3.0)
    vel_dir = cue.vel / np.linalg.norm(cue.vel)
    expected_dir = np.array([1, 2], dtype=float) / np.linalg.norm([1, 2])
    assert Decimal(str(vel_dir[0])).quantize(THREE_PLACES) == Decimal(str(expected_dir[0])).quantize(THREE_PLACES)
    assert Decimal(str(vel_dir[1])).quantize(THREE_PLACES) == Decimal(str(expected_dir[1])).quantize(THREE_PLACES)

########## SLIDING TO ROLLING
def test_default_time_sliding_to_rolling():
    time_to_rolling = time_sliding_to_rolling(DEFAULT_CUE_BALL(), G)
    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.291")

def test_time_sliding_to_rolling_soft():
    time_to_rolling = time_sliding_to_rolling(SOFT_CUE_BALL(), G)
    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.146")

def test_time_sliding_to_rolling_hard():
    time_to_rolling = time_sliding_to_rolling(HARD_CUE_BALL(), G)
    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.437")


# ── time_sliding_to_rolling: edge cases ──

def test_time_sliding_to_rolling_zero_velocity_raises():
    cue = BallState(pos=[0.5, 0.7], vel=[0, 0], omega=0.0, motion=MotionState.SLIDING)
    with pytest.raises(ValueError):
        time_sliding_to_rolling(cue, G)

def test_time_sliding_to_rolling_very_slow():
    cue = cue_strike(position=[0.5, 0.7], direction=[1, 0], speed=0.001)
    t = time_sliding_to_rolling(cue, G)
    # t = 2*0.001 / (7*0.20*9.81) = 0.002/13.734 ≈ 0.000146
    assert t is not None
    assert t > 0


# ── time_sliding_to_rolling: self-consistency ──

def test_sliding_velocity_at_rolling_time_equals_5_7_v0():
    # At transition, v should equal 5/7 * v0
    cue = DEFAULT_CUE_BALL()
    v0 = np.linalg.norm(cue.vel)
    t = time_sliding_to_rolling(cue, G)
    _, vel = sliding_motion(cue, t, G)
    expected_speed = 5.0 / 7.0 * v0  # = 10/7 ≈ 1.429
    assert Decimal(str(np.linalg.norm(vel))).quantize(THREE_PLACES) == Decimal(str(expected_speed)).quantize(THREE_PLACES)


def test_state_after_sliding():
    time_to_rolling = time_sliding_to_rolling(DEFAULT_CUE_BALL(), G)
    pos, vel = sliding_motion(DEFAULT_CUE_BALL(), time_to_rolling, G)

    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.999")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")

    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("1.429")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("0.000")


###### ROLLING TO STOP
def test_rolling_to_stop_time_standard():
    cue = DEFAULT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel = sliding_motion(cue, time_to_rolling, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("14.562")

def test_rolling_to_stop_time_soft():
    cue = SOFT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel = sliding_motion(cue, time_to_rolling, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("7.281")

def test_rolling_to_stop_time_hard():
    cue = HARD_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel = sliding_motion(cue, time_to_rolling, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("21.844")

# ── time_rolling_to_stop: isolated base case ──

def test_rolling_to_stop_time_isolated():
    # v0=2.0, mu_roll=0.01, g=9.81 → t = 2.0/(0.01*9.81) = 20.387
    cue = BallState(pos=[0.5, 0.7], vel=[2.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    t = time_rolling_to_stop(cue, G)
    assert Decimal(str(t)).quantize(THREE_PLACES) == Decimal("20.387")


# ── time_rolling_to_stop: edge cases ──

def test_time_rolling_to_stop_zero_velocity_raises():
    cue = BallState(pos=[0.5, 0.7], vel=[0, 0], omega=0.0, motion=MotionState.ROLLING)
    with pytest.raises(ValueError):
        time_rolling_to_stop(cue, G)


# ── time_rolling_to_stop: self-consistency ──

def test_rolling_velocity_is_zero_at_stop_time():
    cue = BallState(pos=[0.5, 0.7], vel=[2.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    t = time_rolling_to_stop(cue, G)
    _, vel = rolling_motion(cue, t, G)
    assert Decimal(str(np.linalg.norm(vel))).quantize(THREE_PLACES) == Decimal("0.000")


def test_state_after_rolling_standard():
    cue = DEFAULT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel = sliding_motion(cue, time_to_rolling, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, G)
    pos, vel = rolling_motion(cue, time_to_stop, G)

    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("11.401")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")

    assert vel[0] == 0.0
    assert vel[1] == 0.0

def test_state_after_rolling_angled():
    cue = ANGLED_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel = sliding_motion(cue, time_to_rolling, G)

    assert Decimal(str(time_to_rolling)).quantize(THREE_PLACES) == Decimal("0.291")
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.853")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("1.053")
    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("1.010")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("1.010")

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, G)
    pos, vel = rolling_motion(cue, time_to_stop, G)

    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("8.208")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("8.408")

    assert vel[0] == 0.0
    assert vel[1] == 0.0


def test_time_spin_to_stop_no_spin():
    cue = DEFAULT_CUE_BALL()
    cue.motion = MotionState.SPINNING
    assert time_spin_to_stop(cue) == None

def test_time_spin_to_stop_with_spin():
    cue = DEFAULT_CUE_BALL()
    cue.omega = 10.0  # set spin
    cue.motion = MotionState.SPINNING
    time_to_spin_stop = time_spin_to_stop(cue)
    assert Decimal(str(time_to_spin_stop)).quantize(THREE_PLACES) == Decimal("1.250")

def test_state_after_spin_ended():
    cue = DEFAULT_CUE_BALL()
    cue.omega = 10.0  # set spin
    cue.motion = MotionState.SPINNING
    time_to_spin_stop = time_spin_to_stop(cue)
    pos, omega = spinning_motion(cue, time_to_spin_stop, SPIN_FRICTION)
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.500")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")
    assert Decimal(str(omega)).quantize(THREE_PLACES) == Decimal("0.000")


def test_state_after_spin_ended():
    cue = DEFAULT_CUE_BALL()
    cue.omega = 10.0  # set spin
    cue.motion = MotionState.SPINNING
    time_to_spin_stop = time_spin_to_stop(cue)
    pos, omega = spinning_motion(cue, time_to_spin_stop/2, SPIN_FRICTION)
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.500")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")
    assert Decimal(str(omega)).quantize(THREE_PLACES) == Decimal("0.067")