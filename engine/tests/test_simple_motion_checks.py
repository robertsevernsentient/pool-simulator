from decimal import Decimal
from engine.physics.ball_state import BallState, MotionState
from engine.physics.motion_models import cue_strike, rolling_motion, sliding_motion, time_rolling_to_stop, time_sliding_to_rolling
from engine.physics.simulation_state import SimulationState
from engine.physics.tuneable_constants import G

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
    _, vel, _ = sliding_motion(cue, t, G)
    expected_speed = 5.0 / 7.0 * v0  # = 10/7 ≈ 1.429
    assert Decimal(str(np.linalg.norm(vel))).quantize(THREE_PLACES) == Decimal(str(expected_speed)).quantize(THREE_PLACES)


def test_state_after_sliding():
    time_to_rolling = time_sliding_to_rolling(DEFAULT_CUE_BALL(), G)
    pos, vel, _ = sliding_motion(DEFAULT_CUE_BALL(), time_to_rolling, G)

    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.999")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.700")

    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("1.429")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("0.000")


# ── sliding_motion: base cases ──

def test_sliding_motion_x_axis():
    # ball at [0,0], vel=[2,0], t=0.1
    # a = -mu_slide * g = -0.20 * 9.81 = -1.962
    # vel = 2.0 + (-1.962)(0.1) = 1.804
    # pos = 0 + 2.0(0.1) + 0.5(-1.962)(0.01) = 0.200 - 0.010 = 0.190
    cue = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    pos, vel, _ = sliding_motion(cue, 0.1, G)
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.190")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.000")
    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("1.804")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("0.000")

def test_sliding_motion_angled():
    # ball at [0,0], vel=[1,1] (speed √2), t=0.1
    # direction = [1/√2, 1/√2], a = -1.962 * [1/√2, 1/√2] = [-1.387, -1.387]
    # vel = [1 - 0.139, 1 - 0.139] = [0.861, 0.861]
    # pos = [0.1 - 0.007, 0.1 - 0.007] = [0.093, 0.093]
    cue = BallState(pos=[0.0, 0.0], vel=[1.0, 1.0], omega=0.0, motion=MotionState.SLIDING)
    pos, vel, _ = sliding_motion(cue, 0.1, G)
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.093")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.093")
    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("0.861")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("0.861")


# ── sliding_motion: edge cases ──

def test_sliding_motion_zero_velocity_returns_unchanged():
    cue = BallState(pos=[1.0, 2.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    pos, vel, _ = sliding_motion(cue, 0.5, G)
    assert pos[0] == 1.0
    assert pos[1] == 2.0
    assert vel[0] == 0.0
    assert vel[1] == 0.0

def test_sliding_motion_t_zero():
    cue = BallState(pos=[1.0, 2.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    pos, vel, _ = sliding_motion(cue, 0.0, G)
    assert pos[0] == 1.0
    assert pos[1] == 2.0
    assert vel[0] == 2.0
    assert vel[1] == 0.0


# ── sliding_motion: self-consistency ──

def test_sliding_motion_decelerates_monotonically():
    cue = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    _, vel1, _ = sliding_motion(cue, 0.1, G)
    cue2 = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    _, vel2, _ = sliding_motion(cue2, 0.2, G)
    assert np.linalg.norm(vel1) > np.linalg.norm(vel2)


###### ROLLING TO STOP
def test_rolling_to_stop_time_standard():
    cue = DEFAULT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel, _ = sliding_motion(cue, time_to_rolling, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("14.562")

def test_rolling_to_stop_time_soft():
    cue = SOFT_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel, _ = sliding_motion(cue, time_to_rolling, G)

    cue.pos = pos
    cue.vel = vel
    cue.motion = MotionState.ROLLING

    time_to_stop = time_rolling_to_stop(cue, G)
    assert Decimal(str(time_to_stop)).quantize(THREE_PLACES) == Decimal("7.281")

def test_rolling_to_stop_time_hard():
    cue = HARD_CUE_BALL()
    time_to_rolling = time_sliding_to_rolling(cue, G)
    pos, vel, _ = sliding_motion(cue, time_to_rolling, G)

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
    pos, vel, _ = sliding_motion(cue, time_to_rolling, G)

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
    pos, vel, _ = sliding_motion(cue, time_to_rolling, G)

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


# ── rolling_motion: base cases ──

def test_rolling_motion_x_axis():
    # ball at [0,0], vel=[2,0], t=0.5
    # a = -mu_roll * g = -0.01 * 9.81 = -0.0981
    # vel = 2.0 - 0.0981*0.5 = 1.951
    # pos = 0 + 2.0*0.5 + 0.5*(-0.0981)*0.25 = 1.000 - 0.012 = 0.988
    cue = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    pos, vel = rolling_motion(cue, 0.5, G)
    assert Decimal(str(pos[0])).quantize(THREE_PLACES) == Decimal("0.988")
    assert Decimal(str(pos[1])).quantize(THREE_PLACES) == Decimal("0.000")
    assert Decimal(str(vel[0])).quantize(THREE_PLACES) == Decimal("1.951")
    assert Decimal(str(vel[1])).quantize(THREE_PLACES) == Decimal("0.000")


# ── rolling_motion: edge cases ──

def test_rolling_motion_zero_velocity_returns_unchanged():
    cue = BallState(pos=[1.0, 2.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    pos, vel = rolling_motion(cue, 0.5, G)
    assert pos[0] == 1.0
    assert pos[1] == 2.0
    assert vel[0] == 0.0
    assert vel[1] == 0.0

def test_rolling_motion_t_zero():
    cue = BallState(pos=[1.0, 2.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    pos, vel = rolling_motion(cue, 0.0, G)
    assert pos[0] == 1.0
    assert pos[1] == 2.0
    assert vel[0] == 2.0
    assert vel[1] == 0.0



# ── time_sliding_to_rolling: draw/topspin ──

def test_time_sliding_to_rolling_stun_shot():
    # omega=0 (stun): same as original formula t = 2*v0/(7*mu*g)
    # v0=2, mu=0.20, g=9.81 → t = 4/13.734 = 0.291
    cue = cue_strike(position=[0.5, 0.7], direction=[1, 0], speed=2.0)
    t = time_sliding_to_rolling(cue, G)
    assert Decimal(str(t)).quantize(THREE_PLACES) == Decimal("0.291")

def test_time_sliding_to_rolling_draw():
    # Draw shot: omega=-20 (backspin), v0=2
    # slip = |v0 - R*omega| = |2 - 0.028575*(-20)| = |2 + 0.5715| = 2.5715
    # t = 2*2.5715 / (7*0.20*9.81) = 5.143/13.734 = 0.375
    cue = BallState(pos=[0.5, 0.7], vel=[2.0, 0.0], omega=-20.0, motion=MotionState.SLIDING)
    t = time_sliding_to_rolling(cue, G)
    assert Decimal(str(t)).quantize(THREE_PLACES) == Decimal("0.374")

def test_time_sliding_to_rolling_topspin():
    # Topspin: omega=+100, v0=2, R*omega=2.8575
    # slip = |2 - 2.8575| = 0.8575
    # t = 2*0.8575 / 13.734 = 0.125
    cue = BallState(pos=[0.5, 0.7], vel=[2.0, 0.0], omega=100.0, motion=MotionState.SLIDING)
    t = time_sliding_to_rolling(cue, G)
    assert Decimal(str(t)).quantize(THREE_PLACES) == Decimal("0.125")

def test_sliding_to_rolling_self_consistency_v_equals_r_omega():
    # At transition time, v should equal R*omega (rolling condition)
    cue = BallState(pos=[0.5, 0.7], vel=[2.0, 0.0], omega=-20.0, motion=MotionState.SLIDING)
    t = time_sliding_to_rolling(cue, G)
    _, vel, omega = sliding_motion(cue, t, G)
    speed = np.linalg.norm(vel)
    assert Decimal(str(speed)).quantize(THREE_PLACES) == Decimal(str(cue.radius * omega)).quantize(THREE_PLACES)


# ── sliding_motion: draw/topspin ──

def test_sliding_motion_backspin_omega_increases():
    # Backspin: omega=0 (or negative), slip > 0, so omega should increase
    cue = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    _, _, omega = sliding_motion(cue, 0.1, G)
    assert omega > 0.0

def test_sliding_motion_topspin_omega_decreases():
    # Topspin: R*omega > v, slip < 0, so omega should decrease
    cue = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=100.0, motion=MotionState.SLIDING)
    _, _, omega = sliding_motion(cue, 0.1, G)
    assert omega < 100.0

def test_sliding_motion_topspin_self_consistency_v_equals_r_omega():
    # Topspin case: at transition, v = R*omega
    cue = BallState(pos=[0.5, 0.7], vel=[2.0, 0.0], omega=100.0, motion=MotionState.SLIDING)
    t = time_sliding_to_rolling(cue, G)
    _, vel, omega = sliding_motion(cue, t, G)
    speed = np.linalg.norm(vel)
    assert Decimal(str(speed)).quantize(THREE_PLACES) == Decimal(str(cue.radius * omega)).quantize(THREE_PLACES)