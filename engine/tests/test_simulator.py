from decimal import Decimal
import numpy as np
from engine.physics.ball_state import BallState, MotionState
from engine.physics.simulation_state import SimulationState
from engine.physics.simulator import advance_state, simulate
from engine.physics.motion_models import sliding_motion, rolling_motion, cue_strike
from engine.physics.tuneable_constants import BALL_RADIUS, G, STANDARD_9_FOOT

THREE_PLACES = Decimal('0.000')


# ── advance_state: base cases ──

def test_advance_state_single_sliding_ball():
    ball = BallState(pos=[1.0, 0.7], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    state = SimulationState(balls=[ball], time=0.0)
    dt = 0.1
    # Expected from sliding_motion
    exp_pos, exp_vel, exp_omega = sliding_motion(
        BallState(pos=[1.0, 0.7], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING), dt, G
    )
    advance_state(state, dt)
    np.testing.assert_array_almost_equal(ball.pos, exp_pos)
    np.testing.assert_array_almost_equal(ball.vel, exp_vel)
    assert abs(ball.omega - exp_omega) < 1e-9

def test_advance_state_single_rolling_ball():
    ball = BallState(pos=[1.0, 0.7], vel=[2.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    state = SimulationState(balls=[ball], time=0.0)
    dt = 0.1
    exp_pos, exp_vel = rolling_motion(
        BallState(pos=[1.0, 0.7], vel=[2.0, 0.0], omega=0.0, motion=MotionState.ROLLING), dt, G
    )
    advance_state(state, dt)
    np.testing.assert_array_almost_equal(ball.pos, exp_pos)
    np.testing.assert_array_almost_equal(ball.vel, exp_vel)

def test_advance_state_mixed_balls():
    sliding = BallState(pos=[0.5, 0.5], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    rolling = BallState(pos=[1.5, 0.5], vel=[1.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    state = SimulationState(balls=[sliding, rolling], time=0.0)
    dt = 0.05
    exp_s_pos, exp_s_vel, exp_s_omega = sliding_motion(
        BallState(pos=[0.5, 0.5], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING), dt, G
    )
    exp_r_pos, exp_r_vel = rolling_motion(
        BallState(pos=[1.5, 0.5], vel=[1.0, 0.0], omega=0.0, motion=MotionState.ROLLING), dt, G
    )
    advance_state(state, dt)
    np.testing.assert_array_almost_equal(sliding.pos, exp_s_pos)
    np.testing.assert_array_almost_equal(sliding.vel, exp_s_vel)
    np.testing.assert_array_almost_equal(rolling.pos, exp_r_pos)
    np.testing.assert_array_almost_equal(rolling.vel, exp_r_vel)


# ── advance_state: edge cases ──

def test_advance_state_stopped_ball_unchanged():
    ball = BallState(pos=[1.0, 0.7], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    state = SimulationState(balls=[ball], time=0.0)
    advance_state(state, 0.1)
    np.testing.assert_array_equal(ball.pos, [1.0, 0.7])
    np.testing.assert_array_equal(ball.vel, [0.0, 0.0])

def test_advance_state_updates_time():
    ball = BallState(pos=[1.0, 0.7], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    state = SimulationState(balls=[ball], time=1.5)
    advance_state(state, 0.3)
    assert Decimal(str(state.time)).quantize(THREE_PLACES) == Decimal("1.800")


# ── simulate: base cases ──

def test_simulate_single_ball_slides_rolls_stops():
    # Ball struck gently from left side — should slide, roll, then stop without hitting a rail
    ball = cue_strike(position=[0.5, 0.71], direction=[1, 0], speed=1.0)
    state = SimulationState(balls=[ball], time=0.0)
    simulate(state, STANDARD_9_FOOT)
    # Must end stopped
    assert ball.motion == MotionState.STOPPED
    assert np.allclose(ball.vel, [0.0, 0.0])
    # Ball should have moved to the right from starting position
    assert ball.pos[0] > 0.5
    # Ball shouldn't have reached the right wall (speed=1 is gentle)
    assert ball.pos[0] < STANDARD_9_FOOT.width - BALL_RADIUS

def test_simulate_ball_hits_rail_and_stops():
    # Ball rolling fast toward right wall, close to it
    ball = BallState(pos=[2.5, 0.71], vel=[3.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    state = SimulationState(balls=[ball], time=0.0)
    simulate(state, STANDARD_9_FOOT)
    assert ball.motion == MotionState.STOPPED
    assert np.allclose(ball.vel, [0.0, 0.0])


# ── simulate: edge cases ──

def test_simulate_all_stopped_immediately():
    a = BallState(pos=[0.5, 0.5], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    b = BallState(pos=[1.5, 0.5], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    state = SimulationState(balls=[a, b], time=0.0)
    simulate(state, STANDARD_9_FOOT)
    assert state.time == 0.0
    np.testing.assert_array_equal(a.pos, [0.5, 0.5])
    np.testing.assert_array_equal(b.pos, [1.5, 0.5])
