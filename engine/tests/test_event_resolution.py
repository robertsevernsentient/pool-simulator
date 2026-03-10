from decimal import Decimal
import numpy as np
from engine.physics.ball_state import BallState, MotionState
from engine.physics.event_prediction import Event
from engine.physics.event_resolution import resolve_ball_collision, resolve_rail_collision, resolve_event
from engine.physics.simulation_state import SimulationState
from engine.physics.tuneable_constants import BALL_RADIUS, RAIL_RESTITUTION, STANDARD_9_FOOT

THREE_PLACES = Decimal('0.000')


# ── resolve_ball_collision: base cases ──

def test_resolve_ball_collision_head_on_equal_speed():
    # Two equal-mass balls head-on along x-axis: velocities swap
    # A at [0,0] vel=[2,0], B at [0.05715,0] vel=[-2,0]  (touching, 2r apart)
    a = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.05715, 0.0], vel=[-2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    resolve_ball_collision(a, b)
    assert Decimal(str(a.vel[0])).quantize(THREE_PLACES) == Decimal("-2.000")
    assert Decimal(str(a.vel[1])).quantize(THREE_PLACES) == Decimal("0.000")
    assert Decimal(str(b.vel[0])).quantize(THREE_PLACES) == Decimal("2.000")
    assert Decimal(str(b.vel[1])).quantize(THREE_PLACES) == Decimal("0.000")

def test_resolve_ball_collision_moving_hits_stationary():
    # Equal mass: velocities swap along collision normal
    # A vel=[3,0] hits stationary B on x-axis
    a = BallState(pos=[0.0, 0.0], vel=[3.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.05715, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    resolve_ball_collision(a, b)
    assert Decimal(str(a.vel[0])).quantize(THREE_PLACES) == Decimal("0.000")
    assert Decimal(str(a.vel[1])).quantize(THREE_PLACES) == Decimal("0.000")
    assert Decimal(str(b.vel[0])).quantize(THREE_PLACES) == Decimal("3.000")
    assert Decimal(str(b.vel[1])).quantize(THREE_PLACES) == Decimal("0.000")

def test_resolve_ball_collision_topspin_re_accelerates_cue_ball():
    # Head-on collision, A has forward spin (omega > 0).
    # After collision A.vel = [0,0] but omega is preserved and motion is SLIDING.
    # Since slip = v - Rω = 0 - Rω < 0 (topspin), friction will accelerate A forward.
    omega_initial = 50.0  # forward spin
    a = BallState(pos=[0.0, 0.0], vel=[3.0, 0.0], omega=omega_initial, motion=MotionState.SLIDING)
    b = BallState(pos=[0.05715, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    resolve_ball_collision(a, b)
    # A's velocity is zero after collision (equal mass, head-on)
    assert Decimal(str(a.vel[0])).quantize(THREE_PLACES) == Decimal("0.000")
    # But omega is preserved — collision doesn't affect spin
    assert a.omega == omega_initial
    # And A is set to SLIDING, so friction will act on the slip (v=0, Rω>0 → topspin re-acceleration)
    assert a.motion == MotionState.SLIDING

def test_resolve_ball_collision_stun_shot_follows_tangent_line():
    # A perfect stun shot (omega=0) at an angle: cue ball deflects along the tangent.
    # 45° cut: A at origin vel=[3,0], B at [r√2, r√2] (touching, 45° angle).
    # Collision normal n = [-1/√2, -1/√2].
    # After collision, A retains only the tangential component:
    #   A.vel_after = [3,0] - (3/√2)*[-1/√2,-1/√2] inverted = [1.5, -1.5]
    # This is perpendicular to n: dot([1.5,-1.5], [-1/√2,-1/√2]) = 0
    r = BALL_RADIUS
    s2 = np.sqrt(2)
    a = BallState(pos=[0.0, 0.0], vel=[3.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[r * s2, r * s2], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    resolve_ball_collision(a, b)
    # A follows the tangent: velocity is perpendicular to the collision normal
    n = a.pos - b.pos
    n = n / np.linalg.norm(n)
    assert abs(np.dot(a.vel, n)) < 1e-10
    # Verify actual values: A.vel_after = [1.5, -1.5]
    assert Decimal(str(a.vel[0])).quantize(THREE_PLACES) == Decimal("1.500")
    assert Decimal(str(a.vel[1])).quantize(THREE_PLACES) == Decimal("-1.500")


# ── resolve_ball_collision: edge cases ──

def test_resolve_ball_collision_separating_no_change():
    # Balls moving apart — vel_norm > 0, function returns early
    a = BallState(pos=[0.0, 0.0], vel=[-1.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.05715, 0.0], vel=[1.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    vel_a_before = a.vel.copy()
    vel_b_before = b.vel.copy()
    resolve_ball_collision(a, b)
    assert np.array_equal(a.vel, vel_a_before)
    assert np.array_equal(b.vel, vel_b_before)

def test_resolve_ball_collision_sets_both_sliding():
    a = BallState(pos=[0.0, 0.0], vel=[3.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    b = BallState(pos=[0.05715, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    resolve_ball_collision(a, b)
    assert a.motion == MotionState.SLIDING
    assert b.motion == MotionState.SLIDING


# ── resolve_ball_collision: self-consistency ──

def test_resolve_ball_collision_momentum_conserved():
    a = BallState(pos=[0.0, 0.0], vel=[3.0, 1.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.05715, 0.0], vel=[-1.0, 0.5], omega=0.0, motion=MotionState.SLIDING)
    p_before = a.mass * a.vel + b.mass * b.vel
    resolve_ball_collision(a, b)
    p_after = a.mass * a.vel + b.mass * b.vel
    assert Decimal(str(p_after[0])).quantize(THREE_PLACES) == Decimal(str(p_before[0])).quantize(THREE_PLACES)
    assert Decimal(str(p_after[1])).quantize(THREE_PLACES) == Decimal(str(p_before[1])).quantize(THREE_PLACES)


# ── resolve_rail_collision: base cases ──

def test_resolve_rail_collision_perpendicular():
    # Ball hitting right wall: vel=[3,0], normal=[-1,0], e=0.9
    # v_n = dot([3,0],[-1,0])*[-1,0] = -3*[-1,0] = [3,0]
    # v_t = [3,0] - [3,0] = [0,0]
    # new_vel = [0,0] - 0.9*[3,0] = [-2.7, 0]
    ball = BallState(pos=[2.81, 0.7], vel=[3.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    resolve_rail_collision(ball, np.array([-1, 0], dtype=float), RAIL_RESTITUTION)
    assert Decimal(str(ball.vel[0])).quantize(THREE_PLACES) == Decimal("-2.700")
    assert Decimal(str(ball.vel[1])).quantize(THREE_PLACES) == Decimal("0.000")

def test_resolve_rail_collision_angled():
    # Ball hitting top wall: vel=[2,2], normal=[0,-1], e=0.9
    # v_n = dot([2,2],[0,-1])*[0,-1] = -2*[0,-1] = [0,2]
    # v_t = [2,2] - [0,2] = [2,0]
    # new_vel = [2,0] - 0.9*[0,2] = [2, -1.8]
    ball = BallState(pos=[0.5, 1.39], vel=[2.0, 2.0], omega=0.0, motion=MotionState.SLIDING)
    resolve_rail_collision(ball, np.array([0, -1], dtype=float), RAIL_RESTITUTION)
    assert Decimal(str(ball.vel[0])).quantize(THREE_PLACES) == Decimal("2.000")
    assert Decimal(str(ball.vel[1])).quantize(THREE_PLACES) == Decimal("-1.800")


# ── resolve_rail_collision: edge cases ──

def test_resolve_rail_collision_sets_sliding():
    ball = BallState(pos=[2.81, 0.7], vel=[3.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    resolve_rail_collision(ball, np.array([-1, 0], dtype=float), RAIL_RESTITUTION)
    assert ball.motion == MotionState.SLIDING


# ── resolve_event: base cases ──

def test_resolve_event_state_change_sliding_to_rolling():
    ball = BallState(pos=[1.0, 0.7], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    state = SimulationState(balls=[ball], time=0.0)
    event = Event(time=0.1, event_type="STATE_CHANGE", a=0, b=None)
    resolve_event(state, event, STANDARD_9_FOOT)
    assert ball.motion == MotionState.ROLLING

def test_resolve_event_state_change_rolling_to_stopped():
    ball = BallState(pos=[1.0, 0.7], vel=[0.5, 0.0], omega=0.0, motion=MotionState.ROLLING)
    state = SimulationState(balls=[ball], time=0.0)
    event = Event(time=0.1, event_type="STATE_CHANGE", a=0, b=None)
    resolve_event(state, event, STANDARD_9_FOOT)
    assert ball.motion == MotionState.STOPPED
    assert ball.vel[0] == 0.0
    assert ball.vel[1] == 0.0

def test_resolve_event_rail_collision_right_wall():
    # Ball at right wall edge, moving right
    ball = BallState(pos=[STANDARD_9_FOOT.width - BALL_RADIUS, 0.7], vel=[3.0, 0.0], omega=0.0, motion=MotionState.ROLLING)
    state = SimulationState(balls=[ball], time=0.0)
    event = Event(time=0.1, event_type="RAIL_COLLISION", a=0, b=None)
    resolve_event(state, event, STANDARD_9_FOOT)
    # Normal is [-1,0], so vel reverses x with restitution
    assert Decimal(str(ball.vel[0])).quantize(THREE_PLACES) == Decimal("-2.700")
    assert Decimal(str(ball.vel[1])).quantize(THREE_PLACES) == Decimal("0.000")
    assert ball.motion == MotionState.SLIDING

def test_resolve_event_ball_collision():
    a = BallState(pos=[0.0, 0.0], vel=[3.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[2 * BALL_RADIUS, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    state = SimulationState(balls=[a, b], time=0.0)
    event = Event(time=0.1, event_type="BALL_COLLISION", a=0, b=1)
    resolve_event(state, event, STANDARD_9_FOOT)
    # Equal mass head-on: velocities swap
    assert Decimal(str(a.vel[0])).quantize(THREE_PLACES) == Decimal("0.000")
    assert Decimal(str(b.vel[0])).quantize(THREE_PLACES) == Decimal("3.000")


# ── resolve_event: edge cases (normal detection) ──

def test_resolve_event_rail_collision_left_wall():
    ball = BallState(pos=[BALL_RADIUS, 0.7], vel=[-3.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    state = SimulationState(balls=[ball], time=0.0)
    event = Event(time=0.1, event_type="RAIL_COLLISION", a=0, b=None)
    resolve_event(state, event, STANDARD_9_FOOT)
    # Normal is [1,0]
    assert Decimal(str(ball.vel[0])).quantize(THREE_PLACES) == Decimal("2.700")
    assert Decimal(str(ball.vel[1])).quantize(THREE_PLACES) == Decimal("0.000")

def test_resolve_event_rail_collision_bottom_wall():
    ball = BallState(pos=[1.0, BALL_RADIUS], vel=[0.0, -3.0], omega=0.0, motion=MotionState.SLIDING)
    state = SimulationState(balls=[ball], time=0.0)
    event = Event(time=0.1, event_type="RAIL_COLLISION", a=0, b=None)
    resolve_event(state, event, STANDARD_9_FOOT)
    # Normal is [0,1]
    assert Decimal(str(ball.vel[0])).quantize(THREE_PLACES) == Decimal("0.000")
    assert Decimal(str(ball.vel[1])).quantize(THREE_PLACES) == Decimal("2.700")
