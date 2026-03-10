from decimal import Decimal
import numpy as np
from engine.physics.ball_state import BallState, MotionState
from engine.physics.event_prediction import _predict_rail_collision_position, predict_ball_ball_collision, predict_rail_collision, predict_state_transition
from engine.physics.motion_models import cue_strike, sliding_motion
from engine.physics.tuneable_constants import G, STANDARD_9_FOOT, BALL_RADIUS

THREE_PLACES = Decimal('0.000')

def test_predict_rail_collision_heading_right():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1,0],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert times == [STANDARD_9_FOOT.width - cue.radius, 0.7]

def test_predict_rail_collision_heading_left():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[-1,0],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert times == [cue.radius, 0.7]

def test_predict_rail_collision_heading_up():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[0,1],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert times == [0.5, STANDARD_9_FOOT.height - cue.radius]

def test_predict_rail_collision_heading_down():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[0,-1],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert times == [0.5, cue.radius]

def test_predict_rail_collision_none():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[0,0],
        speed=0.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert times is None

def test_predict_rail_collision_45_top():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1,1],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert Decimal(times[0]).quantize(THREE_PLACES) == Decimal(str("1.191"))
    assert Decimal(times[1]).quantize(THREE_PLACES) == Decimal(str("1.391")).quantize(THREE_PLACES)

    # We hi the top of the table
    assert Decimal(str(STANDARD_9_FOOT.height - cue.radius)).quantize(THREE_PLACES) == Decimal(str("1.391"))

def test_predict_rail_collision_45_right():
    cue = cue_strike(
        position=[2.5,0.7],
        direction=[1,1],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert Decimal(times[0]).quantize(THREE_PLACES) == Decimal(str("2.811")).quantize(THREE_PLACES)
    assert Decimal(times[1]).quantize(THREE_PLACES) == Decimal(str("1.011")).quantize(THREE_PLACES)

    # We hit the right of the table
    assert Decimal(str(STANDARD_9_FOOT.width - cue.radius)).quantize(THREE_PLACES) == Decimal(str("2.811"))

def test_predict_rail_collision_45_left():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[-1,-1],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert Decimal(times[0]).quantize(THREE_PLACES) == Decimal(str("0.029")).quantize(THREE_PLACES)
    assert Decimal(times[1]).quantize(THREE_PLACES) == Decimal(str("0.229")).quantize(THREE_PLACES)

    # We hit the left of the table
    assert Decimal(str(cue.radius)).quantize(THREE_PLACES) == Decimal(str("0.029"))

def test_predict_rail_collision_45_bottom():
    cue = cue_strike(
        position=[0.5,0.2],
        direction=[-1,-1],
        speed=2.0
    )

    times = _predict_rail_collision_position(cue, STANDARD_9_FOOT)
    assert Decimal(times[0]).quantize(THREE_PLACES) == Decimal(str("0.329")).quantize(THREE_PLACES)
    assert Decimal(times[1]).quantize(THREE_PLACES) == Decimal(str("0.029")).quantize(THREE_PLACES)

    # We hit the bottom of the table
    assert Decimal(str(cue.radius)).quantize(THREE_PLACES) == Decimal(str("0.029"))

# Test the times to hit the rail
def test_predict_rail_roll_before_collide():
    # This won't hit, as it will slide before hitting
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[0,1],
        speed=2.0
    )

    times = predict_rail_collision(cue, STANDARD_9_FOOT)
    assert times is None

def test_predict_rail_collision_top():
    # This one doesn't stop sliding
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[0,1],
        speed=3.0
    )

    times = predict_rail_collision(cue, STANDARD_9_FOOT)
    assert Decimal(times).quantize(THREE_PLACES) == Decimal(str("0.251"))

def test_predict_rail_collision_top_start_rolling():
    # This one starts rolling
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[0,1],
        speed=3.0
    )

    cue.motion = MotionState.ROLLING

    times = predict_rail_collision(cue, STANDARD_9_FOOT)
    assert Decimal(times).quantize(THREE_PLACES) == Decimal(str("0.231"))

def test_predict_rail_collision_top_rolling_slower():
    # This one doesn't starts sliding -- at the pace that previously didn't get to rail before rolling when it starts sliding
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[0,1],
        speed=2.0
    )

    cue.motion = MotionState.ROLLING

    times = predict_rail_collision(cue, STANDARD_9_FOOT)
    assert Decimal(times).quantize(THREE_PLACES) == Decimal(str("0.349"))

def test_predict_rail_collision_at_angle():
    # 45 degree angle to bottom left
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[-1,-1],
        speed=2.0
    )

    cue.motion = MotionState.ROLLING

    times = predict_rail_collision(cue, STANDARD_9_FOOT)
    assert Decimal(times).quantize(THREE_PLACES) == Decimal(str("0.336"))


# Test state change detection
def test_state_change_slide_to_roll_detection():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1,1],
        speed=2.0
    )
    slide_to_roll_time = predict_state_transition(cue)

    assert Decimal(slide_to_roll_time).quantize(THREE_PLACES) == Decimal(str("0.251"))

def test_state_change_slide_to_roll_detection():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1, 1],
        speed=2.0
    )
    cue.motion = MotionState.ROLLING
    slide_to_roll_time = predict_state_transition(cue)

    assert Decimal(slide_to_roll_time).quantize(THREE_PLACES) == Decimal(str("20.387"))

def test_state_change_when_stopped_detection():
    cue = BallState(
        pos=[0.5,0.7],
        vel=[0, 0],
        omega=0.0,
        motion=MotionState.STOPPED
    )

    time = predict_state_transition(cue)

    assert time is None


# ── predict_ball_ball_collision: base cases ──

def test_ball_ball_collision_moving_hits_stationary():
    # Ball A sliding fast toward stationary ball B
    # A at [0,0] vel=[5,0] SLIDING, B at [0.2,0] STOPPED
    # A decelerates: a = -0.20*9.81 = -1.962, p_a(t) = 5t - 0.981t²
    # B stationary: p_b = 0.2
    # Collision when 0.2 - (5t - 0.981t²) = 2r = 0.05715
    # → 0.981t² - 5t + 0.14285 = 0
    # → t = (5 - √(25 - 0.5604)) / 1.962 = (5 - 4.9437) / 1.962 ≈ 0.029
    a = BallState(pos=[0.0, 0.0], vel=[5.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.2, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    t = predict_ball_ball_collision(a, b, G)
    assert t is not None
    assert Decimal(str(t)).quantize(THREE_PLACES) == Decimal("0.029")


def test_ball_ball_collision_moving_away_no_collision():
    # Ball A moving left, away from ball B on the right
    a = BallState(pos=[0.5, 0.0], vel=[-2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[1.0, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    t = predict_ball_ball_collision(a, b, G)
    assert t is None


# ── predict_ball_ball_collision: edge cases ──

def test_ball_ball_collision_both_stopped():
    a = BallState(pos=[0.0, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    b = BallState(pos=[1.0, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    t = predict_ball_ball_collision(a, b, G)
    assert t is None


def test_ball_ball_collision_parallel_same_speed():
    a = BallState(pos=[0.0, 0.0], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.0, 0.2], vel=[2.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    t = predict_ball_ball_collision(a, b, G)
    assert t is None


# ── predict_ball_ball_collision: self-consistency ──

def test_ball_ball_collision_distance_equals_2r_at_collision():
    a = BallState(pos=[0.0, 0.0], vel=[5.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.2, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    t = predict_ball_ball_collision(a, b, G)
    assert t is not None

    # Advance ball A by t (B is stationary, stays at [0.2, 0])
    pos_a, _, _ = sliding_motion(a, t, G)
    pos_b = b.pos

    dist = np.linalg.norm(pos_a - pos_b)
    assert Decimal(str(dist)).quantize(THREE_PLACES) == Decimal(str(2 * BALL_RADIUS)).quantize(THREE_PLACES)


# ── predict_ball_ball_collision: additional base cases ──

def test_ball_ball_collision_head_on():
    # Two balls moving toward each other along x-axis
    # A at [0,0] vel=[3,0], B at [0.5,0] vel=[-3,0]
    # Closing speed is much higher, should collide quickly
    a = BallState(pos=[0.0, 0.0], vel=[3.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.5, 0.0], vel=[-3.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    t = predict_ball_ball_collision(a, b, G)
    assert t is not None
    # Gap = 0.5 - 2*0.028575 = 0.44285, closing speed ≈ 6 m/s
    # Approximate: t ≈ 0.44285/6 ≈ 0.074 (friction makes it slightly more)
    assert t < 0.1

def test_ball_ball_collision_overtaking():
    # Fast ball catches slow ball, both moving same direction
    # A at [0,0] vel=[5,0], B at [0.2,0] vel=[1,0]
    a = BallState(pos=[0.0, 0.0], vel=[5.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[0.2, 0.0], vel=[1.0, 0.0], omega=0.0, motion=MotionState.SLIDING)
    t = predict_ball_ball_collision(a, b, G)
    assert t is not None
    # Relative closing speed ≈ 4 m/s, gap ≈ 0.143m → t ≈ 0.036
    assert t < 0.1


# ── predict_ball_ball_collision: additional edge cases ──

def test_ball_ball_collision_slow_ball_stops_before_reaching():
    # Very slow ball far from stationary ball — decelerates to stop
    a = BallState(pos=[0.0, 0.0], vel=[0.1, 0.0], omega=0.0, motion=MotionState.SLIDING)
    b = BallState(pos=[5.0, 0.0], vel=[0.0, 0.0], omega=0.0, motion=MotionState.STOPPED)
    t = predict_ball_ball_collision(a, b, G)
    assert t is None