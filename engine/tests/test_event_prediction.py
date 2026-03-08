from decimal import Decimal
from engine.physics.ball_state import BallState, MotionState
from engine.physics.event_prediction import _predict_rail_collision_position, predict_rail_collision, predict_state_transition
from engine.physics.motion_models import cue_strike
from engine.physics.tuneable_constants import STANDARD_9_FOOT

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

# TODO I don't like this. You can be rolling and spinning at the same time
def test_state_change_spin_to_stop_detection():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1, 1],
        speed=2.0
    )
    cue.omega = 10.0  # Set initial spin
    cue.motion = MotionState.SPINNING
    spin_to_stop_time = predict_state_transition(cue)

    assert Decimal(spin_to_stop_time).quantize(THREE_PLACES) == Decimal(str("1.250"))

def test_state_change_spin_to_stop_detection_zero_omega():
    cue = cue_strike(
        position=[0.5,0.7],
        direction=[1, 1],
        speed=2.0
    )
    cue.omega = 0.0  # Set zero spin
    cue.motion = MotionState.SPINNING
    spin_to_stop_time = predict_state_transition(cue)

    assert spin_to_stop_time is None

def test_state_change_when_stopped_detection():
    cue = BallState(
        pos=[0.5,0.7],
        vel=[0, 0],
        omega=0.0,
        motion=MotionState.STOPPED,
        radius=1,
        mass=1.0
    )

    time = predict_state_transition(cue)

    assert time is None