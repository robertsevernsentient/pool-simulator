from engine.physics.ball_state import BallState, MotionState
from engine.physics.motion_models import cue_strike
from engine.physics.tuneable_constants import BALL_RADIUS, STANDARD_9_FOOT

TABLE = STANDARD_9_FOOT
R = BALL_RADIUS

# Balls placed 1/3 of a table apart, centred on the table
GAP = TABLE.width / 3
CUE_X = TABLE.width / 2 - GAP / 2
OBJ_X = CUE_X + GAP
CY = TABLE.height / 2


def scenario_rolling_direct():
    """1: Cue strike that transitions to rolling before hitting the object ball."""
    cue = cue_strike(position=[CUE_X, CY], direction=[1, 0], speed=2.0)
    obj = BallState(pos=[OBJ_X, CY], vel=[0, 0], omega=0.0, motion=MotionState.STOPPED)
    return [cue, obj], "Rolling direct"


def scenario_half_ball_rolling():
    """2: Cue strike that transitions to rolling before a 1/2 ball hit."""
    cue = cue_strike(position=[CUE_X, CY], direction=[1, 0], speed=2.5)
    obj = BallState(pos=[OBJ_X, CY + R], vel=[0, 0], omega=0.0, motion=MotionState.STOPPED)
    return [cue, obj], "Rolling 1/2 ball"


def scenario_stop_shot():
    """3: Stop shot -- hit low on cue ball (backspin). Friction brings omega to exactly 0
    at contact, so cue stops dead on a direct hit."""
    cue = cue_strike(position=[CUE_X, CY], direction=[1, 0], speed=3.0, spin=-0.435)
    obj = BallState(pos=[OBJ_X, CY], vel=[0, 0], omega=0.0, motion=MotionState.STOPPED)
    return [cue, obj], "Stop shot"


def scenario_half_ball_stun():
    """4: 1/2 ball stun shot. Same backspin as stop shot -- omega=0 at contact,
    cue deflects along the tangent line."""
    cue = cue_strike(position=[CUE_X, CY], direction=[1, 0], speed=3.0, spin=-0.435)
    obj = BallState(pos=[OBJ_X, CY + R], vel=[0, 0], omega=0.0, motion=MotionState.STOPPED)
    return [cue, obj], "Stun 1/2 ball"


def scenario_max_draw_shot():
    """5: Max draw shot -- extra backspin beyond stop shot. Omega still strongly negative
    at contact, so cue draws back after the hit."""
    cue = cue_strike(position=[CUE_X, CY], direction=[1, 0], speed=3.0, spin=-1.0)
    obj = BallState(pos=[OBJ_X, CY], vel=[0, 0], omega=0.0, motion=MotionState.STOPPED)
    return [cue, obj], "Max draw shot"

def scenario_max_follow_shot():
    """6: Max follow shot -- extra topspin beyond stun shot. Omega still strongly positive
    at contact, so cue follows through after the hit."""
    cue = cue_strike(position=[CUE_X, CY], direction=[1, 0], speed=3.0, spin=1.0)
    obj = BallState(pos=[OBJ_X, CY], vel=[0, 0], omega=0.0, motion=MotionState.STOPPED)
    return [cue, obj], "Max follow shot"

def scenario_lag_shot():
    """7: Lag shot -- reference pace. Ball starts at the baulk line (1/4 table from
    the left rail), travels to the far rail, bounces, and should roll back to
    stop near the near rail. Single ball, centre-ball hit (rolling from the start)."""
    baulk_x = TABLE.width / 4          # ≈ 0.71 m from left rail
    speed = 1.832                      # m/s — gentle, controlled pace
    cue = BallState(pos=[baulk_x, CY], vel=[speed, 0.0],
                    omega=speed / R, motion=MotionState.ROLLING)
    return [cue], "Lag shot"


def scenario_baulk_to_rail():
    """8: Calibration shot -- ball from baulk line should just barely reach the far
    rail and stop. Isolates rolling friction from restitution."""
    baulk_x = TABLE.width / 4
    far_rail_dist = TABLE.width - R - baulk_x
    # v0 = sqrt(2 * mu_roll * g * d) — just enough to reach the rail
    from engine.physics.tuneable_constants import MU_ROLL, G
    import numpy as np
    speed = np.sqrt(2 * MU_ROLL * G * far_rail_dist)
    cue = BallState(pos=[baulk_x, CY], vel=[speed, 0.0],
                    omega=speed / R, motion=MotionState.ROLLING)
    return [cue], "Baulk to rail"


ALL_SCENARIOS = [
    scenario_rolling_direct,
    scenario_half_ball_rolling,
    scenario_stop_shot,
    scenario_half_ball_stun,
    scenario_max_draw_shot,
    scenario_max_follow_shot,
    scenario_lag_shot,
    scenario_baulk_to_rail,
]
