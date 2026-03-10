from dataclasses import dataclass

import numpy as np
from engine.physics.ball_state import MotionState
from engine.physics.motion_models import ball_acceleration, time_rolling_to_stop, time_sliding_to_rolling, time_to_reach_point
from engine.physics.tuneable_constants import G


@dataclass(order=True)
class Event:
    time: float
    event_type: str
    a: int
    b: int | None

def predict_ball_ball_collision(a, b, g):

    dp = a.pos - b.pos
    dv = a.vel - b.vel
    da = ball_acceleration(a, g) - ball_acceleration(b, g)

    r = a.radius + b.radius

    # Δp(t) = dp + dv·t + ½·da·t²
    # |Δp(t)|² = (2r)²  →  quartic in t
    half_da = 0.5 * da

    c4 = np.dot(half_da, half_da)
    c3 = 2 * np.dot(dv, half_da)
    c2 = np.dot(dv, dv) + 2 * np.dot(dp, half_da)
    c1 = 2 * np.dot(dp, dv)
    c0 = np.dot(dp, dp) - r * r

    coeffs = [c4, c3, c2, c1, c0]

    # Strip leading zeros to avoid degenerate polynomial
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)

    if len(coeffs) <= 1:
        return None

    # Cap at the earliest state transition for either ball
    t_max = float('inf')
    for ball in [a, b]:
        t_trans = predict_state_transition(ball)
        if t_trans is not None and t_trans < t_max:
            t_max = t_trans

    roots = np.roots(coeffs)

    # Filter: real, positive, beyond epsilon, within current motion regime
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 1e-6 and r.real <= t_max]

    if not real_roots:
        return None

    return min(real_roots)

def predict_rail_collision(ball, table):
    pos = _predict_rail_collision_position(ball, table)
    if pos is None:
        return None
    return time_to_reach_point(ball, pos, G)

def _predict_rail_collision_position(ball, table):

    collisions = []

    vx, vy = ball.vel
    x, y = ball.pos
    r = ball.radius

    # right rail collision
    if vx > 0:
        collision_time = (table.width - r - x) / vx
        collision_position = [table.width - r, y + vy * collision_time]
        collisions.append((collision_time, collision_position))

    # left rail collision
    if vx < 0:
        collision_time = (r - x) / vx
        collision_position = [r, y + vy * collision_time]
        collisions.append((collision_time, collision_position))

    # top rail collision
    if vy > 0:
        collision_time = (table.height - r - y) / vy
        collision_position = [x + vx * collision_time, table.height - r]
        collisions.append((collision_time, collision_position))

    # bottom rail collision
    if vy < 0:
        collision_time = (r - y) / vy
        collision_position = [x + vx * collision_time, r]
        collisions.append((collision_time, collision_position))

    collisions = [c for c in collisions if c[0] > 1e-6]

    if not collisions:
        return None

    return min(collisions, key=lambda x: x[0])[1]

def predict_state_transition(ball):

    if ball.motion == MotionState.SLIDING:
        return time_sliding_to_rolling(ball, G)

    if ball.motion == MotionState.ROLLING:
        return time_rolling_to_stop(ball, G)

    return None


def compute_next_event(state, table):

    earliest = None

    # ball-ball collisions
    for i in range(len(state.balls)):
        for j in range(i+1, len(state.balls)):
            t = predict_ball_ball_collision(state.balls[i], state.balls[j], G)

            if t and (earliest is None or state.time + t < earliest.time):
                earliest = Event(state.time + t, "BALL_COLLISION", i, j)

    # rail collisions
    for i, ball in enumerate(state.balls):

        t = predict_rail_collision(ball, table)

        if t and (earliest is None or state.time + t < earliest.time):
            earliest = Event(state.time + t, "RAIL_COLLISION", i, None)

    # state transitions
    for i, ball in enumerate(state.balls):

        t = predict_state_transition(ball)

        if t and (earliest is None or state.time + t < earliest.time):
            earliest = Event(state.time + t, "STATE_CHANGE", i, None)

    return earliest