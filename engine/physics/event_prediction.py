from dataclasses import dataclass

import numpy as np
from engine.physics.ball_state import MotionState
from engine.physics.motion_models import time_rolling_to_stop, time_sliding_to_rolling, time_spin_to_stop
from engine.physics.tuneable_constants import G, MU_ROLL, MU_SLIDE, SPIN_FRICTION


@dataclass(order=True)
class Event:
    time: float
    event_type: str
    a: int
    b: int | None

def predict_ball_ball_collision(a, b):

    dp = a.pos - b.pos
    dv = a.vel - b.vel

    r = a.radius + b.radius

    A = np.dot(dv, dv)
    B = 2 * np.dot(dp, dv)
    C = np.dot(dp, dp) - r*r

    disc = B*B - 4*A*C

    if disc < 0 or A == 0:
        return None

    t = (-B - np.sqrt(disc)) / (2*A)

    if t <= 1e-6:
        return None

    return t

def predict_rail_collision(ball, table):

    times = []

    vx, vy = ball.vel
    x, y = ball.pos
    r = ball.radius

    if vx > 0:
        times.append((table.width - r - x) / vx)

    if vx < 0:
        times.append((r - x) / vx)

    if vy > 0:
        times.append((table.height - r - y) / vy)

    if vy < 0:
        times.append((r - y) / vy)

    times = [t for t in times if t > 1e-6]

    if not times:
        return None

    return min(times)

def predict_state_transition(ball):

    if ball.motion == MotionState.SLIDING:
        return time_sliding_to_rolling(ball, MU_SLIDE, G)

    if ball.motion == MotionState.ROLLING:
        return time_rolling_to_stop(ball, MU_ROLL, G)

    if ball.motion == MotionState.SPINNING:
        return time_spin_to_stop(ball, SPIN_FRICTION)

    return None


def compute_next_event(state, table):

    earliest = None

    # ball-ball collisions
    for i in range(len(state.balls)):
        for j in range(i+1, len(state.balls)):
            t = predict_ball_ball_collision(state.balls[i], state.balls[j])

            if t and (earliest is None or t < earliest.time):
                earliest = Event(state.time + t, "BALL_COLLISION", i, j)

    # rail collisions
    for i, ball in enumerate(state.balls):

        t = predict_rail_collision(ball, table)

        if t and (earliest is None or t < earliest.time):
            earliest = Event(state.time + t, "RAIL_COLLISION", i, None)

    # state transitions
    for i, ball in enumerate(state.balls):

        t = predict_state_transition(ball)

        if t and (earliest is None or t < earliest.time):
            earliest = Event(state.time + t, "STATE_CHANGE", i, None)

    return earliest