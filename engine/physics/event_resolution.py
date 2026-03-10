from engine.physics.ball_state import MotionState
import numpy as np
from engine.physics.tuneable_constants import RAIL_RESTITUTION

def resolve_ball_collision(a, b):

    n = a.pos - b.pos
    n /= np.linalg.norm(n)

    rel_vel = a.vel - b.vel
    vel_norm = np.dot(rel_vel, n)

    if vel_norm > 0:
        return

    impulse = -2 * vel_norm / (a.mass + b.mass)

    a.vel += impulse * b.mass * n
    b.vel -= impulse * a.mass * n

    a.motion = MotionState.SLIDING
    b.motion = MotionState.SLIDING

def resolve_rail_collision(ball, normal, restitution):

    v_n = np.dot(ball.vel, normal) * normal
    v_t = ball.vel - v_n

    ball.vel = v_t - restitution * v_n
    ball.motion = MotionState.SLIDING


def resolve_event(state, event, table):

    if event.event_type == "BALL_COLLISION":

        a = state.balls[event.a]
        b = state.balls[event.b]

        resolve_ball_collision(a, b)

    elif event.event_type == "RAIL_COLLISION":

        ball = state.balls[event.a]

        # determine rail normal
        if ball.pos[0] <= ball.radius:
            normal = np.array([1.0, 0.0])
        elif ball.pos[0] >= table.width - ball.radius:
            normal = np.array([-1.0, 0.0])
        elif ball.pos[1] <= ball.radius:
            normal = np.array([0.0, 1.0])
        else:
            normal = np.array([0.0, -1.0])

        resolve_rail_collision(ball, normal, RAIL_RESTITUTION)

    elif event.event_type == "STATE_CHANGE":

        ball = state.balls[event.a]

        if ball.motion == MotionState.SLIDING:
            ball.motion = MotionState.ROLLING

        elif ball.motion == MotionState.ROLLING:
            ball.motion = MotionState.STOPPED
            ball.vel[:] = 0