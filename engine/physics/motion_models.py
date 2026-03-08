import numpy as np
from engine.physics.ball_state import BallState, MotionState
from engine.physics.tuneable_constants import BALL_MASS, BALL_RADIUS

def cue_strike(position, direction, speed):

    direction = direction / np.linalg.norm(direction)

    vel = direction * speed

    return BallState(
        pos=np.array(position, dtype=float),
        vel=vel,
        omega=0.0,
        motion=MotionState.SLIDING,
        radius=BALL_RADIUS,
        mass=BALL_MASS
    )

def sliding_motion(ball, t, mu_slide, g):

    v0 = ball.vel
    speed = np.linalg.norm(v0)

    if speed == 0:
        return ball.pos, ball.vel

    direction = v0 / speed

    # linear deceleration
    a = -mu_slide * g * direction

    new_vel = v0 + a * t
    new_pos = ball.pos + v0 * t + 0.5 * a * t * t

    # angular acceleration
    alpha = (5 * mu_slide * g) / (2 * ball.radius)

    new_omega = ball.omega + alpha * t

    ball.omega = new_omega

    return new_pos, new_vel

def rolling_motion(ball, t, mu_roll, g):

    v = ball.vel
    speed = np.linalg.norm(v)

    if speed == 0:
        return ball.pos, ball.vel

    direction = v / speed
    a = -mu_roll * g * direction

    pos = ball.pos + v*t + 0.5*a*t*t
    vel = v + a*t

    return pos, vel

def spinning_motion(ball, t, spin_friction):

    omega = ball.omega * np.exp(-spin_friction * t)
    return ball.pos, omega

def time_sliding_to_rolling(ball, mu_slide, g):

    v0 = np.linalg.norm(ball.vel)

    if v0 == 0:
        return None

    return (2 * v0) / (7 * mu_slide * g)

def time_rolling_to_stop(ball, mu_roll, g):

    speed = np.linalg.norm(ball.vel)
    return speed / (mu_roll * g)

def time_spin_to_stop(ball, spin_friction):

    if ball.omega == 0:
        return None

    return abs(ball.omega) / spin_friction