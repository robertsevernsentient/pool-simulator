import numpy as np
from engine.physics.ball_state import BallState, MotionState
from engine.physics.tuneable_constants import BALL_MASS, BALL_RADIUS

def cue_strike(position, direction, speed):

    if np.linalg.norm(direction) == 0:
        direction = np.array([0.0, 0.0])  # Default direction if zero vector
    else:
        direction = direction / np.linalg.norm(direction)

    vel = direction * speed

    return BallState(
        pos=np.array(position, dtype=float),
        vel=vel,
        omega=0.0,
        motion=MotionState.SLIDING
    )

def sliding_motion(ball, t, g):

    v0 = ball.vel
    speed = np.linalg.norm(v0)

    if speed == 0:
        return ball.pos, ball.vel

    direction = v0 / speed

    # linear deceleration
    a = -ball.mu() * g * direction

    new_vel = v0 + a * t
    new_pos = ball.pos + v0 * t + 0.5 * a * t * t

    # angular acceleration
    alpha = (5 * ball.mu() * g) / (2 * ball.radius)

    new_omega = ball.omega + alpha * t

    ball.omega = new_omega

    return new_pos, new_vel

def rolling_motion(ball, t, g):

    v = ball.vel
    speed = np.linalg.norm(v)

    if speed == 0:
        return ball.pos, ball.vel

    direction = v / speed
    a = -ball.mu() * g * direction

    pos = ball.pos + v*t + 0.5*a*t*t
    vel = v + a*t

    return pos, vel

def time_to_reach_point(ball, target, g):
    """
    Calculate the time for a sliding or rolling ball to reach a given point.
    Returns None if the ball stops before reaching the point.

    Assumes the ball moves in a straight line toward the target.
    """
    pos = np.array(ball.pos, dtype=float)
    target = np.array(target, dtype=float)
    vel = np.array(ball.vel, dtype=float)

    if np.allclose(vel, 0):
        return None  # ball not moving

    # Direction from ball to target
    direction = target - pos
    distance = np.linalg.norm(direction)
    if distance == 0:
        return 0.0

    direction /= distance  # normalize

    # Velocity component along direction
    v_along = np.dot(vel, direction)
    if v_along <= 0:
        return None  # ball moving away from target

    # Deceleration magnitude
    if ball.motion == MotionState.SLIDING:
        a = ball.mu() * g
        t_max = time_sliding_to_rolling(ball, g)
    elif ball.motion == MotionState.ROLLING:
        a = ball.mu() * g
        t_max = time_rolling_to_stop(ball, g)
    else:
        return None  # stopped or spinning-only

    # Solve 0.5 * a * t^2 - v0 * t + s = 0
    # Quadratic: 0.5 * a * t^2 - v_along * t + distance = 0
    A = 0.5 * a
    B = -v_along
    C = distance

    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        return None  # never reaches target

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B + sqrt_disc) / (2*A)
    t2 = (-B - sqrt_disc) / (2*A)

    # Pick the positive root
    t_candidates = [t for t in [t1, t2] if t >= 0]
    if not t_candidates:
        return None

    t = min(t_candidates)

    # Check if ball stops sliding/rolling before reaching
    if t > t_max:
        return None

    return t

def spinning_motion(ball, t, spin_friction):

    omega = ball.omega * np.exp(-spin_friction * t)
    return ball.pos, omega

def time_sliding_to_rolling(ball, g):

    v0 = np.linalg.norm(ball.vel)

    if v0 == 0:
        return None

    return (2 * v0) / (7 * ball.mu() * g)

def time_rolling_to_stop(ball, g):

    speed = np.linalg.norm(ball.vel)
    return speed / (ball.mu() * g)

def time_spin_to_stop(ball):

    if ball.omega == 0:
        return None

    return abs(ball.omega) / ball.mu()