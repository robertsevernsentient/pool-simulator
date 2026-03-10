import numpy as np
from engine.physics.ball_state import BallState, MotionState
from engine.physics.tuneable_constants import BALL_MASS

POSITION_DP = 6

def ball_acceleration(ball, g):
    if ball.motion == MotionState.STOPPED:
        return np.array([0.0, 0.0])
    speed = np.linalg.norm(ball.vel)
    if speed == 0:
        return np.array([0.0, 0.0])
    direction = ball.vel / speed

    if ball.motion == MotionState.ROLLING:
        # Rolling friction always opposes motion (no slip by definition)
        return -ball.mu() * g * direction

    # Sliding: friction opposes slip direction
    slip = speed - ball.radius * ball.omega
    s = 1.0 if slip >= 0 else -1.0
    return -s * ball.mu() * g * direction


def cue_strike(position, direction, speed):

    direction = np.array(direction, dtype=float)

    if speed == 0.0:
        return BallState(
            pos=np.array(position, dtype=float),
            vel=np.array([0.0, 0.0]),
            omega=0.0,
            motion=MotionState.STOPPED
        )

    if np.linalg.norm(direction) == 0:
        raise ValueError("direction must be non-zero when speed is non-zero")

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
        return ball.pos, ball.vel, ball.omega

    direction = v0 / speed

    # Slip at contact point: v_slip = v - Rω
    # Friction opposes slip, so its sign depends on slip direction
    slip = speed - ball.radius * ball.omega
    s = 1.0 if slip >= 0 else -1.0

    # Linear: friction opposes slip (decelerates for backspin/stun, accelerates for topspin)
    a = -s * ball.mu() * g * direction

    new_vel = v0 + a * t
    new_pos = np.round(ball.pos + v0 * t + 0.5 * a * t * t, POSITION_DP)

    # Angular: friction reduces slip
    alpha = s * (5 * ball.mu() * g) / (2 * ball.radius)

    new_omega = ball.omega + alpha * t

    return new_pos, new_vel, new_omega

def rolling_motion(ball, t, g):

    v = ball.vel
    speed = np.linalg.norm(v)

    if speed == 0:
        return ball.pos, ball.vel, ball.omega

    direction = v / speed
    a = -ball.mu() * g * direction

    pos = np.round(ball.pos + v*t + 0.5*a*t*t, POSITION_DP)
    vel = v + a*t
    omega = np.linalg.norm(vel) / ball.radius

    return pos, vel, omega

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
        return None  # stopped

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

def time_sliding_to_rolling(ball, g):

    v0 = np.linalg.norm(ball.vel)

    if v0 == 0:
        raise ValueError("ball is sliding with zero velocity")

    # Generalized: accounts for initial omega (backspin/topspin)
    slip = abs(v0 - ball.radius * ball.omega)
    return (2 * slip) / (7 * ball.mu() * g)

def time_rolling_to_stop(ball, g):

    speed = np.linalg.norm(ball.vel)

    if speed == 0:
        raise ValueError("ball is rolling with zero velocity")

    return speed / (ball.mu() * g)

