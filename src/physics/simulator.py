from src.physics.ball_state import MotionState
from src.physics.event_prediction import compute_next_event
from src.physics.event_resolution import resolve_event
from src.physics.motion_models import rolling_motion, sliding_motion
from src.physics.tuneable_constants import G, MU_ROLL, MU_SLIDE


def advance_state(state, dt):

    for ball in state.balls:

        if ball.motion == MotionState.SLIDING:

            ball.pos, ball.vel = sliding_motion(ball, dt, MU_SLIDE, G)

        elif ball.motion == MotionState.ROLLING:

            ball.pos, ball.vel = rolling_motion(ball, dt, MU_ROLL, G)

    state.time += dt

def simulate(state, table):

    while True:

        event = compute_next_event(state, table)

        if event is None:
            break

        dt = event.time - state.time

        advance_state(state, dt)

        resolve_event(state, event)