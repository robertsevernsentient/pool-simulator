from engine.physics.ball_state import MotionState
from engine.physics.event_prediction import compute_next_event
from engine.physics.event_resolution import resolve_event
from engine.physics.motion_models import rolling_motion, sliding_motion
from engine.physics.tuneable_constants import G


def advance_state(state, dt):

    for ball in state.balls:

        if ball.motion == MotionState.SLIDING:

            ball.pos, ball.vel = sliding_motion(ball, dt, G)

        elif ball.motion == MotionState.ROLLING:

            ball.pos, ball.vel = rolling_motion(ball, dt, G)

    state.time += dt

def simulate(state, table):

    while True:

        event = compute_next_event(state, table)

        if event is None:
            break

        dt = event.time - state.time

        advance_state(state, dt)

        resolve_event(state, event)