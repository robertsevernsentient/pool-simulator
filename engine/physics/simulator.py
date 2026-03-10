import logging

from engine.physics.ball_state import MotionState
from engine.physics.event_prediction import compute_next_event
from engine.physics.event_resolution import resolve_event
from engine.physics.motion_models import rolling_motion, sliding_motion
from engine.physics.tuneable_constants import G

log = logging.getLogger(__name__)


def advance_state(state, dt):

    for ball in state.balls:

        if ball.motion == MotionState.SLIDING:

            ball.pos, ball.vel, ball.omega = sliding_motion(ball, dt, G)

        elif ball.motion == MotionState.ROLLING:

            ball.pos, ball.vel, ball.omega = rolling_motion(ball, dt, G)

    state.time += dt


def simulate(state, table):

    max_events = 10000

    for step in range(max_events):

        event = compute_next_event(state, table)

        if event is None:
            log.info("No more events at t=%.6f", state.time)
            break

        dt = event.time - state.time

        log.debug(
            "Event %d: %s t=%.6f dt=%.6f a=%s b=%s",
            step, event.event_type, event.time, dt, event.a, event.b
        )

        if dt < 0:
            log.warning("Negative dt=%.9f, skipping event", dt)
            break

        advance_state(state, dt)

        for i, ball in enumerate(state.balls):
            log.debug(
                "  ball[%d] pos=[%.6f, %.6f] vel=[%.6f, %.6f] %s",
                i, ball.pos[0], ball.pos[1], ball.vel[0], ball.vel[1], ball.motion.name
            )

        resolve_event(state, event, table)

    else:
        log.warning("Hit max events (%d) — possible infinite loop", max_events)