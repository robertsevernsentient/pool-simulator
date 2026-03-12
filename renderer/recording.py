import logging

from engine.physics.ball_state import BallState, MotionState
from engine.physics.event_prediction import compute_next_event
from engine.physics.event_resolution import resolve_event
from engine.physics.motion_models import rolling_motion, sliding_motion
from engine.physics.simulation_state import SimulationState
from engine.physics.tuneable_constants import G
from renderer.constants import FPS, PLAYBACK_SPEED, TABLE

log = logging.getLogger(__name__)


def record_simulation(state, table):
    """Run the event-driven sim and capture ball positions + motion states at small time steps."""
    snapshots = []
    max_events = 10000

    for step in range(max_events):
        event = compute_next_event(state, table)
        if event is None:
            log.info("Recording done: no more events at t=%.6f", state.time)
            break

        dt = event.time - state.time

        log.debug(
            "Event %d: %s t=%.6f dt=%.6f a=%s b=%s",
            step, event.event_type, event.time, dt, event.a, event.b
        )

        if dt < 0:
            log.warning("Negative dt=%.9f at step %d, stopping", dt, step)
            break

        # Sample positions within this interval for smooth animation
        n_steps = max(1, int(dt * FPS / PLAYBACK_SPEED))
        sub_dt = dt / n_steps
        for i in range(n_steps):
            frame = []
            t = sub_dt * i
            for ball in state.balls:
                if ball.motion == MotionState.SLIDING:
                    pos, _, _ = sliding_motion(ball, t, G)
                elif ball.motion == MotionState.ROLLING:
                    pos, _, _ = rolling_motion(ball, t, G)
                else:
                    pos = ball.pos.copy()
                frame.append((pos.copy(), ball.motion))
            snapshots.append(frame)

        # Advance to event and resolve
        for ball in state.balls:
            if ball.motion == MotionState.SLIDING:
                ball.pos, ball.vel, ball.omega = sliding_motion(ball, dt, G)
            elif ball.motion == MotionState.ROLLING:
                ball.pos, ball.vel, ball.omega = rolling_motion(ball, dt, G)
        state.time += dt

        for i, ball in enumerate(state.balls):
            log.debug(
                "  ball[%d] pos=[%.6f, %.6f] vel=[%.6f, %.6f] omega=%.3f %s",
                i, ball.pos[0], ball.pos[1], ball.vel[0], ball.vel[1],
                ball.omega, ball.motion.name
            )

        resolve_event(state, event, table)

    else:
        log.warning("Hit max events (%d) — possible infinite loop", max_events)

    # Final resting positions
    snapshots.append([(b.pos.copy(), b.motion) for b in state.balls])
    return snapshots


def launch_scenario(scenario_fn):
    """Build and record a preset scenario, return (snapshots, balls)."""
    balls, name = scenario_fn()
    log.info("Scenario: %s", name)
    sim_balls = [BallState(pos=b.pos.copy(), vel=b.vel.copy(),
                           omega=b.omega, motion=b.motion) for b in balls]
    state = SimulationState(balls=sim_balls, time=0.0)
    snapshots = record_simulation(state, TABLE)
    log.info("Recorded %d frames", len(snapshots))
    return snapshots, balls
