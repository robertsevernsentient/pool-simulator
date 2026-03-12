import logging
import sys

import numpy as np
import pygame

from engine.physics.ball_state import BallState, MotionState
from engine.physics.event_prediction import compute_next_event
from engine.physics.event_resolution import resolve_event
from engine.physics.motion_models import rolling_motion, sliding_motion
from engine.physics.simulation_state import SimulationState
from engine.physics.tuneable_constants import BALL_RADIUS, G, STANDARD_9_FOOT
from scenarios import ALL_SCENARIOS

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ── layout constants ──

TABLE = STANDARD_9_FOOT
SCALE = 400                       # pixels per meter
MARGIN = 40                       # pixels around table
TABLE_W = int(TABLE.width * SCALE)
TABLE_H = int(TABLE.height * SCALE)
WIN_W = TABLE_W + 2 * MARGIN
WIN_H = TABLE_H + 2 * MARGIN
BALL_PX = max(int(BALL_RADIUS * SCALE), 4)
FPS = 60
PLAYBACK_SPEED = 1.0

# ── colours ──

COL_BG = (30, 30, 30)
COL_CLOTH = (0, 100, 50)
COL_RAIL = (60, 30, 10)
COL_AIM = (255, 255, 255)
COL_TEXT = (220, 220, 220)
RAIL_WIDTH = 8

# Ball colours by motion state: (sliding, rolling, stopped)
CUE_COLOURS = {
    MotionState.SLIDING: (255, 255, 240),
    MotionState.ROLLING: (200, 200, 185),
    MotionState.STOPPED: (150, 150, 140),
}
OBJ_COLOURS = {
    MotionState.SLIDING: (230, 50, 50),
    MotionState.ROLLING: (170, 30, 30),
    MotionState.STOPPED: (120, 20, 20),
}


# ── coordinate helpers ──

def world_to_screen(pos):
    return (int(MARGIN + pos[0] * SCALE),
            int(MARGIN + (TABLE.height - pos[1]) * SCALE))

def screen_to_world(pixel):
    x = (pixel[0] - MARGIN) / SCALE
    y = TABLE.height - (pixel[1] - MARGIN) / SCALE
    return np.array([x, y])

def clamp_to_table(world_pos):
    r = BALL_RADIUS
    x = max(r, min(TABLE.width - r, world_pos[0]))
    y = max(r, min(TABLE.height - r, world_pos[1]))
    return np.array([x, y])

def overlaps_any(pos, balls, ignore_index=None):
    for i, b in enumerate(balls):
        if i == ignore_index:
            continue
        if np.linalg.norm(pos - b.pos) < 2 * BALL_RADIUS + 1e-6:
            return True
    return False


# ── preset scenarios ──

SCENARIO_KEYS = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8]
SCENARIOS = dict(zip(SCENARIO_KEYS, ALL_SCENARIOS))


# ── simulation snapshot recording ──

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


# ── drawing ──

def draw_table(surface):
    pygame.draw.rect(surface, COL_RAIL,
                     (MARGIN - RAIL_WIDTH, MARGIN - RAIL_WIDTH,
                      TABLE_W + 2 * RAIL_WIDTH, TABLE_H + 2 * RAIL_WIDTH))
    pygame.draw.rect(surface, COL_CLOTH,
                     (MARGIN, MARGIN, TABLE_W, TABLE_H))

def draw_balls_with_state(surface, frame):
    for i, (pos, motion) in enumerate(frame):
        sx, sy = world_to_screen(pos)
        colours = CUE_COLOURS if i == 0 else OBJ_COLOURS
        colour = colours[motion]
        pygame.draw.circle(surface, colour, (sx, sy), BALL_PX)
        pygame.draw.circle(surface, (0, 0, 0), (sx, sy), BALL_PX, 1)

def draw_balls_static(surface, balls):
    for i, b in enumerate(balls):
        sx, sy = world_to_screen(b.pos)
        colours = CUE_COLOURS if i == 0 else OBJ_COLOURS
        colour = colours[b.motion]
        pygame.draw.circle(surface, colour, (sx, sy), BALL_PX)
        pygame.draw.circle(surface, (0, 0, 0), (sx, sy), BALL_PX, 1)

def draw_hud(surface, font, mode, speed):
    lines = []
    if mode == "place_cue":
        lines.append("Click to place cue ball  |  1-5 presets")
    elif mode == "place_object":
        lines.append("Click to place object balls  |  SPACE to aim  |  1-5 presets")
    elif mode == "aim":
        lines.append("Drag from cue ball to shoot  |  R to reset  |  1-5 presets")
    elif mode == "playing":
        lines.append(f"Playing (x{speed:.1f})  |  +/- speed  |  R to reset")
    elif mode == "done":
        lines.append("Done  |  R to reset  |  1-5 presets")
    lines.append("1:Roll 2:1/2Roll 3:Stop 4:Stun 5:Draw 6:Follow 7:Lag 8:Baulk")
    for i, line in enumerate(lines):
        txt = font.render(line, True, COL_TEXT)
        surface.blit(txt, (10, WIN_H - 44 + i * 20))


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


# ── main ──

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Pool Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    mode = "place_cue"
    balls = []
    snapshots = []
    frame_idx = 0
    aiming = False
    aim_start = None
    speed = 1.0

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_world = screen_to_world(mouse_pos)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False

                elif ev.key == pygame.K_r:
                    mode = "place_cue"
                    balls.clear()
                    snapshots.clear()
                    frame_idx = 0
                    aiming = False
                    speed = 1.0

                elif ev.key in SCENARIOS:
                    snapshots, balls = launch_scenario(SCENARIOS[ev.key])
                    frame_idx = 0
                    speed = 1.0
                    mode = "playing"

                elif ev.key == pygame.K_SPACE and mode == "place_object":
                    if len(balls) >= 1:
                        mode = "aim"

                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS) and mode == "playing":
                    speed = min(5.0, speed + 0.5)

                elif ev.key == pygame.K_MINUS and mode == "playing":
                    speed = max(0.25, speed - 0.25)

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                w = clamp_to_table(mouse_world)

                if mode == "place_cue":
                    if not overlaps_any(w, balls):
                        balls.append(BallState(pos=w, vel=[0, 0], omega=0.0,
                                               motion=MotionState.STOPPED))
                        mode = "place_object"

                elif mode == "place_object":
                    if not overlaps_any(w, balls):
                        balls.append(BallState(pos=w, vel=[0, 0], omega=0.0,
                                               motion=MotionState.STOPPED))

                elif mode == "aim":
                    cue_screen = world_to_screen(balls[0].pos)
                    dist = np.linalg.norm(np.array(mouse_pos) - np.array(cue_screen))
                    if dist < BALL_PX * 3:
                        aiming = True
                        aim_start = np.array(mouse_pos, dtype=float)

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if aiming and mode == "aim":
                    aiming = False
                    drag = np.array(aim_start) - np.array(mouse_pos, dtype=float)
                    drag_len = np.linalg.norm(drag)
                    if drag_len > 5:
                        shot_speed = min(drag_len / 65.0, 6.0)
                        direction = drag / drag_len
                        world_dir = np.array([direction[0], -direction[1]])

                        sim_balls = [BallState(pos=b.pos.copy(), vel=b.vel.copy(),
                                               omega=b.omega, motion=b.motion)
                                     for b in balls]
                        sim_balls[0].vel = world_dir * shot_speed
                        sim_balls[0].motion = MotionState.SLIDING
                        state = SimulationState(balls=sim_balls, time=0.0)

                        log.info("Shot: speed=%.3f dir=[%.3f, %.3f]",
                                 shot_speed, world_dir[0], world_dir[1])
                        snapshots = record_simulation(state, TABLE)
                        log.info("Recorded %d frames", len(snapshots))
                        frame_idx = 0
                        mode = "playing"

        # ── update ──
        if mode == "playing" and snapshots:
            frame_idx += speed
            if int(frame_idx) >= len(snapshots):
                frame_idx = len(snapshots) - 1
                mode = "done"

        # ── draw ──
        screen.fill(COL_BG)
        draw_table(screen)

        if mode in ("playing", "done") and snapshots:
            idx = min(int(frame_idx), len(snapshots) - 1)
            draw_balls_with_state(screen, snapshots[idx])
        else:
            if balls:
                draw_balls_static(screen, balls)

            if mode in ("place_cue", "place_object"):
                w = clamp_to_table(mouse_world)
                if not overlaps_any(w, balls):
                    sx, sy = world_to_screen(w)
                    ghost_surf = pygame.Surface((BALL_PX * 2, BALL_PX * 2), pygame.SRCALPHA)
                    colours = CUE_COLOURS if mode == "place_cue" else OBJ_COLOURS
                    colour = (*colours[MotionState.STOPPED], 100)
                    pygame.draw.circle(ghost_surf, colour,
                                       (BALL_PX, BALL_PX), BALL_PX)
                    screen.blit(ghost_surf, (sx - BALL_PX, sy - BALL_PX))

            if aiming and mode == "aim":
                drag = np.array(aim_start) - np.array(mouse_pos, dtype=float)
                drag_len = np.linalg.norm(drag)
                if drag_len > 0:
                    direction = drag / drag_len
                    cue_screen = world_to_screen(balls[0].pos)
                    end = (int(cue_screen[0] + direction[0] * drag_len),
                           int(cue_screen[1] + direction[1] * drag_len))
                    pygame.draw.line(screen, COL_AIM, cue_screen, end, 2)
                    power_pct = min(drag_len / 65.0 / 6.0, 1.0)
                    power_txt = font.render(f"Power: {power_pct*100:.0f}%", True, COL_TEXT)
                    screen.blit(power_txt, (10, 10))

        draw_hud(screen, font, mode, speed)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
