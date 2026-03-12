import logging

import numpy as np
import pygame

from engine.physics.ball_state import BallState, MotionState
from engine.physics.simulation_state import SimulationState
from renderer.constants import BALL_PX, TABLE
from renderer.coordinates import clamp_to_table, overlaps_any, world_to_screen
from renderer.recording import launch_scenario, record_simulation
from scenarios import ALL_SCENARIOS

log = logging.getLogger(__name__)

SCENARIO_KEYS = [
    pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
]
SCENARIOS = dict(zip(SCENARIO_KEYS, ALL_SCENARIOS))


def handle_events(events, state):
    """Process pygame events and return updated app state dict.

    `state` keys: mode, balls, snapshots, frame_idx, aiming, aim_start, speed, running
    """
    mouse_pos = pygame.mouse.get_pos()

    for ev in events:
        if ev.type == pygame.QUIT:
            state["running"] = False

        elif ev.type == pygame.KEYDOWN:
            _handle_keydown(ev, state)

        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            _handle_mouse_down(ev, mouse_pos, state)

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            _handle_mouse_up(ev, mouse_pos, state)

    return state


def _handle_keydown(ev, state):
    if ev.key == pygame.K_ESCAPE:
        state["running"] = False

    elif ev.key == pygame.K_r:
        state["mode"] = "place_cue"
        state["balls"].clear()
        state["snapshots"].clear()
        state["frame_idx"] = 0
        state["aiming"] = False
        state["speed"] = 1.0

    elif ev.key in SCENARIOS:
        state["snapshots"], state["balls"] = launch_scenario(SCENARIOS[ev.key])
        state["frame_idx"] = 0
        state["speed"] = 1.0
        state["mode"] = "playing"

    elif ev.key == pygame.K_SPACE and state["mode"] == "place_object":
        if len(state["balls"]) >= 1:
            state["mode"] = "aim"

    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS) and state["mode"] == "playing":
        state["speed"] = min(5.0, state["speed"] + 0.5)

    elif ev.key == pygame.K_MINUS and state["mode"] == "playing":
        state["speed"] = max(0.25, state["speed"] - 0.25)


def _handle_mouse_down(ev, mouse_pos, state):
    from renderer.coordinates import screen_to_world
    mouse_world = screen_to_world(mouse_pos)
    w = clamp_to_table(mouse_world)
    mode = state["mode"]
    balls = state["balls"]

    if mode == "place_cue":
        if not overlaps_any(w, balls):
            balls.append(BallState(pos=w, vel=[0, 0], omega=0.0,
                                   motion=MotionState.STOPPED))
            state["mode"] = "place_object"

    elif mode == "place_object":
        if not overlaps_any(w, balls):
            balls.append(BallState(pos=w, vel=[0, 0], omega=0.0,
                                   motion=MotionState.STOPPED))

    elif mode == "aim":
        cue_screen = world_to_screen(balls[0].pos)
        dist = np.linalg.norm(np.array(mouse_pos) - np.array(cue_screen))
        if dist < BALL_PX * 3:
            state["aiming"] = True
            state["aim_start"] = np.array(mouse_pos, dtype=float)


def _handle_mouse_up(ev, mouse_pos, state):
    if not (state["aiming"] and state["mode"] == "aim"):
        return

    state["aiming"] = False
    drag = np.array(state["aim_start"]) - np.array(mouse_pos, dtype=float)
    drag_len = np.linalg.norm(drag)

    if drag_len <= 5:
        return

    shot_speed = min(drag_len / 65.0, 6.0)
    direction = drag / drag_len
    world_dir = np.array([direction[0], -direction[1]])

    balls = state["balls"]
    sim_balls = [BallState(pos=b.pos.copy(), vel=b.vel.copy(),
                           omega=b.omega, motion=b.motion)
                 for b in balls]
    sim_balls[0].vel = world_dir * shot_speed
    sim_balls[0].motion = MotionState.SLIDING
    sim_state = SimulationState(balls=sim_balls, time=0.0)

    log.info("Shot: speed=%.3f dir=[%.3f, %.3f]",
             shot_speed, world_dir[0], world_dir[1])
    state["snapshots"] = record_simulation(sim_state, TABLE)
    log.info("Recorded %d frames", len(state["snapshots"]))
    state["frame_idx"] = 0
    state["mode"] = "playing"
