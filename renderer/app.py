import logging
import sys

import pygame

from renderer.constants import COL_BG, FPS, WIN_H, WIN_W
from renderer.coordinates import clamp_to_table, overlaps_any, screen_to_world, world_to_screen
from renderer.drawing import (
    draw_aim_line, draw_balls_static, draw_balls_with_state,
    draw_ghost_ball, draw_hud, draw_table,
)
from renderer.input_handler import handle_events

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Pool Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    state = {
        "mode": "place_cue",
        "balls": [],
        "snapshots": [],
        "frame_idx": 0,
        "aiming": False,
        "aim_start": None,
        "speed": 1.0,
        "running": True,
    }

    while state["running"]:
        mouse_pos = pygame.mouse.get_pos()

        handle_events(pygame.event.get(), state)

        # ── update ──
        if state["mode"] == "playing" and state["snapshots"]:
            state["frame_idx"] += state["speed"]
            if int(state["frame_idx"]) >= len(state["snapshots"]):
                state["frame_idx"] = len(state["snapshots"]) - 1
                state["mode"] = "done"

        # ── draw ──
        screen.fill(COL_BG)
        draw_table(screen)

        mode = state["mode"]
        snapshots = state["snapshots"]
        balls = state["balls"]

        if mode in ("playing", "done") and snapshots:
            idx = min(int(state["frame_idx"]), len(snapshots) - 1)
            draw_balls_with_state(screen, snapshots[idx])
        else:
            if balls:
                draw_balls_static(screen, balls)

            if mode in ("place_cue", "place_object"):
                mouse_world = screen_to_world(mouse_pos)
                w = clamp_to_table(mouse_world)
                if not overlaps_any(w, balls):
                    draw_ghost_ball(screen, world_to_screen(w), mode)

            if state["aiming"] and mode == "aim":
                cue_screen = world_to_screen(balls[0].pos)
                draw_aim_line(screen, font, cue_screen,
                              state["aim_start"], mouse_pos)

        draw_hud(screen, font, mode, state["speed"])
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
