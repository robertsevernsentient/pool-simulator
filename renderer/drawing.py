import pygame

from engine.physics.ball_state import MotionState
from renderer.constants import (
    BALL_PX, COL_AIM, COL_CLOTH, COL_RAIL, COL_TEXT,
    CUE_COLOURS, MARGIN, OBJ_COLOURS, RAIL_WIDTH,
    TABLE_H, TABLE_W, WIN_H,
)
from renderer.coordinates import world_to_screen


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


def draw_ghost_ball(surface, screen_pos, mode):
    ghost_surf = pygame.Surface((BALL_PX * 2, BALL_PX * 2), pygame.SRCALPHA)
    colours = CUE_COLOURS if mode == "place_cue" else OBJ_COLOURS
    colour = (*colours[MotionState.STOPPED], 100)
    pygame.draw.circle(ghost_surf, colour, (BALL_PX, BALL_PX), BALL_PX)
    sx, sy = screen_pos
    surface.blit(ghost_surf, (sx - BALL_PX, sy - BALL_PX))


def draw_aim_line(surface, font, cue_screen, aim_start, mouse_pos):
    import numpy as np
    drag = np.array(aim_start) - np.array(mouse_pos, dtype=float)
    drag_len = np.linalg.norm(drag)
    if drag_len > 0:
        direction = drag / drag_len
        end = (int(cue_screen[0] + direction[0] * drag_len),
               int(cue_screen[1] + direction[1] * drag_len))
        pygame.draw.line(surface, COL_AIM, cue_screen, end, 2)
        power_pct = min(drag_len / 65.0 / 6.0, 1.0)
        power_txt = font.render(f"Power: {power_pct*100:.0f}%", True, COL_TEXT)
        surface.blit(power_txt, (10, 10))


def draw_hud(surface, font, mode, speed):
    lines = []
    if mode == "place_cue":
        lines.append("Click to place cue ball  |  1-8 presets")
    elif mode == "place_object":
        lines.append("Click to place object balls  |  SPACE to aim  |  1-8 presets")
    elif mode == "aim":
        lines.append("Drag from cue ball to shoot  |  R to reset  |  1-8 presets")
    elif mode == "playing":
        lines.append(f"Playing (x{speed:.1f})  |  +/- speed  |  R to reset")
    elif mode == "done":
        lines.append("Done  |  R to reset  |  1-8 presets")
    lines.append("1:Roll 2:1/2Roll 3:Stop 4:Stun 5:Draw 6:Follow 7:Lag 8:Baulk")
    for i, line in enumerate(lines):
        txt = font.render(line, True, COL_TEXT)
        surface.blit(txt, (10, WIN_H - 44 + i * 20))
