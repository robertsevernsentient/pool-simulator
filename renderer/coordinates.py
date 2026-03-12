import numpy as np

from engine.physics.tuneable_constants import BALL_RADIUS
from renderer.constants import MARGIN, SCALE, TABLE


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
