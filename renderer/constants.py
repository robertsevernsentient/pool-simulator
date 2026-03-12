from engine.physics.ball_state import MotionState
from engine.physics.tuneable_constants import BALL_RADIUS, STANDARD_9_FOOT

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
