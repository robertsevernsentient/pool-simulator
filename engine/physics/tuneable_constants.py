from engine.physics.table_state import Table


G = 9.81

# cloth friction (sliding)
MU_SLIDE = 0.20

# rolling resistance
MU_ROLL = 0.03

# rail restitution
RAIL_RESTITUTION = 0.82

BALL_RADIUS = 0.028575      # meters
BALL_MASS = 0.17

# Maximum omega per unit speed from a cue strike.
# Derived from striking at max offset (R/2) before miscue:
#   omega = 5*v*h / (2*R²), h_max = R/2  →  omega_max = 5*v / (4*R)
MAX_CUE_SPIN = 5.0 / (4.0 * BALL_RADIUS)   # ≈ 43.7 rad/s per m/s

STANDARD_9_FOOT = Table(
    width = 2.84,
    height = 1.42,
    rail_restitution = RAIL_RESTITUTION
)