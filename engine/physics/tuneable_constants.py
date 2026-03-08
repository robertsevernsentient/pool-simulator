from engine.physics.table_state import Table


G = 9.81

# cloth friction (sliding)
MU_SLIDE = 0.20

# rolling resistance
MU_ROLL = 0.01

# spin decay
SPIN_FRICTION = 8.0

# rail restitution
RAIL_RESTITUTION = 0.9

BALL_RADIUS = 0.028575      # meters
BALL_MASS = 0.17

STANDARD_9_FOOT = Table(
    width = 2.84,
    height = 1.42,
    rail_restitution = RAIL_RESTITUTION,
    cloth_friction = MU_SLIDE
)