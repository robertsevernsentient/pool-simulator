from dataclasses import dataclass

from engine.physics.ball_state import BallState


@dataclass
class SimulationState:
    balls: list[BallState]
    time: float