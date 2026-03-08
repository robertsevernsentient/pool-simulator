from dataclasses import dataclass

from src.physics.ball_state import BallState


@dataclass
class SimulationState:
    balls: list[BallState]
    time: float