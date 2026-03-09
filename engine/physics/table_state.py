from dataclasses import dataclass


@dataclass
class Table:
    width: float
    height: float
    rail_restitution: float