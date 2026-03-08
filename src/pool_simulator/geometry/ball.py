class Ball:
    def __init__(self, position, velocity, radius=0.05715):
        self.position = position  # Position should be a tuple (x, y)
        self.velocity = velocity  # Velocity should be a tuple (vx, vy)
        self.radius = radius  # Default radius for a pool ball

    def update(self, time_delta):
        self.position = (
            self.position[0] + self.velocity[0] * time_delta,
            self.position[1] + self.velocity[1] * time_delta
        )

    def apply_force(self, force, time_delta):
        # Assuming force is a tuple (fx, fy)
        acceleration = (force[0], force[1])  # Simplified, mass is assumed to be 1
        self.velocity = (
            self.velocity[0] + acceleration[0] * time_delta,
            self.velocity[1] + acceleration[1] * time_delta
        )