def calculate_velocity(initial_velocity, acceleration, time):
    return initial_velocity + acceleration * time

def update_position(initial_position, velocity, time):
    return initial_position + velocity * time