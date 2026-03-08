def check_collision(ball1, ball2):
    distance = ((ball1.position[0] - ball2.position[0]) ** 2 + (ball1.position[1] - ball2.position[1]) ** 2) ** 0.5
    return distance < (ball1.radius + ball2.radius)

def resolve_collision(ball1, ball2):
    normal = (ball2.position[0] - ball1.position[0], ball2.position[1] - ball1.position[1])
    distance = ((normal[0] ** 2 + normal[1] ** 2) ** 0.5)
    
    if distance == 0:
        return
    
    normal = (normal[0] / distance, normal[1] / distance)
    
    relative_velocity = (ball2.velocity[0] - ball1.velocity[0], ball2.velocity[1] - ball1.velocity[1])
    velocity_along_normal = (relative_velocity[0] * normal[0] + relative_velocity[1] * normal[1])
    
    if velocity_along_normal > 0:
        return
    
    restitution = 0.8  # Coefficient of restitution
    impulse_strength = -(1 + restitution) * velocity_along_normal
    
    ball1.velocity = (ball1.velocity[0] - impulse_strength * normal[0], ball1.velocity[1] - impulse_strength * normal[1])
    ball2.velocity = (ball2.velocity[0] + impulse_strength * normal[0], ball2.velocity[1] + impulse_strength * normal[1])