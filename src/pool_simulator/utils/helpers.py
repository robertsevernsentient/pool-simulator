def convert_coordinates(x, y, table_width, table_height):
    """Convert (x, y) coordinates to a different reference frame."""
    return (x / table_width, y / table_height)

def format_output(ball_position, ball_velocity):
    """Format the output for displaying ball state."""
    return f"Position: {ball_position}, Velocity: {ball_velocity}"