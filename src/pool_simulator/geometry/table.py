class Table:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def check_boundaries(self, position):
        x, y = position
        return 0 <= x <= self.length and 0 <= y <= self.width

    def get_corners(self):
        return [(0, 0), (self.length, 0), (self.length, self.width), (0, self.width)]

    def is_ball_on_table(self, ball_position):
        return self.check_boundaries(ball_position)