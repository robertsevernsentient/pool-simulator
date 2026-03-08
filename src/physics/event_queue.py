import heapq

class EventQueue:

    def __init__(self):
        self.q = []

    def push(self, event):
        heapq.heappush(self.q, event)

    def pop(self):
        return heapq.heappop(self.q)

    def empty(self):
        return len(self.q) == 0