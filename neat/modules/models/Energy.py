from .Position import Position
import random


class Energy:


    def __init__(self, initial_position_x, initial_position_y, image):
        self.position = Position(initial_position_x, initial_position_y)
        self.collected = False  # collision purpose
        self.image = image
