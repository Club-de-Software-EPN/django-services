from os import DirEntry
from .Position import Position


class Human:

    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3

    ROTATION_MAX = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5

    images = {
        "BIRD_WINGS_UP": None,
        "BIRD_WINGS_MIDDLE": None,
        "BIRD_WINGS_DOWN": None
    }

    def __init__(self, initial_position_x, initial_position_y):
        self.position = Position(initial_position_x, initial_position_y)
        self.velocity = 16
        self.image_count = 0
        self.image_actual = Human.images["BIRD_WINGS_MIDDLE"]
        self.direction = -1
        self.tick_count = 1
        self.last_action = -1
        self.actual_action = -1
        self.initial_distance = 500
        self.initial_distance_monster = 100
        self.life = 3

    def action(self, action: int):
        # direction 0=up,1=right,2=down,3=left
        # self.velocity = quantity

        self.tick_count += 0.2     

        if action in range(0, 4):
            self.direction = action
        else:
            self.direction = 0

        move_pixels = self.velocity  # * self.tick_count + 1.5 * self.tick_count**2

        # clamp the value
        if move_pixels >= 16:
            move_pixels = 16

        # TO-DO: add animations

        if self.direction == Human.DIRECTION_UP:
            # in pygame negative y is UP
            self.position.y -= move_pixels
            self.image_actual = Human.images['walk_up']
        elif self.direction == Human.DIRECTION_RIGHT:
            self.position.x += move_pixels
            self.image_actual = Human.images['walk_right']
        elif self.direction == Human.DIRECTION_DOWN:
            self.position.y += move_pixels
            self.image_actual = Human.images['walk_down']
        elif self.direction == Human.DIRECTION_LEFT:
            self.position.x -= move_pixels
            self.image_actual = Human.images['walk_left']

        self.last_action = self.actual_action
        self.actual_action = action


    def is_looping(self):


        if self.actual_action == Human.DIRECTION_UP and self.last_action == Human.DIRECTION_DOWN:
            return True
        elif self.actual_action == Human.DIRECTION_DOWN and self.last_action == Human.DIRECTION_UP:
            return True
        elif self.actual_action == Human.DIRECTION_RIGHT and self.last_action == Human.DIRECTION_LEFT:
            return True
        elif self.actual_action == Human.DIRECTION_LEFT and self.last_action == Human.DIRECTION_RIGHT:
            return True

        return False
            


    def isMovingUpwards(self, move_down_pixels):
        return move_down_pixels < 0 or self.position.y < self.height + 50

    def rotateUpwards(self):
        if self.tilt < self.ROTATION_MAX:
            self.tilt = self.ROTATION_MAX

    def rotateDownwards(self):
        self.tilt = self.tilt - self.ROTATION_VELOCITY

    def isOutsideScreen(self, screen_height, screen_width):
        came_out_below = self.position.y + self.image_actual.get_height() >= screen_height
        came_out_above = self.position.y < 0
        came_out_left = self.position.x < 0
        came_out_right = self.position.x + self.image_actual.get_width() >= screen_width
        
        return came_out_above or came_out_below or came_out_left or came_out_right
