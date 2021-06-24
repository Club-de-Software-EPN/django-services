from pygame.transform import rotate


class PygameDrawer:

    STAT_FONT = None
    WINDOW_WIDTH = 0
    WINDOW_HEIGHT = 0

    def __init__(self, window, stat_font, window_width, window_height):
        self.window = window
        self.STAT_FONT = stat_font
        self.WINDOW_WIDTH = window_width
        self.WINDOW_HEIGHT = window_height

    def draw_window(self, humans, enegys, monsters, score, background_image, gen):
        self.window.blit(background_image, (0, 0))

        for energy in enegys:
            if not energy.collected:
                self.__drawEnergy(energy)

        text = self.STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
        self.window.blit(text, (self.WINDOW_WIDTH - 10 - text.get_width(), 10))

        text2 = self.STAT_FONT.render("Generation: " + str(gen), 1, (255, 255, 255))
        self.window.blit(text2, (10, 10))

        text3 = self.STAT_FONT.render("Population (genomes): " + str(len(humans)), 1, (255, 255, 255))
        self.window.blit(text3, (10, 40))

        # self.__drawBase(base)

        for human in humans:
            self.__drawHuman(human)

        for monster in monsters:
            self.__drawMonster(monster)


    def __drawHuman(self, human):
        # human.image_count = bird.image_count + 1
        # bird.chooseImageOfAnimation(bird.image_count)
        # new_rect = rotated_image.get_rect(center=human.image_actual.get_rect(topleft=(human.position.x, human.position.y)).center)
        self.window.blit(human.image_actual, (human.position.x, human.position.y))

    def __drawMonster(self, monster):
        self.window.blit(monster.image_actual, (monster.position.x, monster.position.y))

    def __drawEnergy(self, energy):
        self.window.blit(energy.image, (energy.position.x, energy.position.y))
