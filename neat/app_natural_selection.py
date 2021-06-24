import pygame
import neat
import os
from modules.PygameDrawerNaturalSelection import PygameDrawer
from modules.models.Energy import Energy
from modules.models.Human import Human
from modules.models.Monster import Monster
from modules.models.Base import Base
from modules.ParametersNaturalSelection import Parameters
import math
import random

def getPygameImageFromPath(image_name):
    return pygame.transform.scale2x(pygame.image.load(os.path.join("static", "neat", "img", image_name)))


def getPygameMask(image):
    return pygame.mask.from_surface(image)


def flipImage(image, xbool, ybool):
    return pygame.transform.flip(image, xbool, ybool)


GENERATION = 0
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)

pygame_window = pygame.display.set_mode((Parameters.WINDOW_WIDTH, Parameters.WINDOW_HEIGHT))
drawer = PygameDrawer(pygame_window, STAT_FONT, Parameters.WINDOW_WIDTH, Parameters.WINDOW_HEIGHT)
clock = pygame.time.Clock()



# Section: Transforming to pygame images

BACKGROUND_PYGAME_IMAGE = [getPygameImageFromPath(Parameters.BACKGROUND_IMAGE)]

for key, url_image in Parameters.HUMAN_IMAGES.items():

    url = os.path.join("humans", url_image)
    print(url)

    Human.images[key] = getPygameImageFromPath(url)

Human.images["idle_left"] = flipImage(Human.images["idle_right"], True, False)
Human.images["walk_left"] = flipImage(Human.images["walk_right"], True, False)


for key, url_image in Parameters.MONSTER_IMAGES.items():

    url = os.path.join("monsters", url_image)
    print(url)

    Monster.images[key] = getPygameImageFromPath(url)

Monster.images["idle_left"] = flipImage(Monster.images["idle_right"], True, False)
Monster.images["walk_left"] = flipImage(Monster.images["walk_right"], True, False)




OTHER_PYGAME_IMAGES = [getPygameImageFromPath(image).convert_alpha() for image in Parameters.OTHER_IMAGES]

PYGAME_IMAGE_GOOD_ENERGY = OTHER_PYGAME_IMAGES[0]




def existCollition(human, other):
    human_mask = getPygameMask(human.image_actual)
    other_mask = getPygameMask(other.image)
    

    offset = (other.position.x - human.position.x, other.position.y - human.position.y)

    point = human_mask.overlap(other_mask, offset)

    return point


def findPipeToAnalize(birds, pipes):
    if len(pipes) > 1 and birds[0].position.x > pipes[0].position.x + pipes[0].image_top.get_width():
        return 1
    else:
        return 0


def distance(a, b):
    return abs(a - b)

def real_distance(a, b):
    return a - b


def fitnessFunction(genomes, config):
    global GENERATION
    GENERATION += 1
    score = 0

    feed_forward_networks = []
    genomes_copy = []  # with initial fitness equal to 0
    humans = []

    monsters = [Monster(100, 100), Monster(Parameters.WINDOW_WIDTH-100, Parameters.WINDOW_HEIGHT-100)]


    REWARD = 0.1

    for _, genome in genomes:
        # genome = individual = human
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        feed_forward_networks.append(net)
        humans.append(Human(Parameters.WINDOW_WIDTH//2, Parameters.WINDOW_HEIGHT//2))
        genome.fitness = 10  # generally this value is 0, here initial value of 1 to survive
        genomes_copy.append(genome)

    MAX_GOOD_ENERGY = 1

    good_energys = [Energy(random.randint(1, Parameters.WINDOW_WIDTH-70) , random.randint(100, Parameters.WINDOW_HEIGHT-70), PYGAME_IMAGE_GOOD_ENERGY) for _ in range(MAX_GOOD_ENERGY)]

    game_running = True

    passed = [-1, -1, -1, -1]

    GAP = 120
    GAP_MIN = 100


    iterations = 0
    while game_running:

        if score >= 50:
            clock.tick(30)
        


        # PARA AUMENTAR LA VELOCIDAD COMENTAR ESTO
        # clock.tick(30)  # 30 ticks every second

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
                pygame.quit()
                quit()

        if len(humans) == 0:
            game_running = False
            continue

        for monster in monsters:

            monster.action(random.randint(0, 3))

            while monster.is_looping():
                monster.action(random.randint(0, 3))

            """
            if monster.position.x >= GAP_MIN and monster.position.x < Parameters.WINDOW_WIDTH - GAP and passed[0] == -1:
                monster.action(1)
            elif monster.position.y >= GAP_MIN and  monster.position.y < Parameters.WINDOW_HEIGHT - GAP and passed[1] == -1:
                passed[0] = 1
                monster.action(2)
            elif monster.position.x <= Parameters.WINDOW_WIDTH - GAP_MIN and monster.position.x >= GAP and passed[2] == -1:
                passed[1] = 1
                monster.action(3)
            elif monster.position.y <= Parameters.WINDOW_HEIGHT - GAP_MIN and  monster.position.y >= GAP and passed[3] == -1:
                passed[2] = 1
                monster.action(0)
            else:
                # reset
                passed[0] = -1
                passed[1] = -1
                passed[2] = -1
            """
        
        



        # action for the frame
        for index, human in enumerate(humans):

            # each network of each human
            # input (posy, distance_to_pipe_top, distance_to_pipe_botton)
            # output (up, rigth, down, left)

            


            

            monsters

            output = feed_forward_networks[index].activate((
                human.position.x,
                human.position.y,
                real_distance(human.position.x, good_energys[0].position.x),
                real_distance(human.position.y, good_energys[0].position.y),
                real_distance(human.position.x, monsters[0].position.x),
                real_distance(human.position.y, monsters[0].position.y),
                real_distance(human.position.x, monsters[1].position.x),
                real_distance(human.position.y, monsters[1].position.y)
            ))

            max = -math.inf
            max_index = 0
            for index, out in enumerate(output):
                if out > max:
                    max = out
                    max_index = index

            # print(human.position.x, human.position.y)
            # print(max_index)

            # print(monster.position.x, monster.position.y)
            

            
            human.initial_distance = distance(human.position.x, good_energys[0].position.x) + distance(human.position.y, good_energys[0].position.y)
            human.initial_distance_monster = distance(human.position.x, monster.position.x) + distance(human.position.y, monster.position.y)
            human.action(max_index)
            # print(human.position.x, human.position.y)
            


            
        # scoring the action

        pending_good_energy_remove = []
        for index, human in enumerate(humans):

            # should move
            # genomes_copy[index].fitness -= REWARD
            
            

            for good in good_energys:

                if existCollition(human, good) and not good.collected:
                    score += 1
                    human.life = 10
                    genomes_copy[index].fitness += 10
                    good.collected = True
                    pending_good_energy_remove.append(good)
                    human.tick_count = 0  # reset multiply penalization  


            for monster in monsters:
                if existCollition(human, monster):
                    genomes_copy[index].fitness -= REWARD * 2                       


            new_distance = distance(human.position.x, good_energys[0].position.x) + distance(human.position.y, good_energys[0].position.y)
            new_distance_monster = distance(human.position.x, monster.position.x) + distance(human.position.y, monster.position.y)

            

            if new_distance < human.initial_distance:
                genomes_copy[index].fitness += REWARD * 2
            else:
                genomes_copy[index].fitness -= REWARD

            """
            if new_distance_monster < human.initial_distance_monster:
                genomes_copy[index].fitness -= REWARD
            else:
                genomes_copy[index].fitness += REWARD
            """
            
            if human.is_looping():
                genomes_copy[index].fitness -= (REWARD * 10)
                human.life -= 1
                if human.life <= 0:
                    humans.pop(index)
                    feed_forward_networks.pop(index)
                    genomes_copy.pop(index)     


            elif human.isOutsideScreen(Parameters.WINDOW_HEIGHT, Parameters.WINDOW_WIDTH):
                genomes_copy[index].fitness -= REWARD * 10
                # dead
                humans.pop(index)
                feed_forward_networks.pop(index)
                genomes_copy.pop(index)


        


        for pending_remove in pending_good_energy_remove:
            good_energys.remove(pending_remove)

        if len(good_energys) < MAX_GOOD_ENERGY:
            # always generating
            difference = MAX_GOOD_ENERGY - len(good_energys)

            for _ in range(difference):
                good_energys.append(Energy(random.randint(1, Parameters.WINDOW_WIDTH-70) , random.randint(100, Parameters.WINDOW_HEIGHT-70), PYGAME_IMAGE_GOOD_ENERGY))

        
        iterations += 1

        #if iterations >= MAX_ITERATIONS:
        #    game_running = False


        



        drawer.draw_window(humans, good_energys, monsters, score, BACKGROUND_PYGAME_IMAGE[0], GENERATION)
        pygame.display.update()   



if __name__ == '__main__':
    """
    runs the NEAT algorithm to train a neural network to play a game
    """
    local_directory = os.path.dirname(__file__)
    configuration_neat_path = os.path.join(local_directory, "data", "configuration_neat_natural_selection.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                configuration_neat_path)

    population = neat.Population(config)
    number_generations = 2000

    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # population.add_reporter(neat.Checkpointer(5))

    winner = population.run(fitnessFunction, number_generations)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # break if score gets large enough
    '''if score > 20:
        pickle.dump(nets[0],open("best.pickle", "wb"))
        break'''

