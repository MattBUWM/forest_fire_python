import random

import noise
import numpy as np
import matplotlib.pyplot as plt


def generate_perlin_noise(shape, scale, octaves, persistence, lacunarity):
    perl_noise = np.zeros(shape)
    random_int = random.randint(0, 255)
    for i in range(shape[0]):
        for j in range(shape[1]):
            perl_noise[i][j] = noise.pnoise2(i / scale,
                                             j / scale,
                                             octaves=octaves,
                                             persistence=persistence,
                                             lacunarity=lacunarity,
                                             repeatx=1024,
                                             repeaty=1024,
                                             base=random_int
                                             )
    return perl_noise


def generate_random_normal_vector():
    


class ForestFireSimulation:
    def __init__(self, shape, cutout, p, ps, k, moore=False, scale=75, octaves=5, persistence=0.5, lacunarity=1.5):
        self.shape = shape
        self.p = p
        self.ps = ps
        self.k = k
        self.neighbourhood_moore = moore
        perl_noise = generate_perlin_noise(shape, scale, octaves, persistence, lacunarity)
        perl_noise = perl_noise > cutout
        self.forest = np.array(perl_noise, int)
        self.timers = np.zeros((self.shape[0], self.shape[1]), int)

    def display_current_state(self):
        image = np.zeros((self.shape[0], self.shape[1], 3), int)
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                value = self.forest[x, y]
                if value == 0:
                    image[x, y] = [0, 0, 200]
                elif value == 1:
                    image[x, y] = [0, 200, 0]
                elif value == 2:
                    image[x, y] = [250, 30, 30]
                elif value == 3:
                    image[x, y] = [75, 60, 35]

        plt.imshow(image)
        plt.show()

    def get_tiles_of_value(self, value):
        tiles = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.forest[x, y] == value:
                    tiles.append((x,y))
        return tiles

    def start_fire(self):
        tree_tiles = self.get_tiles_of_value(1)
        if len(tree_tiles) > 0:
            tile = random.choice(tree_tiles)
            self.forest[tile[0], tile[1]] = 2

    def get_neighbour_tiles(self, x, y):
        tiles = []
        if self.neighbourhood_moore:
            neighbours_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            neighbours_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for offset in neighbours_offsets:
            neighbour_x = x - offset[0]
            neighbour_y = y - offset[1]
            if neighbour_x in range(self.shape[0]) and neighbour_y in range(self.shape[1]):
                tiles.append((neighbour_x, neighbour_y))
        return tiles

    def iteration_tick(self):
        new_state = np.zeros((self.shape[0], self.shape[1]), int)
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                value = self.forest[x, y]
                if value == 0:
                    new_state[x, y] = 0
                elif value == 1:
                    neighbours = self.get_neighbour_tiles(x, y)
                    chance = self.ps
                    for neighbour in neighbours:
                        if self.forest[neighbour[0], neighbour[1]] == 2:
                            chance = chance + self.p
                    if random.random() < chance:
                        new_state[x, y] = 2
                    else:
                        new_state[x, y] = 1
                elif value == 2:
                    new_state[x, y] = 3
                    self.timers[x, y] = self.k
                elif value == 3:
                    self.timers[x, y] = self.timers[x, y] - 1
                    if self.timers[x, y] <= 0:
                        new_state[x, y] = 1
                    else:
                        new_state[x, y] = 3
        self.forest = new_state


if __name__ == '__main__':
    simulation = ForestFireSimulation((255, 255), -0.1, 0.3, 0.00001,  50, True)
    simulation.start_fire()
    simulation.display_current_state()
    for _ in range(100):
        simulation.iteration_tick()
        simulation.display_current_state()

