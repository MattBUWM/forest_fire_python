import random
import noise
import numpy as np
import cv2


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


def get_vector_from_angle(angle):
    angle_rad = np.deg2rad(angle)
    sin = np.sin(angle_rad)
    cos = np.cos(angle_rad)
    return np.array([sin, cos])


def calculate_angle(x, y):
    unit_x = x / np.linalg.norm(x)
    unit_y = y / np.linalg.norm(y)
    angle_rad = np.arccos(np.dot(unit_x, unit_y))
    return np.rad2deg(angle_rad)


class ForestFireSimulation:
    def __init__(self, shape, cutout, p, ps, fk, rk, moore=False, scale=75, octaves=5, persistence=0.5, lacunarity=1.5, rescale=None):
        self.shape = shape
        self.rescale = rescale
        self.p = p
        self.ps = ps
        self.fk = fk
        self.rk = rk
        self.neighbourhood_moore = moore
        wind_direction = random.random()
        self.wind_direction_degrees = wind_direction * 360
        print(self.wind_direction_degrees)
        self.wind_direction_vector = get_vector_from_angle(self.wind_direction_degrees)
        print(self.wind_direction_vector)
        self.wind_intensity = random.random()
        print(self.wind_intensity)
        perl_noise = generate_perlin_noise(shape, scale, octaves, persistence, lacunarity)
        perl_noise = perl_noise > cutout
        self.forest = np.array(perl_noise, int)
        self.timers = np.zeros((self.shape[0], self.shape[1]), int)

    def display_current_state(self):
        image = np.zeros((self.shape[0], self.shape[1], 3), np.uint8)
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
        if self.rescale is None:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            rescaled = cv2.resize(image, (self.shape[0] * self.rescale, self.shape[1] * self.rescale), interpolation=cv2.INTER_NEAREST)
            return cv2.cvtColor(rescaled, cv2.COLOR_RGB2BGR)


    def display_wind_direction(self):
        wind_image = np.ones((340, 320, 3), np.uint8) * 255
        arrow = cv2.imread('arrow.jpg')
        wind_image[110: 230, :] = arrow
        wind_image = cv2.bitwise_not(wind_image)
        matrix = cv2.getRotationMatrix2D((320 / 2, 340 / 2), self.wind_direction_degrees, 1)
        wind_image = cv2.warpAffine(wind_image, matrix, (320, 340))
        wind_image = cv2.bitwise_not(wind_image)
        cv2.putText(wind_image, f'Intensity:{round(self.wind_intensity, 5)}', (2, 337), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
        cv2.imshow('Wind', wind_image)

    def get_tiles_of_value(self, value):
        tiles = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.forest[x, y] == value:
                    tiles.append((x, y))
        return tiles

    def start_fire(self):
        tree_tiles = self.get_tiles_of_value(1)
        if len(tree_tiles) > 0:
            tile = random.choice(tree_tiles)
            self.forest[tile[0], tile[1]] = 2
            self.timers[tile[0], tile[1]] = self.fk

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
                            offset = np.array([x - neighbour[0], y - neighbour[1]])
                            offset_norm = np.linalg.norm(offset)
                            relative_angle = calculate_angle(offset, self.wind_direction_vector)
                            wind_influence = self.p * (self.wind_intensity / offset_norm) * ((relative_angle - 90) / 180)
                            if wind_influence < -self.p:
                                wind_influence = -self.p
                            chance = chance + self.p + wind_influence
                    if random.random() < chance:
                        new_state[x, y] = 2
                        self.timers[x, y] = self.fk
                    else:
                        new_state[x, y] = 1
                elif value == 2:
                    self.timers[x, y] = self.timers[x, y] - 1
                    if self.timers[x, y] <= 0:
                        new_state[x, y] = 3
                        self.timers[x, y] = self.rk
                    else:
                        new_state[x, y] = 2
                elif value == 3:
                    self.timers[x, y] = self.timers[x, y] - 1
                    if self.timers[x, y] <= 0:
                        new_state[x, y] = 1
                    else:
                        new_state[x, y] = 3
        self.forest = new_state


if __name__ == '__main__':
    simulation = ForestFireSimulation((128, 128), -0.1, 0.1, 0.000001, 3, 100, True, rescale=4)
    simulation.start_fire()
    simulation.display_wind_direction()
    image = simulation.display_current_state()
    cv2.imshow('Simulation', image)
    cv2.waitKey(200)
    for _ in range(1000):
        simulation.iteration_tick()
        image = simulation.display_current_state()
        cv2.imshow('Simulation', image)

        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
