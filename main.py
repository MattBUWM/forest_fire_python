import noise
import numpy as np
import matplotlib.pyplot as plt

shape = (256, 256)
scale = 125
octaves = 6
persistence = 0.5
lacunarity = 2.0

forest = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        forest[i][j] = noise.pnoise2(i / scale,
                                     j / scale,
                                     octaves=octaves,
                                     persistence=persistence,
                                     lacunarity=lacunarity,
                                     repeatx=1024,
                                     repeaty=1024,
                                     base=0)
forest = (forest + 1)/2
forest = forest > 0.45
print(forest)
plt.imshow(forest)
plt.show()
