from math import *
import random

beta = 0.0
index = int(random.random() * 1000)  # set random index (0..999)

for i in range(1000):

        beta += random.random() * 2.0 * abs(random.gauss(0.0, 0.05)) * 3  # beta is the cumulated sum over all particles (1000) as a random double maximum weight

        w = abs(random.gauss(0.0, 0.05))

        while beta > w:       # while beta is greater than a concrete weight do
            beta -= w         # substract the concrete weight from beta and
            index = (index + 1) % 1000  #
            print ("index: ", index)