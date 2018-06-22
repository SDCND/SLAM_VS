from math import *

def f(mu, sigma2, x):
    return 1/sqrt(2. *pi*sigma2) * exp(-.5 * (x-mu) ** 2 / sigma2)

# u se the formula as we did manually above
print f(10., 4., 10.)




