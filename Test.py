import random
from scipy.stats.qmc import LatinHypercube

dimension = 4
n_samples = 1000

sampler = LatinHypercube(d=dimension)
print(sampler)
samples = sampler.random(n=n_samples)
print(sampler)
