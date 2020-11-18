import torch
import numpy as np

dim = 3 # or 1
bins = 10


# form the n-cube with 10^n cells
probabilityDistribution = torch.ones(probabilityStarter, dtype = torch.int8)
zeroProbability = torch.ones(probabilityStarter, dtype = torch.int8)


# if lower than the treshold, I want to zero the probability
threshold = 0.5


# for 1 dim, I just need to re-arrange skewing the probabilities "to the right"
if dim == 1:
    for j in range(bins):
        step = (j + 0.5) / bins
        if step < threshold:
            probabilityDistribution[j] = 0
    
    probabilityDistribution = probabilityDistribution.float()
    probabilityDistribution = probabilityDistribution / torch.sum(probabilityDistribution)


# for n dim, I need to do n for loops and updante the d-th dimension [:, ..., d, :, ..., :]
elif dim == 3:
    for j in range(bins):
        step = (j + 0.5) / bins
        if step > threshold:
            zeroProbability[j, :, :] = 0 # for dim 1

    for j in range(bins):
        step = (j + 0.5) / bins
        if step > threshold:
            zeroProbability[:, j, :] = 0 # for dim 2

    for j in range(bins):
        step = (j + 0.5) / bins
        if step > threshold:
            zeroProbability[:, :, j] = 0 # for dim 3

    probabilityDistribution = probabilityDistribution - zeroProbability
    probabilityDistribution = probabilityDistribution.float()
    probabilityDistribution = probabilityDistribution / torch.sum(probabilityDistribution)

# print(probabilityDistribution)