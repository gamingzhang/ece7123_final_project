import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy
import pickle

from utils import *
from ec import *
from pruning import *


def evaIndividual(ind):
    filter_nums = [20, 50, 500, 10]
    solution = np.ones((sum(filter_nums), 1))
    solution = ind.reshape(ind.shape[0], 1)
    solution[-10:] = 1  # Last 10 output should not be changed

    # Prune model according to the solution
    model_new = prune_model(model, solution, filter_nums)
    # Validate
    acc, loss = test_forward(val_loader, model_new, criterion)  # test_forward(model_new)
    return 100 - acc, np.sum(ind)


class Individual():

    def __init__(self, gene_length):
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            self.dec[i] = 1  # always begin with 1
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        self.evaluate()

    def evaluate(self):
        self.obj[0], self.obj[1] = evaIndividual(self.dec)


def initialization(pop_size, gene_length):
    population = []
    for i in range(pop_size):
        ind = Individual(gene_length)
        population.append(ind)
    return population


# Global variables, avoid loading data at each generation
val_loader, model, criterion = construct_data_model_criterion()

filter_nums = [20, 50, 500, 10]

target_dir = 'Results/'

if __name__ == '__main__':
    # Configuration
    pop_size = 30  # Population size
    n_obj = 2  # Objective variable dimensionality

    dec_dim = sum(filter_nums)  # Decision variable dimensionality

    gen = 500  # Iteration number

    p_crossover = 1  # crossover probability
    p_mutation = 1  # mutation probability

    # Initialization
    population = initialization(pop_size, dec_dim)

    g_begin = 0

    path_save = './' + target_dir

    for g in range(g_begin + 1, gen):
        # generate reference lines and association
        V, association, ideal = generate_ref_association(population)

        # Variation
        offspring = variation(population, p_crossover, p_mutation)

        # Update ideal point
        PopObjs_Offspring = np.array([x.obj for x in offspring])
        PopObjs_Offspring = np.vstack((ideal, PopObjs_Offspring))
        ideal = np.min(PopObjs_Offspring, axis=0)

        # P+Q
        population.extend(offspring)

        # Environmental Selection
        population = environmental_selection(population, V, ideal, pop_size)

        # generation
        print('Gen:', g)

        # Save population
        with open(path_save + "population-{}.pkl".format(g), 'wb') as f:
            pickle.dump(population, f)