#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
from math import *
from scipy.spatial.distance import cdist


# Domination check
def dominate(p, q):
    result = False
    for i, j in zip(p.obj, q.obj):
        if i < j:  # at least less in one dimension
            result = True
        elif i > j:  # not greater in any dimension, return false immediately
            return False
    return result


def non_dominate_sorting(population):
    # find non-dominated sorted
    dominated_set = {}
    dominating_num = {}
    rank = {}
    for p in population:
        dominated_set[p] = []
        dominating_num[p] = 0

    sorted_pop = [[]]
    rank_init = 0
    for i, p in enumerate(population):
        for q in population[i + 1:]:
            if dominate(p, q):
                dominated_set[p].append(q)
                dominating_num[q] += 1
            elif dominate(q, p):
                dominating_num[p] += 1
                dominated_set[q].append(p)
        # rank 0
        if dominating_num[p] == 0:
            rank[p] = rank_init  # rank set to 0
            sorted_pop[0].append(p)

    while len(sorted_pop[rank_init]) > 0:
        current_front = []
        for ppp in sorted_pop[rank_init]:
            for qqq in dominated_set[ppp]:
                dominating_num[qqq] -= 1
                if dominating_num[qqq] == 0:
                    rank[qqq] = rank_init + 1
                    current_front.append(qqq)
        rank_init += 1

        sorted_pop.append(current_front)

    return sorted_pop


class Individual():

    def __init__(self, gene_length):
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            self.dec[i] = np.random.randint(2)  # random binary code
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        self.evaluate()

    def evaluate(self):
        # self.obj[0], self.obj[1] = evaCNN(self.dec)
        self.obj[0], self.obj[1] = 0, 0


def initialization(pop_size, gene_length):
    population = []
    for i in range(pop_size):
        ind = Individual(gene_length)
        population.append(ind)
    return population


def evaluation(population):
    # Evaluation
    [ind.evaluate() for ind in population]
    return population


# one point crossover
def one_point_crossover(p, q):
    gene_length = len(p.dec)
    child1 = np.zeros(gene_length, dtype=np.uint8)
    child2 = np.zeros(gene_length, dtype=np.uint8)
    k = np.random.randint(gene_length)
    child1[:k] = p.dec[:k]
    child1[k:] = q.dec[k:]

    child2[:k] = q.dec[:k]
    child2[k:] = p.dec[k:]

    return child1, child2


# Bit wise mutation
def bitwise_mutation(p, p_m):
    gene_length = len(p.dec)
    p_mutation = p_m / gene_length
    p_mutation = 0.01  ## constant mutation rate
    for i in range(gene_length):
        if np.random.random() < p_mutation:
            p.dec[i] = not p.dec[i]
    return p


# Variation (Crossover & Mutation)
def variation(population, p_crossover, p_mutation):
    offspring = copy.deepcopy(population)
    len_pop = int(np.ceil(len(population) / 2) * 2)
    candidate_idx = np.random.permutation(len_pop)

    # Crossover
    for i in range(int(len_pop / 2)):
        if np.random.random() <= p_crossover:
            individual1 = offspring[candidate_idx[i]]
            individual2 = offspring[candidate_idx[-i - 1]]
            [child1, child2] = one_point_crossover(individual1, individual2)
            offspring[candidate_idx[i]].dec[:] = child1
            offspring[candidate_idx[-i - 1]].dec[:] = child2

    # Mutation
    for i in range(len_pop):
        individual = offspring[i]
        offspring[i] = bitwise_mutation(individual, p_mutation)

    # Evaluate offspring
    offspring = evaluation(offspring)

    return offspring


# Crowding distance
def crowding_dist_old(population):
    pop_size = len(population)
    crowding_dis = np.zeros((pop_size, ))

    obj_dim_size = len(population[0].obj)
    # crowding distance
    for m in range(obj_dim_size):
        obj_current = [x.obj[m] for x in population]
        sorted_idx = np.argsort(obj_current)  # sort current dim with ascending order
        obj_max = np.max(obj_current)
        obj_min = np.min(obj_current)

        # keep boundary point
        crowding_dis[sorted_idx[0]] = np.inf
        crowding_dis[sorted_idx[-1]] = np.inf
        for i in range(1, pop_size - 1):
            crowding_dis[sorted_idx[i]] = crowding_dis[sorted_idx[i]] + \
                                                      1.0 * (obj_current[sorted_idx[i + 1]] - \
                                                             obj_current[sorted_idx[i - 1]]) / (obj_max - obj_min)
    return crowding_dis


def crowding_dist(population, V, ideal):
    PopObjs = np.array([x.obj for x in population])
    N = PopObjs.shape[0]
    nadir = np.max(PopObjs, axis=0)
    PopObjs = (PopObjs - ideal) / (nadir - ideal)
    Angle = np.arccos(1 - cdist(PopObjs, V, 'cosine'))

    Ranks = []

    mmd = np.sum(PopObjs, axis=1)
    mmd_rank = np.argsort(mmd)[::-1]  # descending ordering, small mmd large rank
    rank_temp = np.zeros((1, N))
    rank_temp[0, mmd_rank] = np.arange(N)
    Ranks.append(rank_temp)

    for i in range(V.shape[0]):
        rank = np.argsort(Angle[:, i])[::-1]  # descending order, small angle (large cosine) large rank
        rank_temp = np.zeros((1, N))
        rank_temp[0, rank] = np.arange(N)
        Ranks.append(rank_temp)

    cwd_dist = crowding_dist_old(population)
    cwd_dist[rank[-1]] = inf  # Set inf for the knee vector

    cwd_rank = np.argsort(cwd_dist)[::-1]  # descending order, large cwd large rank
    crowding_dst = np.max(np.squeeze(np.array(Ranks)), axis=0)
    crowding_dst[np.where(cwd_dist == inf)[0]] = inf
    crowding_dst = crowding_dst + cwd_rank

    return crowding_dst


def normalize(PopObjs):
    N = PopObjs.shape[0]
    nadir = np.max(PopObjs, 0)
    ideal = np.min(PopObjs, 0)
    PopObjs_norm = (PopObjs - ideal) / (nadir - ideal)
    return PopObjs_norm, ideal


def generate_ref_association(population):
    PopObjs = np.array([x.obj for x in population])
    N = PopObjs.shape[0]
    M = PopObjs.shape[1]
    V = np.eye(M)
    # Finding current knee
    PopObjs_norm, ideal = normalize(PopObjs)
    # MMD
    mmd = np.sum(PopObjs_norm, axis=1)
    knee_idx = np.argmin(mmd)
    # Finding the knee vector
    KneeVector = PopObjs_norm[knee_idx, :]

    V = np.vstack((V, KneeVector))

    Angle = np.arccos(1 - cdist(PopObjs_norm, V, 'cosine'))
    association = np.argmin(Angle, axis=1)

    return V, association, ideal


def environmental_selection(population, V, ideal, n):
    pop_sorted = non_dominate_sorting(population)
    selected = []
    for front in pop_sorted:
        if len(selected) < n:
            if len(selected) + len(front) <= n:
                selected.extend(front)
            else:
                # select individuals according crowding distance here
                crowding_dst = crowding_dist(front, V, ideal)
                k = n - len(selected)
                dist_idx = np.argsort(crowding_dst, axis=0)[::-1]  # descending order, large rank small angel
                for i in dist_idx[:k]:
                    selected.extend([front[i]])
                break
    return selected