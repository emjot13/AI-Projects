import math
import random as rd
import numpy as np


def vectors(v1: list, v2: list):
    if len(v1) != len(v2):
        raise ValueError("The vectors need to have the same length")
    added = []
    product = []
    for x in range(len(v1)):
        added.append(v1[x] + v2[x])
        product.append(v1[x] * v2[x])

    return added, product


def scalar(v1: list, v2: list):
    return sum(vectors(v1, v2)[1])


def euclidanLength(v1: list, v2: list):
    result_v1 = 0
    result_v2 = 0
    for index in range(len(v1)):
        result_v1 += v1[index] ** 2
    for index in range(len(v2)):
        result_v2 += v2[index] ** 2

    return f"Vector one length: {math.sqrt(result_v1)}\nVector two length: {math.sqrt(result_v2)}"


randomVector = [rd.randint(1, 101) for x in range(50)]


def vectorParameters():
    return f"Average: {np.average(randomVector)}, min value: {min(randomVector)}, max value: {max(randomVector)}, standard deviation: {np.std(randomVector)} "


def normalization():
    outcome = []
    for index in range(len(randomVector)):
        outcome.append((randomVector[index] - min(randomVector)) / (max(randomVector) - min(randomVector)))
    return f"{outcome},\n Max value originally: {max(randomVector)}, index: {randomVector.index(max(randomVector))}, " \
           f"max value after normalization: {max(outcome)}, index: {outcome.index(max(outcome))} "


def standarization():
    outcome = []
    mean = np.mean(randomVector)
    std = np.std(randomVector)
    for num in randomVector:
        outcome.append((num - mean)/std)
    return outcome, np.mean(outcome), np.std(outcome)


def discrete():
    outcome = []
    print(randomVector)
    for num in randomVector:
        outcome.append(f"[{math.floor(num//10)*10}, {math.floor(num//10)*10+10})")
    for index, item in enumerate(outcome):
        if item[-3:-1] == "00":  
            outcome[index] = item[:-1] + "]"
    return outcome





print(vectors([5, 6, 7], [2, 1, 0]))
print(scalar([3, 4, 1], [1, 0, 8]))
print(euclidanLength([2, 4, 6], [1, 2, 2]))
print(vectorParameters())
print(normalization())
print(discrete())