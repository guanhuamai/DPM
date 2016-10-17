import numpy as np

class UniformWorkers(object):
    def __init__(self, numWorkers):
        self.behaviour = 'Uniform'
        self.AccMatrix = [] # 2 * 2 matrix, row: bonus, not bonus, column: type 0, type 1
        self.AccMatrix.append([0.5, 0.8]) #accuracy not  given bonus
        self.AccMatrix.append([0.9, 0.9]) #accuracy when given bonus
        self.workersArray = np.choice(2, numWorkers)

    def work(self, workerIDs, isBonus):
        probs = [self.AccMatrix[isBonus[i]][workerIDs[i]] for i in range(len(workerIDs))]
        return [np.choice(2, 1, [1- prob, prob]) for prob in probs]

class BetaWorkers(object):
    def __init__(self, numWorkers):
        self.behaviour = 'Beta'
        self.AccMatrix = [] #AccMatrix[0/1] is a function that will generate a probability of generating right answers
        self.AccMatrix.append(lambda x: np.Beta(2, 2)) #accuracy not  given bonus
        self.AccMatrix.append(lambda x: np.Beta(6, 2)) #accuracy when given bonus

    def work(self, workerIDs, isBonus):
        probs = [self.AccMatrix[isBonus[i]]() for i in range(len(workerIDs))]
        return np.choice(len(workerIDs), 1, probs)


class IOHmmWorkers(object):
    def __init__(self, numWorkers):
        self.behaviour = 'iohmm'
        self.AccMatrix = []
        