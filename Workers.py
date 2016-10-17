import numpy as np

class UniformWorkers(object):

    def __init__(self, numWorkers):
        self.behaviour = 'Uniform'
        self.accMatrix = [] # 2 * 2 accurate matrix, row: bonus, not bonus, column: type 0, type 1
        self.accMatrix.append([0.5, 0.8]) #accuracy not  given bonus
        self.accMatrix.append([0.9, 0.9]) #accuracy when given bonus
        self.workersArray = np.random.choice(2, numWorkers)

    def work(self, workerIDs, bonus):
        bonus = map(lambda x: (x != 0), bonus) #convert value to bool
        probs = [self.accMatrix[bonus[i]][workerIDs[i]] for i in range(len(workerIDs))]
        return [np.random.choice(2, 1, p = [1- prob, prob]) for prob in probs]



class BetaWorkers(object):

    def __init__(self, numWorkers):
        self.behaviour = 'Beta'
        self.accMatrix = [] #accMatrix[0/1] is a function that generates probabilities of correct answer
        self.accMatrix.append(lambda: np.Beta(2, 2)) #accuracy not  given bonus
        self.accMatrix.append(lambda: np.Beta(6, 2)) #accuracy when given bonus

    def work(self, workerIDs, bonus):
        bonus = map(lambda x: (x != 0), bonus) #convert value to bool
        probs = [self.accMatrix[bonus[i]]() for i in range(len(workerIDs))]
        return [np.random.choice(2, 1, p=[1 - prob, prob]) for prob in probs]



class IOHmmWorkers(object):

    def initTransitMatx(self):
        self.transition = []
        #transition matrix when given bonus
        self.transition.append([[0.8, 0.15, 0.05], [0.3, 0.6, 0.1], [0.2, 0.4, 0.4]])
        #transition matrix not  given bonus
        self.transition.append([[0.4, 0.4, 0.2], [0.1, 0.5, 0.4], [0.05, 0.1, 0.85]])

    def __init__(self, numWorkers):
        self.behaviour = 'iohmm'
        self.initTransitMatx()
        self.r = [0.2, 0.6, 1.2]
        self.alph = [0, 1, 3]
        self.pAlph = [0.2, 0.6, 0.2]
        self.beta = [0, 1, 3]
        self.pBeta = [0.2, 0.6, 0.2]

        npChoice = lambda x: np.random.choice(3, numWorkers, x)
        self.situations = zip(npChoice(self.pAlph), npChoice(self.pBeta)) #save alpha and beta parameter of a worker
        self.z = np.random.choice(3, numWorkers)

    def _updateState(self, workerIDs, bonus): #update hidden state for all workers
        newZ = [self.z[x] for x in workerIDs]
        map(lambda x: np.random.choice(3, 1, self.transition[bonus][x]), newZ)
        for i in range(len(workerIDs)):
            self.z[workerIDs[i]] = newZ[i]

    def work(self, workerIDs, bonus):
        # emission probabilitie: 1 / (1 + e ^ (-alphi - betai(at - rzt))  "bonus or not 4.2"
        probs = [1 / (1 + np.exp(-self.alph[self.situations[i][0]] -\
            self.beta[self.situations[i][1]](bonus[i] - self.r[self.z[i]])))\
            for i in range(workerIDs)]
        self._updateState(bonus) #update hidden state
        return [np.random.choice(2, 1, p=[1 - prob, prob]) for prob in probs]
