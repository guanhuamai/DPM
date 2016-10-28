import numpy as np


class UniformWorkers(object):
    """
    initialize the worker model in uniform distribution.
        # model parameter:
        #  accMatrix[i][j]: accuracy of j typed worker when bonus is i(0:no bonus, 1:bonus)
        # input:
        #  num_workers: total number workers
    """
    def __init__(self, num_workers):
        self.behaviour = 'uniform'
        self.accMatrix = []  # 2 * 2 accurate matrix, row: bonus, not bonus, column: type 0, type 1
        self.accMatrix.append([0.5, 0.8])  # accuracy not  given bonus
        self.accMatrix.append([0.9, 0.9])  # accuracy when given bonus
        self.workerTypes = np.random.choice(2, num_workers)

    """
    assign a question to the workers in 'workerIDs'
        # input:
        #  workerIDs: a list of workers' id, they will work on the question
        #  bonus: a list of bonus with same size as workerIDs, indicate the bonus given to the workers
        # output:
        #  0/1 list with same size as workerIDs, indicates their working result is correct(with 1) or not(with 0)
    """
    def work(self, worker_ids, bns):
        bns = map(lambda x: (x != 0) * 1, bns)  # convert value to bool
        probs = [self.accMatrix[bns[i]][self.workerTypes[worker_ids[i]]] for i in range(len(worker_ids))]
        return [np.random.choice(2, 1, p=[1 - prob, prob])[0] for prob in probs]  # return 0:low quality, 1:high quality


class BetaWorkers(object):
    """
    initialize the worker model in beta distribution.
        # model parameter:
        #  accMatrix[i]: probability that
        # input:
        #  numWorkers: total number workers
    """
    def __init__(self, num_workers):
        self.behaviour = 'beta'
        self.accMatrix = zip(np.random.beta(2, 2, num_workers), np.random.beta(6, 2, num_workers))

    """
    assign a question to the workers in 'workerIDs'
        # input:
        #  workerIDs: a list of workers' id, they will work on the question
        #  bonus: a list of bonus with same size as workerIDs, indicate the bonus given to the workers
        # output:
        #  0/1 list with same size as workerIDs, indicates their working result is correct(with 1) or not(with 0)
    """
    def work(self, worker_ids, bns):
        bns = map(lambda x: (x != 0), bns)  # convert value to bool
        probs = [self.accMatrix[idx][bns[idx]] for idx in worker_ids]
        return [np.random.choice(2, 1, p=[1 - prob, prob])[0] for prob in probs]  # return 0:low quality, 1:high quality


class IOHmmWorkers(object):
    """
    initialize the worker model in iohmm.
        # model parameter:
        #  transition[1/0]: bonus (or not) transition probability matrix
        #  r: reference payment level
        #  alpha: skill level alpha
        #  pAlph: the probability distribution of skill level in the population, add up tp 1
        #  beta: responsiveness to to financial incentives
        #  pBeta: the probability distribution of different financial incentives in the population, add up tp 1
        # input:
        #  numWorkers: total number workers
    """
    def __init__(self, num_workers):
        self.behaviour = 'iohmm'

        self.transition = []
        # transition matrix when given bonus
        self.transition.append([[0.8, 0.15, 0.05], [0.3, 0.6, 0.1], [0.2, 0.4, 0.4]])
        # transition matrix not  given bonus
        self.transition.append([[0.4, 0.4, 0.2], [0.1, 0.5, 0.4], [0.05, 0.1, 0.85]])

        # user attributes:
        self.r = [0.2, 0.6, 1.2]  # reference payment level
        self.alph = [0, 1, 3]  # skill level alpha
        self.p_alph = [0.2, 0.6, 0.2]  # the probability distribution of skill level in the population, add up tp 1
        self.beta = [0, 1, 3]  # responsiveness to to financial incentives
        self.p_beta = [0.2, 0.6, 0.2]  # the probability distribution of different financial incentives

        alphs = map(lambda ids: self.alph[ids], np.random.choice(len(self.p_alph), num_workers, p=self.p_alph))
        betas = map(lambda ids: self.alph[ids], np.random.choice(len(self.p_beta), num_workers, p=self.p_beta))
        self.situations = zip(alphs, betas)  # save alpha and beta parameter of a worker
        self.z = np.random.choice(len(self.transition[0]), num_workers)

    """
    update workers' hidden state in the next iteration
        # input:
        #  workerIDs: a list of workers' id, they will work on the question
        #  bonus: a list of bonus with same size as workerIDs, indicate the bonus given to the workers
    """
    def _update_state(self, worker_ids, bns):  # update hidden state for all workers
        self.z = [np.random.choice(len(self.transition[bns[i]]),  # number of hidden states
                  1, p=self.transition[bns[i]][self.z[worker_ids[i]]])  # transition probability
                  for i in range(len(worker_ids))]

    """
    assign a question to the workers in 'workerIDs'
        # input:
        #  workerIDs: a list of workers' id, they will work on the question
        #  bonus: a list of bonus with same size as workerIDs, indicate the bonus given to the workers
        # output:
        #  0/1 list with same size as workerIDs, indicates their working result is correct(with 1) or not(with 0)
    """
    def work(self, worker_ids, bns):
        # emission probabilities: 1 / (1 + e ^ (-alphi - betai(at - rzt))  "bonus or not 4.2"
        probs = [1 / (1 + np.exp(-self.situations[idx][0] -
                 self.situations[idx][1] * (bns[idx] - self.r[self.z[idx]]))) for idx in worker_ids]

        self._update_state(worker_ids, bns)  # update hidden state
        return [np.random.choice(2, 1, p=[1 - prob, prob])[0] for prob in probs]  # return 0:low quality, 1:high quality


class SimulationWorkers(object):
    def __init__(self, num_workers, simtype):
        if simtype == 'uniform':
            self.workers = UniformWorkers(num_workers)
        elif simtype == 'beta':
            self.workers = BetaWorkers(num_workers)
        elif simtype == 'iohmm':
            self.workers = IOHmmWorkers(num_workers)
        else:
            print 'no such type of workers'
            raise Exception
        self.num_workers = num_workers
        self.cmp_pair = (-1, -1)
        self.qualities = None

    def available_workers(self):
        return range(self.num_workers)

    def publish_questions(self, worker_ids, cmp_pair, salariess):
        self.cmp_pair = cmp_pair
        self.qualities = self.workers.work(worker_ids, salariess)

    def collect_answers(self):
        return [int(quality == int(self.cmp_pair[0] < self.cmp_pair[1])) for quality in self.qualities]


if __name__ == '__main__':
    numWorkers = 5
    workerIDs = range(0, 3, 1)
    salaries = [1, 0, 0]
    workers = IOHmmWorkers(numWorkers)
    print workers.work(workerIDs, salaries)
