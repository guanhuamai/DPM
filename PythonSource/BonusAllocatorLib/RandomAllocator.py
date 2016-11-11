from BonusAllocator import BonusAllocator
import numpy as np


class RandomAllocator(BonusAllocator):

    def __init__(self, num_workers, base_cost=5, bns=2, p=1, t=10):
        super(RandomAllocator, self).__init__(num_workers, base_cost, bns, t)
        print 'init a random bonus allocator'
        self.__probability = p
        self.set_parameters()

    def set_parameters(self, p=1):
        self.__probability = p   # probability of giving bonus

    def train(self, train_data):
        pass

    def update(self, worker_ids, answers, spend, majority_vote):
        for i in range(len(worker_ids)):
            try:
                self.hist_qlt_bns[worker_ids[i]].append((int(answers[i] == majority_vote), spend[i]))
            except KeyError:
                self.hist_qlt_bns[worker_ids[i]] = []
                self.hist_qlt_bns[worker_ids[i]].append((int(answers[i] == majority_vote), spend[i]))

        train_data = []  # workers whose history list is long enough to train new iohmm model
        for worker in self.hist_qlt_bns:
            if len(self.hist_qlt_bns[worker]) >= self._t:
                train_data.append(self.hist_qlt_bns[worker][: self._t])  # cut off min_finish
                self.hist_qlt_bns[worker] = self.hist_qlt_bns[worker][self._t:len(self.hist_qlt_bns[worker])]

        if len(train_data) > 3:
            self.train(train_data)

    def bonus_alloc(self, in_obs, ou_obs):
        return self._base_cost + self._bns * np.random.choice(2, 1, p=[1-self.__probability, self.__probability])[0]
