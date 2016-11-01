from BonusAllocator import BonusAllocator
import numpy as np


class RandomAllocator(BonusAllocator):

    def __init__(self, num_workers, base_cost=5, bns=2, p=1):
        super(RandomAllocator, self).__init__(num_workers, base_cost, bns)
        print 'init an IOHMMBaseline bonus allocator'
        self.__probability = p
        self.set_parameters()

    def set_parameters(self, p=1):
        self.__probability = p   # probability of giving bonus

    def worker_evaluate(self, col_ans, spend, majority_vote):
        for worker in self.hist_qlt_bns:
            self.hist_qlt_bns[worker].append((int(col_ans[worker] == majority_vote), spend[worker]))
        print 'totally randomly allocate, do nothing to evaluate'

    def bonus_alloc(self):
        spend = []
        is_bonus = np.random.choice([0, 1], self._num_workers, p=[1-self.__probability, self.__probability])
        for worker_id in range(self._num_workers):
            spend.append(self._base_cost + is_bonus[worker_id] * self._bns)
        return spend
