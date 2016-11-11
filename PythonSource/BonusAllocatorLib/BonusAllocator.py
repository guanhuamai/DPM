

class BonusAllocator(object):

    def __init__(self, num_workers, base_cost=5, bns=2, t=10, weights=None):
        self.hist_qlt_bns = dict(zip(range(num_workers), [[] for _ in range(num_workers)]))
        self._base_cost = base_cost
        self._bns = bns
        self._num_workers = num_workers
        self._t = t

        if weights is None:
            weights = [0, 0.15, 0.0025]  # default value of the weights
        self.weights = weights

    def update(self, worker_ids, answers, spend, majority_vote):
        for i in range(len(worker_ids)):
            try:
                self.hist_qlt_bns[worker_ids[i]].append((int(answers[i] == majority_vote), spend[i]))
            except KeyError:
                self.hist_qlt_bns[worker_ids[i]] = []
                self.hist_qlt_bns[worker_ids[i]].append((int(answers[i] == majority_vote), spend[i]))

    def train(self, *args):
        raise NotImplementedError('Please Implement this method')

    def bonus_alloc(self, *args):
        raise NotImplementedError('Please Implement this method')
