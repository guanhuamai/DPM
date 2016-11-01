

class BonusAllocator(object):

    def __init__(self, num_workers, base_cost=5, bns=2):
        print 'init a base bonus allocator'
        self.hist_qlt_bns = dict(zip(range(num_workers), [[] for _ in range(num_workers)]))
        self._base_cost = base_cost
        self._bns = bns
        self._num_workers = num_workers

    def worker_evaluate(self, *args):
        raise NotImplementedError('Please Implement this method')

    def bonus_alloc(self, *args):
        raise NotImplementedError('Please Implement this method')
