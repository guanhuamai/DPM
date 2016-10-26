from BonusAllocator import BonusAllocator
from matlab import engine
import numpy as np

class IOHMMBaseline(BonusAllocator):

    def __init__(self, num_workers, base_cost=5, bns=2, hist_qlt_bns=None):
        print 'init an IOHMMBaseline bonus allocator'
        super(IOHMMBaseline, self).__init__(num_workers, base_cost, bns)

        if hist_qlt_bns is None:
            hist_qlt_bns = dict(zip(range(num_workers), [[] for _ in range(num_workers)]))

        self.__hist_qlt_bns = hist_qlt_bns
        self.__prfrm_mat = [[[], []] for i in range(num_workers)]  # performance matrix

        self.__nstates = 0
        self.__ostates = 0
        self.__numitr = 0
        self.__weights = None
        self.set_parameters()
        self.__matlab_engine = engine.start_matlab()

    def __del__(self):
        self.__matlab_engine.quit()


    def set_parameters(self, nstates=3, ostates=2, numitr=1000, weights=None):
        if weights is None:
            weights = [0, 1, 23]  # default value of the weights

        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__weights = weights   # utility weight of different performance and cost,
                                   #  bad: 0, good: 1,  the last bit is the weight of the cost

    def worker_evaluate(self, col_ans, spend, majority_vote):
        for worker in self.__hist_qlt_bns:
            self.__hist_qlt_bns[worker].append((int(col_ans[worker] == majority_vote), spend[worker]))

        bonus_vec = [[0, 1], [1, 0]]
        ou_obs = [[io_pairs[0] for io_pairs in self.__hist_qlt_bns[seqid]] for seqid in
                self.__hist_qlt_bns]  # output observations of every sequences
        in_obs = [[bonus_vec[int(io_pairs[1] > self._base_cost)] for io_pairs in self.__hist_qlt_bns[seqid]]
                for seqid in self.__hist_qlt_bns]  # input observations of every sequences
        self.__prfrm_mat = self.__matlab_engine.iohmmTraining(ou_obs, in_obs,
                self.__nstates, self.__ostates, self.__numitr)['result']

    def __expect_util(self, worker_id, is_bns):
        utility = 0
        if len(self.__prfrm_mat[worker_id][is_bns]) == 0:
            return np.random.random()
        else:
            for i in range(len(self.__prfrm_mat[worker_id][is_bns][0])):  # traverse all the different quality
                prob = self.__prfrm_mat[worker_id][is_bns][0][i]
                utility += prob * self.__weights[i]
        utility -= (self._base_cost + is_bns * self._bns) * self.__weights[-1]
        return utility

    def bonus_alloc(self):
        spend = []
        for worker_id in self.__hist_qlt_bns:
            exputl_true = self.__expect_util(worker_id, 1)  # expect utility if given certain bonus
            exputl_fals = self.__expect_util(worker_id, 0)  # expect utility if not given certain bonus
            spend.append(self._base_cost + (exputl_true > exputl_fals) * self._bns)
        return spend
