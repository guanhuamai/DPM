from BonusAllocator import BonusAllocator
from matlab import engine
from os import path
import numpy as np
import mdptoolbox as mdptb


class MLSAllocator(BonusAllocator):

    def __init__(self, num_workers, base_cost=5, bns=2, hist_qlt_bns=None):
        super(MLSAllocator, self).__init__(num_workers, base_cost, bns)
        print 'init an mls-mdp bonus allocator'
        if hist_qlt_bns is None:
            hist_qlt_bns = dict(zip(range(num_workers), [[] for _ in range(num_workers)]))

        self.__hist_qlt_bns = hist_qlt_bns

        self.__nstates = 0
        self.__ostates = 0
        self.__numitr = 0
        self.__weights = None
        self.set_parameters()
        self.__matlab_engine = engine.start_matlab()
        self.__matlab_engine.cd(path.join('..', 'MatlabSource', 'IOHMM'))

    def __del__(self):
        self.__matlab_engine.quit()

    def set_parameters(self, nstates=3, ostates=2, numitr=1000, weights=None):
        if weights is None:
            weights = [0, 1, 23]  # default value of the weights

        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__weights = weights   # utility weight of different performance and cost,   bad: 0, good: 1,\
        #   the last bit is the weight of the cost

    def worker_evaluate(self, col_ans, spend, majority_vote):
        for worker in self.__hist_qlt_bns:
            self.__hist_qlt_bns[worker].append((int(col_ans[worker] == majority_vote), spend[worker]))

        bonus_vec = [[0, 1], [1, 0]]
        ou_obs = [[io_pairs[0] for io_pairs in self.__hist_qlt_bns[seqid]] for seqid in
                  self.__hist_qlt_bns]  # output observations of every sequences
        in_obs = [[bonus_vec[int(io_pairs[1] > self._base_cost)] for io_pairs in self.__hist_qlt_bns[seqid]]
                  for seqid in self.__hist_qlt_bns]  # input observations of every sequences
        self.__prfrm_mat = self.__matlab_engine.iohmmTraining(ou_obs, in_obs, self.__nstates,
                                                              self.__ostates, self.__numitr)['result']

    def viterbi(self, start_probs, tmats, emat, inobs, ouobs, T):  # tmats[0] transition matrix when not bonus
        t_val = list()
        t_val.append([start_probs[i] * emat[i][ouobs[0]] for i in range(self.__nstates)])  # 1 * N
        t_sta = list()
        t_sta.append([-1 for _ in range(self.__nstates)])
        for cur_t in range(1, T, 1):
            for j in range(self.__nstates):
                t_val.append([])
                t_sta.append([])
                max_val = 0
                max_sta = -1
                for i in range(self.__nstates):
                    tmp_val = t_val[cur_t - 1][i] * tmats[inobs[cur_t]][i][j] * emat[j][ouobs[cur_t]]
                    tmp_sta = i
                    if max_val < tmp_val:
                        max_val = tmp_val
                        max_sta = tmp_sta
                t_val[cur_t].append(max_val)
                t_sta[cur_t].append(max_sta)
        max_val = 0
        max_sta = -1
        for i in range(self.__nstates):
            if max_val < t_val[T - 1][i]:
                max_val = t_val[T - 1][i]
                max_sta = t_sta[T - 1][i]
        return max_sta





    def bonus_alloc(self):
        spend = []
        for worker_id in self.__hist_qlt_bns:
            exputl_true = self.__expect_util(worker_id, 1)  # expect utility if given certain bonus
            exputl_fals = self.__expect_util(worker_id, 0)  # expect utility if not given certain bonus
            spend.append(self._base_cost + (exputl_true > exputl_fals) * self._bns)
        return spend
