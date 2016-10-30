from BonusAllocator import BonusAllocator
from matlab import engine
from os import path
import numpy as np
import mdptoolbox


class MLSAllocator(BonusAllocator):

    def __init__(self, num_workers, len_seq=10, base_cost=5, bns=2, hist_qlt_bns=None):
        super(MLSAllocator, self).__init__(num_workers, base_cost, bns)
        print 'init an mls-mdp bonus allocator'
        if hist_qlt_bns is None:
            hist_qlt_bns = dict(zip(range(num_workers), [[] for _ in range(num_workers)]))

        self.__hist_qlt_bns = hist_qlt_bns

        self.__len_seq = len_seq

        self.__nstates = 0   # number of hidden states, denoted as S
        self.__ostates = 0   # number of observations, denoted as O
        self.__strt_prob = None  # start probability of hidden states, shape = 1 * S   fake...
        self.__tmat0 = None  # transition matrix with no bonus, shape = S * S, returned after training
        self.__tmat1 = None  # transiton matrix with bonus, shape = S * S, returned after training
        self.__emat  = None  # emission matrix, shpe = S * O, returned after training

        self.__numitr = 0
        self.__weights = None
        self.__max_stas = None

        self.set_parameters()
        self.__matlab_engine = engine.start_matlab()
        self.__matlab_engine.cd(path.join('..', 'MatlabSource', 'IOHMM'))

    def __del__(self):
        self.__matlab_engine.quit()

    def set_parameters(self, nstates=3, ostates=2, strt_prob=None, numitr=1000, weights=None):
        if weights is None:
            weights = [0, 1, 23]  # default value of the weights

        if strt_prob is None:
            strt_prob = [ 1.0 / nstates for _ in range(nstates)]

        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__weights = weights   # utility weight of different performance and cost, bad: 0, good: 1, cost weight
        self.__strt_prob = strt_prob

    def worker_evaluate(self, col_ans, spend, majority_vote):
        for worker in self.__hist_qlt_bns:
            self.__hist_qlt_bns[worker].append((int(col_ans[worker] == majority_vote), spend[worker]))

        bonus_vec = [[0, 1], [1, 0]]
        ou_obs = [[io_pairs[0] for io_pairs in self.__hist_qlt_bns[seqid]] for seqid in
                  self.__hist_qlt_bns]  # output observations of every sequences
        in_obs = [[bonus_vec[int(io_pairs[1] > self._base_cost)] for io_pairs in self.__hist_qlt_bns[seqid]]
                  for seqid in self.__hist_qlt_bns]  # input observations of every sequences
        model = self.__matlab_engine.iohmmTraining(ou_obs, in_obs, self.__nstates,
                                                              self.__ostates, self.__numitr)
        self.__tmat0 = model['A0']
        self.__tmat1 = model['A1']
        self.__emat  = model['B']
        self.__max_stas = [self.viterbi(in_obs[i], ou_obs[i], len(self.__hist_qlt_bns[i])) for i in range(self._num_workers)]

    def viterbi(self, inobs, ouobs, T):  # tmats[0] transition matrix when not bonus
        t_val = list()
        t_val.append([self.__start_probs[i] * self.__emat[i][ouobs[0]] for i in range(self.__nstates)])  # 1 * N
        t_sta = list()
        t_sta.append([-1 for _ in range(self.__nstates)])
        tmats = (self.__tmat0, self.__tmat1)
        for cur_t in range(1, T, 1):
            for j in range(self.__nstates):
                t_val.append([])
                t_sta.append([])
                max_val = 0
                max_sta = -1
                for i in range(self.__nstates):
                    tmp_val = t_val[cur_t - 1][i] * tmats[inobs[cur_t]][i][j] * self.__emat[j][ouobs[cur_t]]
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
        P = np.array([self.__tmat0, self.__tmat1])
        R = []
        R.append([sum([self.__tmat0[k][i] * (self.__emat[i][0] * self.__weights[0] + self.__emat[i][0] * self.__weights[1]) for i in  range(self.__nstates)]) for k in range(self.__nstates)])
        R.append([sum([self.__tmat0[k][i] * (self.__emat[i][0] * self.__weights[0] + self.__emat[i][0] * (self.__weights[1] - self.__weights[2])) for i in range(self.__nstates)]) for k in range(self.__nstates)])
        R = np.array(R)
        fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.99, self.__len_seq - (len(self.__hist_qlt_bns[i]) % self.__len_seq))
        fh.run()
        spend = []
        for worker_id in self._num_workers:
            spend.append(self._base_cost + (fh.policy[worker_id][0]) * self._bns)
        return spend