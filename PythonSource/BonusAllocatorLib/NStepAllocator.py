from BonusAllocator import BonusAllocator
from matlab import engine
from os import path
import numpy as np


class NStepAllocator(BonusAllocator):

    def __init__(self, num_workers, nstep=5, len_seq=10, base_cost=5, bns=2, hist_qlt_bns=None):
        super(NStepAllocator, self).__init__(num_workers, base_cost, bns)
        print 'init an mls-mdp bonus allocator'
        if hist_qlt_bns is None:
            hist_qlt_bns = dict(zip(range(num_workers), [[] for _ in range(num_workers)]))

        self.__hist_qlt_bns = hist_qlt_bns
        self.__nstep = nstep
        self.__len_seq = len_seq

        self.__nstates = 0   # number of hidden states, denoted as S
        self.__ostates = 0   # number of observations, denoted as O
        self.__strt_prob = None  # start probability of hidden states, shape = 1 * S   fake...
        self.__tmat0 = None  # transition matrix with no bonus, shape = S * S, returned after training
        self.__tmat1 = None  # transiton matrix with bonus, shape = S * S, returned after training
        self.__emat  = None  # emission matrix, shpe = S * O, returned after training

        self.__numitr = 0
        self.__weights = None
        self.__belief = None

        self.set_parameters()
        self.__matlab_engine = engine.start_matlab()
        self.__matlab_engine.cd(path.join('..', 'MatlabSource', 'IOHMM'))

    def __del__(self):
        self.__matlab_engine.quit()

    def set_parameters(self, nstates=3, ostates=2, strt_prob=None, numitr=1000, weights=None, nstep=5, len_seq=10):
        if weights is None:
            weights = [0, 1, 23]  # default value of the weights

        if strt_prob is None:
            strt_prob = [ 1.0 / nstates for _ in range(nstates)]

        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__strt_prob = strt_prob
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__weights = weights   # utility weight of different performance and cost, bad: 0, good: 1, cost weight
        self.__nstep = nstep
        self.__len_seq = len_seq

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
        self.__emat = model['B']
        self.__belief = [self.viterbi(in_obs[i], ou_obs[i], len(self.__hist_qlt_bns[i])) for i in range(self._num_workers)]

    def viterbi(self, inobs, ouobs, T):  # tmats[0] transition matrix when not bonus
        t_val = list()
        t_val.append([self.__strt_prob[i] * self.__emat[i][ouobs[0]] for i in range(self.__nstates)])  # 1 * N
        tmats = (self.__tmat0, self.__tmat1)
        for cur_t in range(1, T, 1):
            t_val.append([])
            for j in range(self.__nstates):
                tmp_val = [t_val[cur_t - 1][i] * tmats[inobs[cur_t]][i][j] * self.__emat[j][ouobs[cur_t]]
                           for i in range(self.__nstates)]
                t_val[cur_t].append(np.max(tmp_val))
        return t_val

    def __cal_reward(self, belief, is_bonus):
        trans_mat = [self.__tmat0, self.__tmat1]
        state_rew = [belief[i] * sum([trans_mat[is_bonus][i][j] *
                                      (self.__emat[j][0] * self.__weights[0] + self.__emat[j][1] *
                                       (self.__weights[1] - self.__weights[2] * int(is_bonus)))
                                      for j in range(self.__nstates)]) for i in range(self.__nstates)]
        return sum(state_rew)

    def __cal_belief(self, belief, is_bonus, obs):
        trans_mat = [self.__tmat0, self.__tmat1]
        return [sum([belief[i] * trans_mat[is_bonus][i][j] * self.__emat[j][obs] for i in range(self.__nstates)])
                for j in range(len(belief))]

    def __exp_utility(self, belief, a, nstep):
        trans_mat = [self.__tmat0, self.__tmat1]
        if nstep == 1:
            return self.__cal_reward(belief, a)
        rslt = 0
        for x in range(self.__ostates):
            sum_state_exp = 0
            for i in range(self.__nstates):
                state_exp = 0
                for j in range(self.__nstates):
                    state_exp += trans_mat[a][i][j] * self.__emat[j][x]
                sum_state_exp += belief[i] * state_exp

            # expected utility when no given bonus
            expt_util0 = self.__exp_utility(self.__cal_belief(belief, 0, x))
            # expected utility when given bonus
            expt_util1 = self.__exp_utility(self.__cal_belief(belief, 1, x))
            v_val = max(expt_util0, expt_util1)
            rslt += sum_state_exp * v_val
        return rslt

    def bonus_alloc(self):
        spend = []
        if self.__belief is not None:
            for worker_id in self._num_workers:
                exp0 = self.__exp_utility(self.__belief[worker_id], 0, self.__nstep)
                exp1 = self.__exp_utility(self.__belief[worker_id], 1, self.__nstep)
                spend.append(self._base_cost + (exp1 > exp0) * self._bns)
        else:
            spend = map(lambda x: self._base_cost + (x == 1) * self._bns, np.random.choice(2, self._num_workers))
        return spend
