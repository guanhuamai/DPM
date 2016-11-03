from BonusAllocator import BonusAllocator
from matlab import engine
from os import path
import numpy as np


class NStepAllocator(BonusAllocator):

    def __init__(self, num_workers, nstep=5, len_seq=10, base_cost=5, bns=2):
        super(NStepAllocator, self).__init__(num_workers, base_cost, bns)
        print 'init an nstep look ahead bonus allocator'
        
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
            weights = [0, 0.15, 0.0025]  # default value of the weights

        if strt_prob is None:
            strt_prob = [ 1.0 / nstates for _ in range(nstates)]

        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__strt_prob = strt_prob
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__weights = weights   # utility weight of different performance and cost, bad: 0, good: 1, cost weight
        self.__nstep = nstep
        self.__len_seq = len_seq

    def train(self, train_data):
        print 'train iohmm model'

        bonus_vec = [[0, 1], [1, 0]]
        ou_obs = [[io_pairs[0] for io_pairs in seq] for seq in train_data]  # output observations of every sequences
        in_obs = [[bonus_vec[int(io_pairs[1] > self._base_cost)] for io_pairs in seq]
                  for seq in train_data]  # input observations of every sequences
        model = self.__matlab_engine.iohmmTraining(ou_obs, in_obs, self.__nstates,
                                                   self.__ostates, self.__numitr)
        self.__tmat0 = list(model['A0'])
        self.__tmat1 = list(model['A1'])
        self.__emat = list(model['B'])

    def update(self, worker_ids, answers, spend, majority_vote, t):
        for i in range(len(worker_ids)):
            try:
                self.hist_qlt_bns[worker_ids[i]].append((int(answers[i] == majority_vote), spend[i]))
            except KeyError:
                self.hist_qlt_bns[worker_ids[i]] = []
                self.hist_qlt_bns[worker_ids[i]].append((int(answers[i] == majority_vote), spend[i]))

        train_data = []  # workers whose history list is long enough to train new iohmm model
        for worker in self.hist_qlt_bns:
            if len(self.hist_qlt_bns[worker]) >= t:
                train_data.append(self.hist_qlt_bns[worker][: t])  # cut off min_finish
                self.hist_qlt_bns[worker] = self.hist_qlt_bns[worker][t:len(self.hist_qlt_bns[worker])]
        if len(train_data) > 3:
            self.train(train_data)

    def viterbi(self, in_obs, ou_obs):  # tmats[0] transition matrix when not bonus
        t_val = list()
        t_val.append([self.__strt_prob[i] * self.__emat[i][ou_obs[0]] for i in range(self.__nstates)])  # 1 * N
        tmats = (self.__tmat0, self.__tmat1)
        for cur_t in range(1, len(in_obs), 1):
            t_val.append([])
            for j in range(self.__nstates):
                tmp_val = [t_val[cur_t - 1][i] * tmats[in_obs[cur_t]][i][j] * self.__emat[j][ou_obs[cur_t]]
                           for i in range(self.__nstates)]  # from i to j
                t_val[cur_t].append(sum(tmp_val))
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
                for j in range(self.__nstates)]

    def __exp_utility(self, belief, a, nstep):
        trans_mat = [self.__tmat0, self.__tmat1]
        rewrd = self.__cal_reward(belief, a)
        if nstep == 1:
            return rewrd
        rslt = 0
        for x in range(self.__ostates):
            sum_state_exp = 0
            for i in range(self.__nstates):
                state_exp = 0
                for j in range(self.__nstates):
                    state_exp += trans_mat[a][i][j] * self.__emat[j][x]
                sum_state_exp += belief[i] * state_exp

            # expected utility when no given bonus
            expt_util0 = self.__exp_utility(self.__cal_belief(belief, 0, x), 0, nstep-1)
            # expected utility when given bonus
            expt_util1 = self.__exp_utility(self.__cal_belief(belief, 1, x), 1, nstep-1)
            v_val = max(expt_util0, expt_util1)
            rslt += sum_state_exp * v_val
        rslt += rewrd
        return rslt

    def bonus_alloc(self, in_obs, ou_obs):
        if in_obs is not None and ou_obs is not None:
            exp0 = self.__exp_utility(self.viterbi(in_obs, ou_obs), 0, self.__nstep)
            exp1 = self.__exp_utility(self.viterbi(in_obs, ou_obs), 1, self.__nstep)
            return self._base_cost + self._bns * int(exp1 > exp0)
        else:
            return self._base_cost + self._bns * np.random.choice(2, 1)[0]

