from BonusAllocator import BonusAllocator
from matlab import engine
from os import path
import numpy as np


class QLearningAllocator(BonusAllocator):

    def __init__(self, num_workers, discnt=0.99, len_seq=10, base_cost=5, bns=2, t=10):
        super(QLearningAllocator, self).__init__(num_workers, base_cost, bns, t)
        print 'init an qlearnig-mdp bonus allocator'

        self.__len_seq = len_seq
        self.__discnt = discnt

        self.__nstates = 0   # number of hidden states, denoted as S
        self.__ostates = 0   # number of observations, denoted as O
        self.__strt_prob = None  # start probability of hidden states, shape = 1 * S   fake...
        self.__tmat0 = None  # transition matrix with no bonus, shape = S * S, returned after training
        self.__tmat1 = None  # transiton matrix with bonus, shape = S * S, returned after training
        self.__emat  = None  # emission matrix, shpe = S * O, returned after training

        self.__numitr = 0
        self.__weights = None
        self.__q_mat = None

        self.set_parameters()
        self.__matlab_engine = engine.start_matlab()
        self.__matlab_engine.cd(path.join('..', 'MatlabSource', 'IOHMM'))

    def __del__(self):
        self.__matlab_engine.quit()

    def set_parameters(self, nstates=3, ostates=2, strt_prob=None, numitr=1000, weights=None, discnt=0.99, len_seq=10):
        if weights is None:
            weights = [0, 0.15, 0.0025]   # default value of the weights

        if strt_prob is None:
            strt_prob = [ 1.0 / nstates for _ in range(nstates)]

        self.__discnt = discnt
        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__strt_prob = strt_prob
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__weights = weights   # utility weight of different performance and cost, bad: 0, good: 1, cost weight
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
        self.__q_mat = [self.__cal_q(tc) for tc in range(self._t + 1)]  # include T moment

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
                # train_data.append(self.hist_qlt_bns[worker])
                train_data.append(self.hist_qlt_bns[worker][: self._t])  # cut off min_finish
                self.hist_qlt_bns[worker] = self.hist_qlt_bns[worker][self._t:len(self.hist_qlt_bns[worker])]
        if len(train_data) > 3:
            self.train(train_data)

    def __viterbi(self, in_obs, ou_obs):  # tmats[0] transition matrix when not bonus
        t_val = list()
        t_val.append([self.__strt_prob[i] * self.__emat[i][ou_obs[0]] for i in range(self.__nstates)])  # 1 * N
        tmats = (self.__tmat0, self.__tmat1)
        for cur_t in range(1, len(in_obs), 1):
            t_val.append([])
            for j in range(self.__nstates):
                tmp_val = [t_val[cur_t - 1][i] * tmats[in_obs[cur_t]][i][j] * self.__emat[j][ou_obs[cur_t]]
                           for i in range(self.__nstates)]  # from i to j
                t_val[cur_t].append(sum(tmp_val))
            t_val[cur_t] = [float(cur_v) / sum(t_val[cur_t]) for cur_v in t_val[cur_t]]
        return t_val[-1]

    def __cal_reward(self, k, a):
        trans_mat = [self.__tmat0, self.__tmat1]
        return sum([trans_mat[a][k][i] * (self.__emat[i][0] * self.__weights[0] + self.__emat[i][1] * (
                    self.__weights[1] - a * self.__weights[2])) for i in range(self.__nstates)])

    def __cal_q(self, t):
        q_mat = np.zeros((self.__nstates, 2))
        tmats = (self.__tmat0, self.__tmat1)
        for __ in range(self.__numitr):
            k = np.random.choice(self.__nstates, 1)[0]  # random select start states
            for i in range(t):
                a = np.random.choice([0, 1], 1)[0]  # select an input randomly
                k_prime = np.random.choice(self.__nstates, 1, tmats[a][k])[0]   # randomly select next states
                q_mat[k][a] = self.__cal_reward(k, a) + self.__discnt * max(q_mat[k_prime])
                k = k_prime
        return q_mat

    def bonus_alloc(self, in_obs, ou_obs):
        if self.__emat is not None and in_obs is not None and ou_obs is not None:
            states_belief = self.__viterbi(in_obs, ou_obs)
            tc = len(in_obs) % self._t
            # print states
            exp0 = sum([states_belief[k] * self.__q_mat[self._t - tc][k][0] for k in range(self.__nstates)])
            exp1 = sum([states_belief[k] * self.__q_mat[self._t - tc][k][1] for k in range(self.__nstates)])
            return self._base_cost + self._bns * int(exp1 > exp0)
        else:
            return self._base_cost + self._bns * np.random.choice(2, 1)[0]
