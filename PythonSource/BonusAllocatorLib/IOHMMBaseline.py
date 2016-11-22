from BonusAllocator import BonusAllocator
from IOHmmModel import IOHmmModel
import numpy as np


class IOHMMBaseline(BonusAllocator):

    def __init__(self, num_workers, base_cost=5, bns=2, t=10, weights=None):
        super(IOHMMBaseline, self).__init__(num_workers, base_cost, bns, t, weights)
        print 'init an IOHMMBaseline bonus allocator'

        self.__nstates = 0
        self.__ostates = 0
        self.__strt_prob = None  # start probability of hidden states, shape = 1 * S   fake...
        self.__tmat0 = None  # transition matrix with no bonus, shape = S * S, returned after training
        self.__tmat1 = None  # transiton matrix with bonus, shape = S * S, returned after training
        self.__emat = None  # emission matrix, shpe = S * O, returned after training

        self.__numitr = 0
        self.set_parameters()

    def set_parameters(self, nstates=2, ostates=2, strt_prob=None, numitr=1000):
        if strt_prob is None:
            strt_prob = [1.0 / nstates for _ in range(nstates)]

        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__strt_prob = strt_prob

    def train(self, model):
        model_param = model.get_model()
        self.__tmat0 = model_param[0]
        self.__tmat1 = model_param[1]
        self.__emat = model_param[2]

    def __viterbi(self, in_obs, ou_obs):  # tmats[0] transition matrix when not bonus
        if len(in_obs) == 0:
            return self.__strt_prob
        t_val = list()
        t_val.append([self.__strt_prob[i] * self.__emat[i][ou_obs[0]] for i in range(self.__nstates)])  # 1 * N
        t_val[0] = [float(cur_v) / sum(t_val[0]) for cur_v in t_val[0]]
        tmats = (self.__tmat0, self.__tmat1)
        for cur_t in range(1, len(in_obs), 1):
            t_val.append([])
            for j in range(self.__nstates):
                tmp_val = [t_val[cur_t - 1][i] * tmats[in_obs[cur_t]][i][j] * self.__emat[j][ou_obs[cur_t]]
                           for i in range(self.__nstates)]  # from i to j
                t_val[cur_t].append(sum(tmp_val))
            t_val[cur_t] = [float(cur_v) / sum(t_val[cur_t]) for cur_v in t_val[cur_t]]
        return t_val[-1]

    def __cal_reward(self, belief, is_bonus):
        trans_mat = [self.__tmat0, self.__tmat1]
        # state_rew = 0
        # for i in range(self.__nstates):
        #     exp_sum = 0
        #     for j in range(self.__nstates):
        #         j_util = (self.__emat[j][0] * self.weights[0] + self.__emat[j][1] *
        #                   self.weights[1] - self.weights[2] * int(is_bonus))
        #         exp_sum += trans_mat[is_bonus][i][j] * j_util
        #     state_rew += belief[i] * exp_sum
        state_rew = [belief[i] * sum([trans_mat[is_bonus][i][j] *
                                      (self.__emat[j][0] * self.weights[0] + self.__emat[j][1] *
                                       (self.weights[1] - self.weights[2] * int(is_bonus)))
                                      for j in range(self.__nstates)]) for i in range(self.__nstates)]
        return sum(state_rew)
        # return state_rew

    def bonus_alloc(self, in_obs, ou_obs):
        if self.__emat is not None and in_obs is not None and ou_obs is not None:
            in_obs = in_obs[len(in_obs)-(len(in_obs) % self._t):len(in_obs)]
            ou_obs = ou_obs[len(ou_obs)-(len(ou_obs) % self._t):len(ou_obs)]
            states_belief = self.__viterbi(in_obs, ou_obs)
            exp0 = self.__cal_reward(states_belief, 0)
            exp1 = self.__cal_reward(states_belief, 1)
            return self._base_cost + self._bns * int(exp1 > exp0)
        else:
            return self._base_cost + self._bns * np.random.choice(2, 1)[0]
