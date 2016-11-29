from BonusAllocator import BonusAllocator
import numpy as np


class NStepAllocator(BonusAllocator):

    def __init__(self, num_workers, nstep=5, len_seq=10, base_cost=5, bns=2, t=10, weights=None):
        super(NStepAllocator, self).__init__(num_workers, base_cost, bns, t, weights)
        print 'init an nstep look ahead bonus allocator'
        
        self.__nstep = nstep
        self.__len_seq = len_seq

        self.__nstates = 0   # number of hidden states, denoted as S
        self.__ostates = 0   # number of observations, denoted as O
        self.__strt_prob = None  # start probability of hidden states, shape = 1 * S   fake...
        self.__tmat0 = None  # transition matrix with no bonus, shape = S * S, returned after training
        self.__tmat1 = None  # transiton matrix with bonus, shape = S * S, returned after training
        self.__emat = None  # emission matrix, shpe = S * O, returned after training

        self.__numitr = 0

        self.set_parameters()

    def set_parameters(self, nstates=2, ostates=2, strt_prob=None, numitr=1000, nstep=2, len_seq=10):
        if strt_prob is None:
            strt_prob = [1.0 / nstates for _ in range(nstates)]

        self.__nstates = nstates   # number of hidden states
        self.__ostates = ostates   # number of observations
        self.__strt_prob = strt_prob
        self.__numitr = numitr     # number of iteration in EM algorithm
        self.__nstep = nstep
        self.__len_seq = len_seq

    def train(self, model):
        model_param = model.get_model()
        self.__tmat0 = model_param[0]
        self.__tmat1 = model_param[1]
        self.__emat = model_param[2]

    def __pseudo_viterbi(self, in_obs, ou_obs):  # tmats[0] transition matrix when not bonus
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

    def __cal_reward(self, belief, is_bonus):
        trans_mat = [self.__tmat0, self.__tmat1]
        state_rew = [belief[i] * sum([trans_mat[is_bonus][i][j] *
                                      (self.__emat[j][0] * self.weights[0] + self.__emat[j][1] *
                                       (self.weights[1] - self.weights[2] * int(is_bonus)))
                                      for j in range(self.__nstates)]) for i in range(self.__nstates)]
        return sum(state_rew)

    def __cal_belief(self, belief, is_bonus, obs):
        trans_mat = [self.__tmat0, self.__tmat1]
        new_belief = [sum([belief[i] * trans_mat[is_bonus][i][j] * self.__emat[j][obs] for i in range(self.__nstates)])
                      for j in range(self.__nstates)]
        new_belief = [new_b / sum(new_belief) for new_b in new_belief]  # normalize
        return new_belief

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

            new_belief = self.__cal_belief(belief, a, x)
            # expected utility when no given bonus
            expt_util0 = self.__exp_utility(new_belief, 0, nstep-1)
            # expected utility when given bonus
            expt_util1 = self.__exp_utility(new_belief, 1, nstep-1)
            v_val = max(expt_util0, expt_util1)
            rslt += sum_state_exp * v_val
        rslt += rewrd
        return rslt

    def bonus_alloc(self, in_obs, ou_obs):
        if self.__emat is not None and in_obs is not None and ou_obs is not None:
            states = self.__pseudo_viterbi(in_obs, ou_obs)
            # print states
            exp0 = self.__exp_utility(states, 0, self.__nstep)
            exp1 = self.__exp_utility(states, 1, self.__nstep)
            return self._base_cost + self._bns * int(exp1 > exp0)
        else:
            return self._base_cost + self._bns * np.random.choice(2, 1)[0]
