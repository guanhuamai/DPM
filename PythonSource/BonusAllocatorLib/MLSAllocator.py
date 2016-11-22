from BonusAllocator import BonusAllocator
from IOHmmModel import IOHmmModel
import numpy as np
import mdptoolbox


class MLSAllocator(BonusAllocator):

    def __init__(self, num_workers, len_seq=10, base_cost=5, bns=2, t=10, weights=None):
        super(MLSAllocator, self).__init__(num_workers, base_cost, bns, t, weights)
        print 'init an mls-mdp bonus allocator'

        self.__len_seq = len_seq

        self.__nstates = 0   # number of hidden states, denoted as S
        self.__ostates = 0   # number of observations, denoted as O
        self.__strt_prob = None  # start probability of hidden states, shape = 1 * S   fake...
        self.__tmat0 = None  # transition matrix with no bonus, shape = S * S, returned after training
        self.__tmat1 = None  # transiton matrix with bonus, shape = S * S, returned after training
        self.__emat = None  # emission matrix, shpe = S * O, returned after training

        self.__numitr = 0
        self.__policy = None  # bonus policy

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

        p = np.array([self.__tmat0, self.__tmat1])
        r = list()
        r.append([sum([self.__tmat0[k][i] *
                       (self.__emat[i][0] * self.weights[0] + self.__emat[i][1] * self.weights[1])
                       for i in range(self.__nstates)]) for k in range(self.__nstates)])
        r.append([sum([self.__tmat1[k][i] *
                       (self.__emat[i][0] * self.weights[0] + self.__emat[i][1] *
                        (self.weights[1] - self.weights[2]))
                       for i in range(self.__nstates)]) for k in range(self.__nstates)])
        r = np.transpose(np.array(r))

        def mdp_policy(horizon):
            fh = mdptoolbox.mdp.FiniteHorizon(p, r, 0.9, horizon)
            fh.run()
            return list(fh.policy)

        self.__policy = map(mdp_policy, range(1, self._t + 1))

    def __viterbi(self, in_obs, ou_obs):  # tmats[0] transition matrix when not bonus
        t_val = list()
        t_val.append([self.__strt_prob[i] * self.__emat[i][ou_obs[0]] for i in range(self.__nstates)])  # 1 * N
        tmats = (self.__tmat0, self.__tmat1)
        for cur_t in range(1, len(in_obs), 1):
            t_val.append([])
            for j in range(self.__nstates):
                tmp_val = [t_val[cur_t - 1][i] * tmats[in_obs[cur_t]][i][j] * self.__emat[j][ou_obs[cur_t]]
                           for i in range(self.__nstates)]
                t_val[cur_t].append(np.max(tmp_val))
            t_val[cur_t] = [float(cur_v) / sum(t_val[cur_t]) for cur_v in t_val[cur_t]]
        return np.argmax(t_val[-1])

    def bonus_alloc(self, in_obs, ou_obs):
        if self.__policy is not None and in_obs is not None and ou_obs is not None:
            tc = len(in_obs) % self._t
            return self._base_cost + self._bns * self.__policy[self._t - tc - 1][self.__viterbi(in_obs, ou_obs)][0]
        else:
            return self._base_cost + self._bns * np.random.choice(2, 1)[0]
