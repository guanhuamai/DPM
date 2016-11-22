import numpy as np


class Example(object):
    def __init__(self):
        self.__nstates = 3
        self.__estates = 2
        self.__tmat0 = [[0.5, 0.2, 0.3], [0.3, 0.6, 0.1], [0.2, 0.4, 0.4]]
        self.__tmat1 = [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
        self.__emat0 = [[0.6, 0.4], [0.8, 0.2], [0.5, 0.5]]
        self.__emat1 = [[0.5, 0.5], [0.4, 0.6], [0.5, 0.5]]
        self.__strt_prob = [0.8, 0.1, 0.1]
        self.__belief = [0.8, 0.1, 0.1]

    def viterbi(self, in_obs, ou_obs):  # tmats[0] transition matrix when not bonus
        tmats = (self.__tmat0, self.__tmat1)
        emats = (self.__emat0, self.__emat1)
        t_val = list()
        t_val.append([self.__strt_prob[i] * emats[in_obs[0]][i][ou_obs[0]] for i in range(self.__nstates)])  # 1 * N

        for cur_t in range(1, len(in_obs), 1):
            t_val.append([])
            for j in range(self.__nstates):
                tmp_val = [t_val[cur_t - 1][i] * tmats[in_obs[cur_t]][i][j] * emats[in_obs[cur_t]][j][ou_obs[cur_t]]
                           for i in range(self.__nstates)]  # from i to j
                t_val[cur_t].append(np.max(tmp_val))
            # t_val[cur_t] = [float(cur_v) / sum(t_val[cur_t]) for cur_v in t_val[cur_t]]
        return t_val[-1]

    def dp_reward(self, in_obs, ou_obs):
        belief = self.viterbi(in_obs, ou_obs)
        max_states = np.argmax(belief)
        rew0 = sum([self.__tmat0[max_states][i] *
                    self.__emat0[i][1] * 1
                    for i in range(self.__nstates)])
        rew1 = sum([self.__tmat1[max_states][i] *
                    self.__emat1[i][1] * (1 - 0.05)
                    for i in range(self.__nstates)])
        return np.argmax([rew0, rew1])

    def reward_matrix(self):
        tmats = (self.__tmat0, self.__tmat1)
        emats = (self.__emat0, self.__emat1)
        f = lambda k, a: sum([tmats[a][k][i] * (emats[a][i][1] * (1 - 0.05 * a)) for i in range(self.__nstates)])
        r = [[f(_k, _a) for _a in range(2)] for _k in range(self.__nstates)]
        return r

    def cal_q(self):
        r = self.reward_matrix()
        q_mat = np.zeros((self.__nstates, 2))
        tmats = (self.__tmat0, self.__tmat1)

        k = 0  # random select start states
        a = 1
        k_prime = 2
        for i in range(3):
            q_mat[k][a] = r[k][a] + 0.8 * max(q_mat[k_prime])
            print 'k =', k
            print 'a =', a
            print 'k\' =', k_prime
            print 'q matrix:\n', q_mat
            print '------------------'
            k = k_prime
            k_prime = np.random.choice(self.__nstates, 1, tmats[a][k])[0]  # randomly select next states
            a = np.random.choice([0, 1], 1)[0]  # select an input randomly
        print '***********************'

        for __ in range(9):
            k = np.random.choice(self.__nstates, 1)[0]  # random select start states
            for i in range(3):
                a = np.random.choice([0, 1], 1)[0]  # select an input randomly
                k_prime = np.random.choice(self.__nstates, 1, tmats[a][k])[0]   # randomly select next states
                q_mat[k][a] = r[k][a] + 0.8 * max(q_mat[k_prime])
                print 'k =', k
                print 'a =', a
                print 'k\' =', k_prime
                print 'q matrix:\n', q_mat
                print '------------------'
                k = k_prime
            print '***********************'
        return q_mat

if __name__ == '__main__':
    e = Example()
    e.dp_reward([0, 1, 1], [1, 0, 1])
    e.cal_q()
