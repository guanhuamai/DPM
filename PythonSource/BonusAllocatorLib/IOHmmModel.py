from os import path
from matlab import engine


class IOHmmModel(object):

    def __init__(self):

        self.__nstates = 0
        self.__ostates = 0
        self.__strt_prob = None  # start probability of hidden states, shape = 1 * S   fake...
        self.__tmat0 = None  # transition matrix with no bonus, shape = S * S, returned after training
        self.__tmat1 = None  # transition matrix with bonus, shape = S * S, returned after training
        self.__emat = None  # emission matrix, shape = S * O, returned after training

        self.__numitr = 0
        self.set_parametes()

        self.__matlab_engine = engine.start_matlab()
        self.__matlab_engine.cd(path.join('..', 'MatlabSource', 'IOHMM'))

    def train(self, train_data, base_cost):  # input base cost to recognize which task is given bonus
        bonus_vec = [[0, 1], [1, 0]]
        ou_obs = [[io_pairs[0] for io_pairs in seq] for seq in train_data]  # output observations of every sequences
        in_obs = [[bonus_vec[int(io_pairs[1] > base_cost)] for io_pairs in seq]
                  for seq in train_data]  # input observations of every sequences
        model = self.__matlab_engine.iohmmTraining(ou_obs, in_obs, self.__nstates,
                                                   self.__ostates, self.__numitr)
        self.__tmat0 = list(model['A0'])
        self.__tmat1 = list(model['A1'])
        self.__emat = list(model['B'])

    def get_model(self):
        return self.__tmat0, self.__tmat1, self.__emat

    def set_parametes(self, nstates=2, ostates=2, strt_prob=None, numitr=500):
        if strt_prob is None:
            strt_prob = [1.0 / nstates for _ in range(nstates)]
        self.__strt_prob = strt_prob

        self.__nstates = nstates
        self.__ostates = ostates
        self.__numitr = numitr

    def write_model(self, m_name):
        with open(path.join('.', m_name), 'w') as model_f:
            model_f.write(str(self.__nstates) + '\n')
            model_f.write(str(self.__ostates) + '\n')
            model_f.write(str(self.__strt_prob) + '\n')
            for i in range(self.__nstates):
                model_f.write(reduce(lambda x, y: str(x) + '\t' + str(y), self.__tmat0[i]) + '\n')
            for i in range(self.__nstates):
                model_f.write(reduce(lambda x, y: str(x) + '\t' + str(y), self.__tmat1[i]) + '\n')
            for i in range(self.__nstates):
                model_f.write(reduce(lambda x, y: str(x) + '\t' + str(y), self.__emat[i]) + '\n')

    def read_model(self, m_name):
        with open(path.join('.', m_name), 'r') as model_f:
            lines = model_f.readlines()
            self.__nstates = int(lines[0])
            self.__ostates = int(lines[1])
            self.__strt_prob = map(lambda x: float(x), lines[2].split('\t'))
            self.__tmat0 = []
            for i in range(self.__nstates):
                self.__tmat0.append(map(lambda x: float(x), lines[3 + i].split('\t')))
            self.__tmat1 = []
            for i in range(self.__nstates):
                self.__tmat1.append(map(lambda x: float(x), lines[3 + self.__nstates + i].split('\t')))
            self.__emat = []
            for i in range(self.__nstates):
                self.__emat.append(map(lambda x: float(x), lines[3 + 2 * self.__nstates + i].split('\t')))
