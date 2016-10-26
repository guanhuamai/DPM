from ApollingInference import *
from ApollingSelect import *
from DecisionMaker import DecisionMaker


class Apolling(DecisionMaker):

    def __init__(self, nu_workers, nu_nodes):
        print 'init an apolling engine'
        super(Apolling, self).__init__(nu_workers, nu_nodes)
        self.__used_edges = []
        self.__matrix = {}

    def __update(self, cmp_pair, col_ans):
        # update answer matrix with answers, votes statistics
        for ans in col_ans:
            tmp_pair = cmp_pair
            if ans == 0:
                tmp_pair = (cmp_pair[1], cmp_pair[0])
            try:
                self.__matrix[tmp_pair] += 1
            except KeyError:
                self.__matrix[tmp_pair] = 1

        # update usedEdges with answers
        self.__used_edges.append(cmp_pair)

    def pair_selection(self, cmp_pair=None, col_ans=None):
        if cmp_pair is not None and col_ans is not None:
            self.__update(cmp_pair, col_ans)
        return apolling_select(self._num_workers, self._all_nodes, self.__used_edges, self._all_edges, self.__matrix)

    def result_inference(self):
        return apolling_inference(self._all_nodes, self._all_edges, self.__matrix)
