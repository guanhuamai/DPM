from CrowdbtInference import *
from CrowdbtSelect import *
from DecisionMaker import DecisionMaker


class Crowdbt(DecisionMaker):

    def __init__(self, nu_workers, nu_nodes):
        super(Crowdbt, self).__init__(nu_workers, nu_nodes)
        print 'init a crowdbt engine'
        self.used_edges = []
        self.matrix = {}

    def update(self, cmp_pair=None, col_ans=None):
        # update answer matrix with answers, votes statistics
        if cmp_pair is not None and col_ans is not None:
            for ans in col_ans:
                tmp_pair = cmp_pair
                if ans == 0:
                    tmp_pair = (cmp_pair[1], cmp_pair[0])
                try:
                    self.matrix[tmp_pair] += 1
                except KeyError:
                    self.matrix[tmp_pair] = 1

        # update usedEdges with answers
        self.used_edges.append(cmp_pair)

    def pair_selection(self):
        return crowdbt_select(self.num_workers, self.all_nodes, self.all_edges, self.used_edges, self.matrix)

    def result_inference(self):
        return crowdbt_inference(self.all_nodes, self.all_edges, self.matrix)
