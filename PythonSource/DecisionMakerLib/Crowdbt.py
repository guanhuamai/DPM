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
        m = cmp_pair[0]
        n = cmp_pair[1]
        if m < n:
            m = cmp_pair[1]
            n = cmp_pair[0]
        self.rest_edges_nums.remove(m * (m-1) / 2 + n)

    def pair_selection(self):
        if len(self.all_nodes) > 100:  # once number of query too large, randomly select all of the edges
            select_edge_num = np.random.choice(self.rest_edges_nums, 1)[0]
            for m in range(len(self.all_nodes)):
                id_last_col = m * (m + 1) / 2 - 1
                if id_last_col > select_edge_num:
                    n = select_edge_num - id_last_col + m - 1
                    return m, n
            return None

        return crowdbt_select(self.num_workers, self.all_nodes, self.all_edges, self.used_edges, self.matrix)

    def result_inference(self):
        return crowdbt_inference(self.all_nodes, self.all_edges, self.matrix)
