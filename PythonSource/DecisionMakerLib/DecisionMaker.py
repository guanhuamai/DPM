

class DecisionMaker(object):

    def __init__(self, nu_workers, nu_nodes):
        print 'init an base decision maker'
        self.num_workers = nu_workers  # protected variable
        self.all_nodes = [i for i in range(nu_nodes)]
        self.all_edges = [(i, j) for i in range(nu_nodes) for j in range(nu_nodes)]
        self.matrix = {}

    def update(self, *args):
        raise NotImplementedError('Please Implement this method')

    def pair_selection(self, *args):
        raise NotImplementedError('Please Implement this method')

    def result_inference(self, *args):
        raise NotImplementedError('Please Implement this method')
