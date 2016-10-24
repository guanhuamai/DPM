from ApollingInference import *
from ApollingSelect import *
import numpy as np

class ApollingEngine(object):
    def __init__(self):
        print 'init an apolling engine'

    def pairSelection(self, numWorkers, allNodes, usedEdges, allEdges, matrix):
        try:
            return apollingSelect(numWorkers, allNodes, usedEdges, allEdges, matrix)
        except:
            while True:
                edge = allEdges[np.random.choice(len(allEdges), 1)[0]]
                if edge not in usedEdges:
                    print 'failure selected', edge
                    return edge


    def resultInference(self, allNodes, allEdges, matrix):
        return apollingInference(allNodes, allEdges, matrix)