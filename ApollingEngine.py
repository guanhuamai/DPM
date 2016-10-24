from ApollingInference import *
from ApollingSelect import *
import numpy as np

class ApollingEngine(object):
    def __init__(self):
        print 'init an apolling engine'

    def pairSelection(self, allNodes, usedEdges, allEdges, matrix):
        try:
            return apollingSelect(allNodes, usedEdges, allEdges, matrix)
        except:
            print 'failure selected'
            while True:
                edge = np.random.sample(allEdges)
                if edge not in usedEdges:
                    return edge


    def resultInference(self, allNodes, allEdges, matrix):
        return apollingInference(allNodes, allEdges, matrix)