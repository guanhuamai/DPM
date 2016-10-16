from ApollingInference import *
from ApollingSelect import *


class ApollingEngine(object):
    def __init__(self):
        print 'init an apolling engine'

    def pairSelection(self, allNodes, usedEdges, allEdges, matrix):
        return apollingSelect(allNodes, usedEdges, allEdges, matrix)

    def resultInference(self, allNodes, allEdges, matrix):
        return apollingInference(allNodes, allEdges, matrix)