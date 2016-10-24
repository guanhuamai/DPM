import sys
sys.path.append("..")
import random
import math
from scipy.stats import norm
from scipy import optimize
#import mall_process
global nodes, edges, M, BaseMatrix

def likelihood(s):
    global nodes, M
    SUM = 0
    min_value = 0.00001
    for edge in edges:
        SUM += M[edge] * \
            math.log10(norm.cdf(s[edge[0]] - s[edge[1]]) + min_value)

    return -1 * SUM


def inference():
    x0 = []
    for item in xrange(len(nodes)):
        x0.append(random.random())
    res = optimize.minimize(
        likelihood, x0, method='BFGS', options={"maxiter": 100})
    score = {}
    for i in xrange(len(nodes)):
        score[nodes[i]] = res.x[i]
    sorted_score = sorted(
        score.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    result = []
    for item in sorted_score:
        result.append(item[0])
    print result
    return result


def apollingInference(allNodes, allEdges, matrix):
    global nodes, edges, M
    nodes, tmpedges, M = allNodes, allEdges, matrix
    edges = []
    for edge in tmpedges:
        try:
            if M[edge] > M[(edge[1], edge[0])]:
                edges.append(edge)
        except KeyError:
            M[edge] = 0

    inference()


'''
if __name__ == "__main__":
    allNodes, allEdges, matrix = mall_process.mall_process()
    apollingInference(allNodes, allEdges, matrix)
'''