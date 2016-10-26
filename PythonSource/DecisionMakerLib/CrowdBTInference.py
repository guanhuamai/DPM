import math
from scipy.optimize import minimize
global nodes, edges, Matrix


def CrowdBT_optimization(s):
    SUM = 0
    yitak = 0.9
    balance = 0.000001
    try:
        for edge in edges:
            i = edge[0]
            j = edge[1]

            factor1 = yitak * \
                math.exp(s[i]) / (math.exp(s[i]) + math.exp(s[j]) + balance)
            factor2 = (
                1 - yitak) * math.exp(s[j]) / (math.exp(s[i]) + math.exp(s[j]) + balance)
            SUM += math.log10(factor1 + factor2)

        for i in range(len(nodes)):
            factor3 = math.exp(
                1.0) / (math.exp(1.0) + math.exp(s[i]) + balance)
            factor4 = math.exp(
                s[i]) / (math.exp(1.0) + math.exp(s[i]) + balance)
            SUM += 0.5 * (math.log10(factor3) + math.log10(factor4))
    except Exception:
        return -1 * SUM

    return -1 * SUM


def optimization_crowdbt():
    x0 = [1.0] * len(nodes)
    res = minimize(
        CrowdBT_optimization, x0, method='BFGS', options={"maxiter": 100})
    score = {}
    for i in range(len(nodes)):
        score[i] = res.x[i]
    sorted_score = sorted(
        score.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    return [item[0] for item in sorted_score]


def apolling_inference(allNodes, allEdges, matrix):
    global nodes, edges, Matrix
    nodes, tmp_edges, Matrix = allNodes, allEdges, matrix
    edges = []
    for edge in tmp_edges:
        try:
            if Matrix[edge] > Matrix[(edge[1], edge[0])]:
                edges.append(edge)
        except KeyError:
            Matrix[edge] = 0
