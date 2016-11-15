import sys
sys.path.append("..")
import math
import heapq
import numpy as np
import copy
from scipy import optimize

global nodes, edges, M, BaseMatrix
global left_edges, selected_edges, last_score, Hessian, record_inv_matrix, record_current_matrix
global crowdgauss_estimation_edges

global num_workers

normal_pdf = lambda x: 1.0 / math.pow(2 * math.pi, 0.5) * math.exp(-1 * x * x / 2.0)
normal_cdf = lambda x: 0.5 * (1 + math.erf(x / 1.414))


def pre_process(used_edges):
    global nodes, edges, M, BaseMatrix
    M = []
    M = copy.deepcopy(BaseMatrix)
    set_used_edges = set(used_edges)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                M[(nodes[i], nodes[j])] = 0
            elif i != j and (nodes[i], nodes[j]) not in set_used_edges:
                M[(nodes[i], nodes[j])] = 0.08


def likelihood(s):
    global called_count, already_used_edges
    global nodes, M
    SUM = 0
    min_value = 0.00001
    for edge in crowdgauss_estimation_edges:
        SUM += M[edge] * \
               math.log10(normal_cdf(s[edge[0]] - s[edge[1]]) + min_value)
    return -1 * SUM


def crowdgauss_estimation(test_edges):
    global crowdgauss_estimation_edges
    crowdgauss_estimation_edges = copy.deepcopy(test_edges)
    x0 = []
    for item in xrange(len(nodes)):
        #         x0.append(random.random())
        x0.append(1.0)
    res = optimize.minimize(likelihood, x0, method='L-BFGS-B', options={"maxiter": 100})
    score = {}
    for i in xrange(len(nodes)):
        score[nodes[i]] = res.x[i]
    sorted_score = sorted(score.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    result = []
    for item in sorted_score:
        result.append(item[0])
    return score


def computation_cov_matrix(test_edges):
    global selected_edges, last_score, Hessian, M, group, record_inv_matrix

    normal_pdf = lambda x: 1.0 / math.pow(2 * math.pi, 0.5) * math.exp(-1 * x * x / 2.0)
    normal_cdf = lambda x: 0.5 * (1 + math.erf(x / 1.414))

    s = crowdgauss_estimation(test_edges)

    min_value = 0.0001

    for i in xrange(len(nodes)):
        for j in xrange(len(nodes)):
            if i != j:
                if s[nodes[i]] != last_score[nodes[i]] or s[nodes[j]] != last_score[nodes[j]]:
                    para_one = normal_pdf(s[nodes[i]] - s[nodes[j]]) / (
                    normal_cdf(s[nodes[i]] - s[nodes[j]]) + min_value)

                    para_two = normal_pdf(s[nodes[j]] - s[nodes[i]]) / (
                    normal_cdf(s[nodes[j]] - s[nodes[i]]) + min_value)

                    factor1 = M[(nodes[i], nodes[j])] * para_one
                    factor2 = s[nodes[i]] - s[nodes[j]] + para_one

                    factor3 = M[(nodes[j], nodes[i])] * para_two
                    factor4 = s[nodes[j]] - s[nodes[i]] + para_two

                    xsum = factor1 * factor2 + factor3 * factor4
                    Hessian[(nodes[j], nodes[i])] = xsum

    for i in xrange(len(nodes)):
        xsum = 0
        for j in xrange(len(nodes)):
            if j != i:
                xsum += Hessian[(nodes[j], nodes[i])]

        xsum = xsum * -1
        Hessian[(nodes[i], nodes[i])] = xsum

    xarray = []
    for i in xrange(len(nodes)):
        xarray.append([])
        for j in xrange(len(nodes)):
            xarray[-1].append(-1 * Hessian[(i, j)])

    matrix = np.array(xarray)

    if np.linalg.det(matrix) != 0:
        inv_matrix = np.linalg.inv(matrix)
        record_inv_matrix.append(inv_matrix)
        if len(record_inv_matrix) >= 100:
            length = len(record_inv_matrix)
            record_inv_matrix = record_inv_matrix[length - 100:]
    else:
        try:
            inv_matrix = record_inv_matrix[-1]
        except IndexError:
            pass

    'For verification'
    '''
    one_matrix = np.dot(matrix,inv_matrix)
    print one_matrix
    '''

    returned_s = np.array(s.values())

    return returned_s, inv_matrix


def gau_kl(curr_s, curr_matrix, past_s, past_matrix):
    global record_current_matrix

    min_value = 0.01

    if np.linalg.det(curr_matrix) != 0:
        inv_curr_matrix = np.linalg.inv(curr_matrix)
        record_current_matrix.append(inv_curr_matrix)

        if len(record_current_matrix) >= 100:
            record_current_matrix = record_current_matrix[len(record_current_matrix) - 100:]

    else:
        inv_curr_matrix = record_inv_matrix[-1]

    factor1 = np.trace(np.dot(inv_curr_matrix, past_matrix))

    factor2_1 = np.transpose(curr_s - past_s)

    if np.linalg.det(curr_matrix) != 0:
        factor2_2 = np.linalg.inv(curr_matrix)
    else:
        try:
            factor2_2 = record_current_matrix[-1]
        except IndexError:
            pass

    factor2_3 = curr_s - past_s
    factor2_4 = np.dot(factor2_1, factor2_2)

    factor2 = np.dot(factor2_4, factor2_3)

    x1 = math.fabs(np.linalg.det(past_matrix))
    x2 = math.fabs(np.linalg.det(curr_matrix) + min_value)

    factor3 = math.log10(x1 / x2 + min_value)

    result = 0.5 * (factor1 + factor2 - factor3 - len(nodes))

    return result


def max_entropy():
    'Prepare for the computation'
    global selected_edges, edges, last_score, left_edges, num_workers
    entropy = {}
    for edge in edges:
        entropy[edge] = 0

    s, cov_matrix = computation_cov_matrix(selected_edges)
    last_score = copy.deepcopy(s)

    current_s = copy.deepcopy(s)
    current_matrix = copy.deepcopy(cov_matrix)
    #     normal_cdf = lambda x : 0.5 * ( 1 + math.erf( x / 1.414))

    estimated_scores = crowdgauss_estimation(selected_edges)
    esti_not_topk = []
    if len(selected_edges) >= 0.1 * len(edges):
        esti_not_topk = heapq.nsmallest(int(0.5 * len(nodes)), estimated_scores, key=estimated_scores.get)

    'Compute the entropy for each candidate edge'
    for edge in edges:
        if edge[0] in esti_not_topk or edge[1] in esti_not_topk:
            entropy[edge] = -10000
            continue

        if edge in selected_edges:
            entropy[edge] = -10000
            continue

        i = edge[0]
        j = edge[1]

        pro_ij = normal_cdf((current_s[i] - current_s[j]) / (
        1 + current_matrix[(i, i)] + current_matrix[(j, j)] - 2 * current_matrix[(i, j)] + 0.001))
        pro_ji = normal_cdf((current_s[j] - current_s[i]) / (
        1 + current_matrix[(i, i)] + current_matrix[(j, j)] - 2 * current_matrix[(j, i)] + 0.001))

        test_edges = copy.deepcopy(selected_edges)
        test_edges.append(edge)

        if BaseMatrix[edge] < 1:
            M[edge] =  num_workers # try new edge
        else:
            M[edge] = BaseMatrix[edge] + num_workers
        s_new, cov_matrix_new = computation_cov_matrix(test_edges)
        M[edge] = BaseMatrix[edge]

        test_edges.remove(edge)
        test_edges.append((edge[1], edge[0]))

        if BaseMatrix[edge] < 1:
            M[edge] = num_workers  # try new edge
        else:
            M[edge] = BaseMatrix[edge] + num_workers # try new edge

        s_new2, cov_matrix_new2 = computation_cov_matrix(test_edges)

        M[(edge[1], edge[0])] = BaseMatrix[edge]

        expected_entropy = pro_ij * gau_kl(current_s, current_matrix, s_new, cov_matrix_new) + \
                           pro_ji * gau_kl(current_s, current_matrix, s_new2, cov_matrix_new2)

        entropy[edge] = expected_entropy

    rslt = sorted(entropy, key=entropy.get, reverse=True)
    return rslt
'''
    try:
        result = heapq.nlargest(20, entropy, key=entropy.get)

    except Exception, e:
        print 'failure selected'
        result = random.sample(edges, 20)

    cnt = 0
    for e in result:
        print e, entropy[e]
        cnt += 1
        if cnt == 9:
            break
    return result
'''


def apolling_select(pnum_workers, all_nodes, all_edges, ans_edges, ans_matrix):
    'Preparation'
    global selected_edges, nodes, edges, M, BaseMatrix, last_score, Hessian, record_inv_matrix, record_current_matrix
    global left_edges, num_workers
    last_score = {}
    Hessian = {}
    record_inv_matrix = []
    record_current_matrix = []
    nodes, edges = all_nodes, all_edges
    num_workers = pnum_workers
    BaseMatrix = copy.deepcopy(ans_matrix)

    for nodeA in nodes:
        for nodeB in nodes:
            if nodeA == nodeB:
                BaseMatrix[(nodeA, nodeB)] = 0
            elif (nodeA, nodeB) not in BaseMatrix or BaseMatrix[(nodeA, nodeB)] == 0:
                BaseMatrix[(nodeA, nodeB)] = 0.08

    M = copy.deepcopy(BaseMatrix)

    for node in nodes:
        last_score[node] = -1
    for nodeA in nodes:
        for nodeB in nodes:
            Hessian[(nodeA, nodeB)] = 0

    selected_edges = ans_edges

    left_edges = []

    try:
        edges_in_this_round = max_entropy()  # select edge using apolling
        for edge in edges_in_this_round:
            if edge[0] != edge[1] and edge not in ans_edges and (edge[1], edge[0]) not in ans_edges:
                return edge
    except Exception:
        pass

    rest_edges = copy.deepcopy(all_edges)  # select an edge from the rest randomly
    for ans_edge in ans_edges:
        rest_edges.remove(ans_edge)
        rest_edges.remove((ans_edge[1], ans_edge[0]))
    for nd in all_nodes:
        rest_edges.remove((nd, nd))
    if len(rest_edges) != 0:
        edge = rest_edges[np.random.choice(len(rest_edges), 1)[0]]
        return edge

    print 'return none edge'  # all edges've been selected
    return None
