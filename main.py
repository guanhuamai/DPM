from matlab import engine
from ApollingEngine import *
from Workers import SimulationWorkers
import numpy as py


weights = [0, 1, 23]  #utility weight of different performance and cost, bad: 0, good: 1,  the last bit is the weight of the cost

answerMatrix = []

crwSrcEngine = ApollingEngine()
matlabEngine = engine.start_matlab()



def expectUtility(workerID, prfrmncMatrix, cost, bonus, isBonus, weights):
    utility = 0
    if len(prfrmncMatrix[workerID][isBonus]) == 0:
        return py.random.random()
    else:
        for i in range(len(prfrmncMatrix[workerID][isBonus][0])):  #traverse all the different quality
            prob = prfrmncMatrix[workerID][isBonus][0][i]
            utility += prob * weights[i]
    utility -= negUtility(cost, bonus, isBonus, weights)
    return utility

def negUtility(cost, bonus, isBonus, weights):  #calculate the negative effect that the cost will bring to the requester's utility
    return (cost + isBonus * bonus) * weights[-1]



def update(cmpPair, answers, spend, answerMatrix, obsMatrix, usedEdges):

#get the vote result and adjust the cmpPair(no need to return this stuff)
    cmpCnt = [0 for x in range(2)]
    for answer in answers:
        cmpCnt[answer] += 1
    majority = 0
    if(cmpCnt[1] > cmpCnt[0]): majority = 1


# update obsMatrix with answers
    for i in obsMatrix:
        obsMatrix[i].append((int(answers[i] == majority), spend[i]))

#update answerMatrix with answers
    for ans in answers:
        tmpPair = cmpPair
        if ans == 0:
            tmpPair = (cmpPair[1], cmpPair[0])
        try:
            answerMatrix[tmpPair] += 1
        except KeyError:
            answerMatrix[tmpPair] = 1

# update usedEdges with answers
    usedEdges.append(cmpPair)




def topK(budget, allNodes, usedEdges, allEdges, answerMatrix, obsMatrix, num_workers):
    workers = SimulationWorkers(num_workers, "uniform")
    prfrmncMatrix = [[[], []] for i in range(num_workers)]
    cost = 5#5 cent will be given to the worker if the task is finished
    bonus = 2#2 cent will be given to the worker if we want to reward the worker
    nstates = 3
    ostates = 2
    numiter = 1000

    while(budget > 0):
        cmpPair = crwSrcEngine.pairSelection(num_workers, allNodes, usedEdges, allEdges, answerMatrix)
        spend = []
        for workerID in obsMatrix:
            expUtlT = expectUtility(workerID, prfrmncMatrix, cost, bonus, 1, weights) #expect utility if given certain bonus
            expUtlF = expectUtility(workerID, prfrmncMatrix, cost, bonus, 0, weights)#expect utility if not given certain bonus
            spend.append(cost + (expUtlT > expUtlF) * bonus)
            budget -= spend[-1]  # the last spend value
        if budget >= 0:
            print 'budget left', budget
            workers.publish_questions(obsMatrix.keys(), cmpPair, spend)
            answers = workers.collect_answers()
            update(cmpPair, answers, spend, answerMatrix, obsMatrix, usedEdges)
            iObsVec = [[0, 1], [1, 0]]
            oObs = [[ioPairs[0] for ioPairs in obsMatrix[seqid]] for seqid in obsMatrix ]  # output observations of every sequences
            iObs = [[iObsVec[int(ioPairs[1] > cost)] for ioPairs in obsMatrix[seqid]] for seqid in obsMatrix]   # input observations of every sequences
            prfrmncMatrix = matlabEngine.iohmmTraining(oObs, iObs, nstates, ostates, numiter)['result']
        else:
            print 'budget not enough!\n'
            break


    return crwSrcEngine.resultInference(allNodes, allEdges, answerMatrix)
        
if __name__ == '__main__':
    num_nd = 10
    num_workers = 200
    used_edges = []
    all_nodes = [i for i in range(num_nd)]
    all_edges = [(i, j) for i in range(num_nd) for j in range(num_nd)]
    answer_matrix = {}
    obs_matrix = dict(zip([i for i in range(num_workers)], [[] for i in range(num_workers)]))
    topK(10000, all_nodes, used_edges, all_edges, answer_matrix, obs_matrix, num_workers)

