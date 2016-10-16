from matlab import engine
from ApollingEngine import *
import time


weights = [0, 1, 23]  #utility weight of different performance and cost, bad: 0, good: 1,  the last bit is the weight of the cost

answerMatrix = []

crwSrcEngine = ApollingEngine()
matlabEngine = engine.start_matlab()



def expectUtility(workerID, prfrmncMatrix, cost, bonus, isBonus, weights):
    utility = 0
    for i in range(len(prfrmncMatrix[workerID][isBonus])):
        prob = prfrmncMatrix[workerID][isBonus][i]
        utility += prob * weights[i]
    utility -= negUtility(cost, bonus, isBonus, weights)
    return utility

def negUtility(cost, bonus, isBonus, weights):  #calculate the negative effect that the cost will bring to the requester's utility
    utility = 0
    utility -= (cost + isBonus * bonus) * weights[-1]
    return utility

def publishQuestions(workerID, cmpPair, salary):  #implement this function with MTurk api
    print "we allocate the %dth  worker (%d, %d) compare pair " \
          "question with %lf cents" %(workerID, cmpPair[0], cmpPair[1], salary)

def collectAnswers():  #implement this function with MTurk api
    answers = []
    return answers

def update(cmpPair, answers, input, answerMatrix, obsMatrix, usedEdges):

#get the vote result and adjust the cmpPair(no need to return this stuff)
    cmpCnt = [0 for x in range(2)]
    for answer in answers:
        cmpCnt[answer] += 1
    majority = 0
    if(cmpCnt[1] > cmpCnt[0]): majority = 1
    if majority != 1:
        cmpPair = (cmpPair[1], cmpPair[0])

#update answerMatrix with answers
    try:
        answerMatrix[cmpPair] += 1
    except KeyError:
        answerMatrix[cmpPair] = 1

# update obsMatrix with answers
    for i in range(len(obsMatrix)):
        obsMatrix[i].append((int(answers[i] == majority), input))

# update usedEdges with answers
    usedEdges.append(cmpPair)




def topK(budget, allNodes, usedEdges, allEdges, answerMatrix, obsMatrix):
    prfrmncMatrix = []
    cost = 5#5 cent will be given to the worker if the task is finished
    bonus = 2#2 cent will be given to the worker if we want to reward the worker
    nstates = 3
    ostates = 3
    numiter = 1000

    while(budget > 0):
        cmpPair = crwSrcEngine.pairSelection(allNodes, usedEdges, allEdges, answerMatrix)
        for workerID in obsMatrix:
            expUtlT = expectUtility(workerID, prfrmncMatrix, cost, bonus, True, weights) #expect utility if given certain bonus
            expUtlF = expectUtility(workerID, prfrmncMatrix, cost, bonus, False, weights)#expect utility if not given certain bonus
            spend = cost + (expUtlT > expUtlF) * bonus
            budget -= spend
        if budget >= 0:
            for workerID in obsMatrix:
                publishQuestions(workerID, cmpPair, spend)

                # replace here with stop_event.is_set(), just wait until all the worker finish their tasks
                time.sleep(5)

                answers = collectAnswers()

                update(cmpPair, answers, input, answerMatrix, obsMatrix, usedEdges)

                oObs = [ioPairs[0] for sequence in obsMatrix for ioPairs in sequence] # output observations of every sequences
                iObs = [ioPairs[0] for sequence in obsMatrix for ioPairs in sequence] # input observations of every sequences
                iohmmModel, prfrmncMatrix = matlabEngine.iohmmTraining(oObs, iObs, nstates, ostates, numiter)

        else:
            print 'budget not enough!\n'
            break



    return crwSrcEngine.resultInference(allNodes, allEdges, answerMatrix)
        

