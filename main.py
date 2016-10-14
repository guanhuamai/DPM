



answer_matrix = [][]

def publish_questions(budget):
    while(budget > 0):
        select = pairSelection()
        price  = HMMPricing(select, budget)
        answer = publish_q(select, price)
        winner = answer[0]
        loser  = answer[1]
        answer_matrix[winner][loser] += 1
        updateHMM(answer, price)
    return resultInference(answer_matrix)
        

