from BonusAllocatorLib.IOHMMBaseline import IOHMMBaseline
from DecisionMakerLib.Apolling import Apolling
from Workers import SimulationWorkers


def get_majority(answers):
    cnt1 = sum(answers)
    return int(cnt1 >= (len(answers) / 2))


def top_k(budget, bonus_allocator, decision_maker, workers):
    while budget > 0:
        cmp_pair = decision_maker.pair_selection()  # choose a question pair for publishing

        spend = bonus_allocator.bonus_alloc()  # allocate bonus to workers according to the bonus policy
        budget -= sum(spend)

        if budget >= 0:
            print 'budget left', budget
            workers.publish_questions(workers.available_workers(), cmp_pair, spend)  # publish questions to workers
            answers = workers.collect_answers()
            majority_vote = get_majority(answers)

            decision_maker.update(cmp_pair, answers)
            bonus_allocator.worker_evaluate(answers, spend, majority_vote)  # train new iohmm model to evaluate workers
        else:
            print 'budget not enough!\n'
            break
    return decision_maker.result_inference()

if __name__ == '__main__':
    num_nd = 10
    num_workers = 200

    bns_allocator = IOHMMBaseline(num_workers)
    bns_allocator.set_parameters(numitr=200)
    dec_maker = Apolling(num_workers, num_nd)
    simworkers = SimulationWorkers(num_workers, "uniform")

    top_k(20000, bns_allocator, dec_maker, simworkers)
