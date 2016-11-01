from BonusAllocatorLib.MLSAllocator import MLSAllocator
from BonusAllocatorLib.IOHMMBaseline import IOHMMBaseline
from BonusAllocatorLib.QLearningAllocator import QLearningAllocator
from BonusAllocatorLib.NStepAllocator import NStepAllocator
from BonusAllocatorLib.RandomAllocator import RandomAllocator
from DecisionMakerLib.Crowdbt import Crowdbt
from Workers import SimulationWorkers


def get_majority(answers):
    cnt1 = sum(answers)
    return int(cnt1 >= (len(answers) / 2))


def top_k(bonus_allocator, decision_maker, workers, runlog):
    num_periter = len(decision_maker.all_edges) / 10
    print num_periter
    while len(decision_maker.matrix) < len(decision_maker.all_edges):
        cmp_pair = decision_maker.pair_selection()  # choose a question pair for publishing

        spend = bonus_allocator.bonus_alloc()  # allocate bonus to workers according to the bonus policy

        workers.publish_questions(workers.available_workers(), cmp_pair, spend)  # publish questions to workers
        answers = workers.collect_answers()
        majority_vote = get_majority(answers)

        decision_maker.update(cmp_pair, answers)
        bonus_allocator.worker_evaluate(answers, spend, majority_vote)  # train new iohmm model to evaluate workers

        if len(decision_maker.matrix) % num_periter == 0:
            print 'write log at %lf' %(len(decision_maker.matrix) / float(len(decision_maker.all_edges)))
            runlog.write(str(bonus_allocator.hist_qlt_bns) + '\n')
            runlog.write(str(decision_maker.matrix) + '\n')





if __name__ == '__main__':

    num_nd = 10
    num_workers = 20
    base_cost = 5
    bns = 2

    bns_allocator = MLSAllocator(num_workers, base_cost=base_cost, bns=bns)
    bns_allocator.set_parameters(numitr=500)
    dec_maker = Crowdbt(num_workers, num_nd)
    simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    with open('mlslog', 'w') as expelog:
        top_k(bns_allocator, dec_maker, simworkers, expelog)


    bns_allocator = IOHMMBaseline(num_workers, base_cost=base_cost, bns=bns)
    bns_allocator.set_parameters(numitr=500)
    dec_maker = Crowdbt(num_workers, num_nd)
    simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    with open('baselinelog', 'w') as expelog:
        top_k(bns_allocator, dec_maker, simworkers, expelog)

    bns_allocator = QLearningAllocator(num_workers, base_cost=base_cost, bns=bns)
    bns_allocator.set_parameters(numitr=500)
    dec_maker = Crowdbt(num_workers, num_nd)
    simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    with open('qlearnlog', 'w') as expelog:
        top_k(bns_allocator, dec_maker, simworkers, expelog)

    bns_allocator = RandomAllocator(num_workers, base_cost=base_cost, bns=bns)
    bns_allocator.set_parameters(p=0.5)
    dec_maker = Crowdbt(num_workers, num_nd)
    simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    with open('randomlog', 'w') as expelog:
        top_k(bns_allocator, dec_maker, simworkers, expelog)

    bns_allocator = NStepAllocator(num_workers, base_cost=base_cost, bns=bns)
    bns_allocator.set_parameters(numitr=500)
    dec_maker = Crowdbt(num_workers, num_nd)
    simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    with open('nsteplog', 'w') as expelog:
        top_k(bns_allocator, dec_maker, simworkers, expelog)
