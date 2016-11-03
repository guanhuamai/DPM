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


def update_hist(hist_qlt_bns, workers, answers, spend, majority_vote):
    for worker in workers:
        try:
            hist_qlt_bns[worker].append((int(answers[worker] == majority_vote), spend[worker]))
        except KeyError:
            hist_qlt_bns[worker] = []
            hist_qlt_bns[worker].append((int(answers[worker] == majority_vote), spend[worker]))


def update_answers(matrix, cmp_pair, answers):
    for answer in answers:
        tmp_pair = cmp_pair
        if answer == 0:
            tmp_pair = (cmp_pair[1], cmp_pair[0])
        try:
            matrix[tmp_pair] += 1
        except KeyError:
            matrix[tmp_pair] = 1


def top_k(base_cost, bns, bonus_allocator, decision_maker, workers, runlog):
    hist_qlt_bns = {}  # note down all workers history, each history is a list of tuples. tuple = (quality, bns)
    matrix = {} # note down all votes, shape = num_nodes * num_nodes

    num_periter = ((len(decision_maker.all_edges) - len(decision_maker.all_nodes)) / 2) / 5

    while (len(decision_maker.matrix) + len(decision_maker.all_nodes)) < len(decision_maker.all_edges):

        cmp_pair = decision_maker.pair_selection()  # choose a question pair for publishing

        if cmp_pair is not None:

            spend = list()  # allocate bonus to worker according to the bonus policy
            for worker in workers.available_workers():
                try:
                    in_obs = [int(io_pairs[1] > base_cost)  # read bonus history
                              for io_pairs in hist_qlt_bns[worker]]
                    ou_obs = [io_pairs[0] for io_pairs in hist_qlt_bns[worker]]  # read quality history
                    spend.append(bonus_allocator.bonus_alloc(in_obs, ou_obs))

                except KeyError:
                    spend.append(bonus_allocator.bonus_alloc(None, None))

            print 'cost: %d\n' %sum(spend)

            workers.publish_questions(workers.available_workers(), cmp_pair, spend)  # publish questions to workers

            answers = workers.collect_answers()  # collect answers from workers

            majority_vote = get_majority(answers)  # calculate the majority answer

            update_hist(hist_qlt_bns, workers.available_workers(), answers, spend, majority_vote)
            update_answers(matrix, cmp_pair, answers)

            decision_maker.update(cmp_pair, answers)
            bonus_allocator.update(workers.available_workers(), answers,
                                   spend, majority_vote, 10)  # train new iohmm model to evaluate workers

        if (len(matrix) / 2) % num_periter == 0:
            print 'write log at %lf' %(len(matrix) / float((len(decision_maker.all_edges)
                                                            - len(decision_maker.all_nodes))))
            runlog.write(str(hist_qlt_bns) + '\n')
            runlog.write(str(matrix) + '\n')

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
        top_k(base_cost, bns, bns_allocator, dec_maker, simworkers, expelog)


    # bns_allocator = IOHMMBaseline(num_workers, base_cost=base_cost, bns=bns)
    # bns_allocator.set_parameters(numitr=500)
    # dec_maker = Crowdbt(num_workers, num_nd)
    # simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    # with open('baselinelog', 'w') as expelog:
    #     top_k(bns_allocator, dec_maker, simworkers, expelog)
    #
    # bns_allocator = QLearningAllocator(num_workers, base_cost=base_cost, bns=bns)
    # bns_allocator.set_parameters(numitr=500)
    # dec_maker = Crowdbt(num_workers, num_nd)
    # simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    # with open('qlearnlog', 'w') as expelog:
    #     top_k(bns_allocator, dec_maker, simworkers, expelog)
    #
    # bns_allocator = RandomAllocator(num_workers, base_cost=base_cost, bns=bns)
    # bns_allocator.set_parameters(p=0.5)
    # dec_maker = Crowdbt(num_workers, num_nd)
    # simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    # with open('randomlog', 'w') as expelog:
    #     top_k(bns_allocator, dec_maker, simworkers, expelog)
    #
    # bns_allocator = NStepAllocator(num_workers, base_cost=base_cost, bns=bns)
    # bns_allocator.set_parameters(numitr=500)
    # dec_maker = Crowdbt(num_workers, num_nd)
    # simworkers = SimulationWorkers(num_workers, "iohmm", base_cost=base_cost, bns=bns)
    # with open('nsteplog', 'w') as expelog:
    #     top_k(bns_allocator, dec_maker, simworkers, expelog)
