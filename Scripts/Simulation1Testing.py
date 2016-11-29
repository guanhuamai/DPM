import sys
import time

sys.path.append("..")
from PythonSource.Workers import UniformWorkers
from PythonSource.BonusAllocatorLib.IOHmmModel import IOHmmModel
from PythonSource.BonusAllocatorLib.MLSAllocator import MLSAllocator
from PythonSource.BonusAllocatorLib.IOHMMBaseline import IOHMMBaseline
from PythonSource.BonusAllocatorLib.QLearningAllocator import QLearningAllocator
from PythonSource.BonusAllocatorLib.NStepAllocator import NStepAllocator
from PythonSource.BonusAllocatorLib.RandomAllocator import RandomAllocator


def get_majority(answers):
    cnt1 = sum(answers)
    return int(cnt1 >= (len(answers) / 2.0))


def gen_stats(workers, bonus_allocator, num_nodes, base_cost):
    pair_id_seq = list(range((num_nodes ** 2 - num_nodes) / 2))

    num_complete = 0
    t_strt2 = time.time()

    bonus_time = 0
    num_correct_ans = 0
    for _ in pair_id_seq:
        num_complete += 1
        if num_complete % 5000 == 0:
            t_end2 = time.time()
            finish_rate = float(num_complete) / len(pair_id_seq) * 100
            print '%d tasks in total, %d tasks finished...\n' % (len(pair_id_seq), num_complete)
            print 'finish %.2lf%% percent...\n' % finish_rate
            print 'spent time %lf...\n' % (t_end2 - t_strt2)
            t_strt2 = time.time()
        cmp_pair = (0, 1)

        spend = list()
        for worker in workers.available_workers():
            if len(bonus_allocator.hist_qlt_bns[worker]) != 0:
                in_obs = [int(io_pairs[1] > base_cost)  # read bonus history
                          for io_pairs in bonus_allocator.hist_qlt_bns[worker]]
                ou_obs = [io_pairs[0] for io_pairs in bonus_allocator.hist_qlt_bns[worker]]  # read quality history
                spend.append(bonus_allocator.bonus_alloc(in_obs, ou_obs))
            else:
                spend.append(bonus_allocator.bonus_alloc(None, None))
        # print sum(spend)

        workers.publish_questions(workers.available_workers(), cmp_pair, spend)  # publish questions to workers

        answers = workers.collect_answers()  # collect answers from workers
        majority_vote = get_majority(answers)  # calculate the majority answer

        bonus_allocator.update(workers.available_workers(), answers, spend, majority_vote)
        bonus_allocator.strip_hist(workers.available_workers())

        bonus_time += sum([int(spend[worker] > base_cost and  # the task was given bonus
                           answers[worker] == majority_vote)  # the worker has provided a correct answer
                           for worker in workers.available_workers()])

        num_correct_ans += sum([int(answers[worker] == majority_vote)  # checked correctness..
                               for worker in workers.available_workers()])

    return num_correct_ans, bonus_time

if __name__ == '__main__':

    base_costs = 45
    bonus = 5
    num_workers = 5
    num_nd = 1000

    model_file = sys.argv[1]
    weights = [0, float(sys.argv[2]), 0.1]
    model = IOHmmModel()
    model.read_model(model_file)
    uni_workers = UniformWorkers(5, base_costs, bonus)
    print weights
    print model_file
    for t in [10, 20, 30, 40, 50]:
        bonus_allocators = [lambda: IOHMMBaseline(num_workers, base_cost=base_costs, bns=bonus,
                                                  t=t, weights=weights),
                            lambda: MLSAllocator(num_workers, base_cost=base_costs, bns=bonus,
                                                 t=t, weights=weights),
                            lambda: NStepAllocator(num_workers, base_cost=base_costs, bns=bonus,
                                                   t=t, weights=weights),
                            lambda: QLearningAllocator(num_workers, base_cost=base_costs, bns=bonus,
                                                       t=t, weights=weights),
                            lambda: RandomAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t, p=1),
                            lambda: RandomAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t, p=0),
                            lambda: RandomAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t, p=0.5)]
        for i in range(len(bonus_allocators)):
            bns_allocator = bonus_allocators[1]()
            bns_allocator.train(model=model)
            stats = gen_stats(workers=uni_workers, bonus_allocator=bns_allocator,
                              num_nodes=num_nd, base_cost=base_costs)
            utility = stats[0] * weights[1] - stats[1] * weights[2]
            with open(model_file + '.' + sys.argv[2], 'a') as f:
                f.write(type(bns_allocator).__name__ + '\t' + str(stats[0]) +
                        '\t' + str(stats[1]) + '\t' + str(utility) + '\n')
