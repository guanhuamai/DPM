import time
import numpy as np
from BonusAllocatorLib.IOHmmModel import IOHmmModel
from BonusAllocatorLib.MLSAllocator import MLSAllocator
from BonusAllocatorLib.IOHMMBaseline import IOHMMBaseline
from BonusAllocatorLib.QLearningAllocator import QLearningAllocator
from BonusAllocatorLib.NStepAllocator import NStepAllocator
from BonusAllocatorLib.RandomAllocator import RandomAllocator
from DecisionMakerLib.Crowdbt import Crowdbt
from DecisionMakerLib.Apolling import Apolling
from Workers import UniformWorkers
from Workers import BetaWorkers
from Workers import IOHmmWorkers


def find_pair_by_id(pair_id, num_nodes):
    for m in range(num_nodes):
        id_last_col = m * (m + 1) / 2 - 1
        if id_last_col > pair_id:
            n = pair_id - id_last_col + m - 1
            return m, n


def get_majority(answers):
    cnt1 = sum(answers)
    return int(cnt1 >= (len(answers) / 2))


def gen_simulation_data(workers, bonus_allocator, num_nodes, base_cost, bns):
    sim_data = list()
    pair_id_seq = list(range((num_nodes ** 2 - num_nodes) / 2))
    np.random.shuffle(pair_id_seq)
    for pair_id in pair_id_seq:
        cmp_pair = find_pair_by_id(pair_id, num_nodes)

        bns_strt = time.time()
        spend = list()
        for worker in workers.available_workers():
            try:
                in_obs = [int(io_pairs[1] > base_cost)  # read bonus history
                          for io_pairs in bonus_allocator.hist_qlt_bns[worker]]
                ou_obs = [io_pairs[0] for io_pairs in bonus_allocator.hist_qlt_bns[worker]]  # read quality history
                spend.append(bonus_allocator.bonus_alloc(in_obs, ou_obs))
            except KeyError:
                spend.append(bonus_allocator.bonus_alloc(None, None))
        bns_end = time.time()

        workers.publish_questions(workers.available_workers(), cmp_pair, spend)  # publish questions to workers

        answers = workers.collect_answers()  # collect answers from workers

        majority_vote = get_majority(answers)  # calculate the majority answer

        bonus_allocator.update(workers.available_workers(), answers, spend, majority_vote)
        bonus_allocator.strip_hist(workers.available_workers())

        cost = sum([base_cost +
                    bns * int(spend[worker] > base_cost and  # the task was given bonus
                              answers[worker] == majority_vote)  # the worker has provided a correct answer
                    for worker in workers.available_workers()])

        num_correct_ans = sum([int(answers[worker] == majority_vote)
                               for worker in workers.available_workers()])

        time_spend = bns_end - bns_strt

        name_worker_model = type(workers).__name__
        name_allocator_model = type(bonus_allocator).__name__

        sim_data.append((name_worker_model, name_allocator_model,
                         pair_id, cost, num_correct_ans, time_spend))
    return sim_data

if __name__ == '__main__':
    num_nd = 1000
    num_workers = 20
    base_costs = 5
    bonus = 2
    t = 10

    worker_models = list()  # 3 worker models: uniform distribution, beta distribution, iohmm distribution
    bonus_allocators = list()  # 5 bonus models(actually 7): baseline, mls-mdp, nstep-lookahead, qlearning
    decision_makers = list()  # 2 decision models: apolling, crowdbt
    num_questions = num_nd  # set to be 10 for now

    worker_models = [lambda: UniformWorkers(num_workers, base_cost=base_costs, bns=bonus),
                     lambda: BetaWorkers(num_workers, base_cost=base_costs, bns=bonus),
                     lambda: IOHmmWorkers(num_workers, base_cost=base_costs, bns=bonus)]
    bonus_allocators = [lambda: IOHMMBaseline(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda: MLSAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda: NStepAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda: QLearningAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda: RandomAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t)]
    decision_makers = [lambda: Crowdbt(num_workers, num_nd),
                       lambda: Apolling(num_workers, num_nd)]

    iohmmmodel = IOHmmModel()
    iohmmmodel.read_model('iohmm.model.500')

    for i in range(len(bonus_allocators)):
        bns_allocator = bonus_allocators[i]()
        bns_allocator.train(model=iohmmmodel)
        results = gen_simulation_data(worker_models[2](), bns_allocator,
                                      num_nd, base_costs, bonus)

        with open('rslt log500', 'w+') as log_file:
            log_file.write('WorkerType\tBonusType\tpair\t'
                           'TotalCost\tCorrectAnswers\tTimeSpend\n')
            for res in results:
                str_res = reduce(lambda x, y: str(x) + '\t' + str(y), res) + '\n'
                log_file.write(str_res)
