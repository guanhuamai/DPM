import time
import numpy as np
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


def cal_precision_recall(rslt_seq, ground_truth, k):
    ak = rslt_seq[:k]  # crowds' top k answers
    tk = ground_truth[:k]  # ground truth of top k answers
    ak_inter_tk = filter(lambda t_a: t_a in ak, tk)

    cnt_same = 0
    for oi in ak_inter_tk:
        for oj in ak_inter_tk:
            if oi == oj:
                continue
            ai = ak.index(oi)
            aj = ak.index(oj)
            ti = tk.index(oi)
            tj = tk.index(oj)
            cnt_same += int(ai < aj and ti < tj)
    precision = float(cnt_same) / (k * (k + 1) / 2.0)
    recall = float(len(ak_inter_tk)) / float(k)
    return precision, recall


def do_experiment(workers, bonus_allocator, decision_maker, base_cost, bns):
    rslts = list()

    hist_qlt_bns = {}  # note down all workers history, each history is a list of tuples. tuple = (quality, bns)
    matrix = {}  # note down all votes, shape = num_nodes * num_nodes

    num_periter = ((len(decision_maker.all_edges) - len(decision_maker.all_nodes)) / 2) / 5

    time_strt = time.time()
    cost = 0
    cmp_pair = (-1, -1)

    def record_result():
        selection_rate = len(decision_maker.used_edges) * 2 / float((len(decision_maker.all_edges) -
                                                                     len(decision_maker.all_nodes)))
        print 'write log at %lf' % selection_rate

        rslt_seq = decision_maker.result_inference()

        num_correct_ans = 0
        num_total_ans = 0
        for _cmp_pair in matrix:
            num_total_ans += matrix[_cmp_pair]
            if _cmp_pair[0] < _cmp_pair[1]:
                num_correct_ans += matrix[_cmp_pair]
        util = bonus_allocator.weights[0] * (num_total_ans - num_correct_ans) + \
            bonus_allocator.weights[1] * num_correct_ans - \
            bonus_allocator.weights[2] * (cost - base_cost * num_total_ans) / bns

        time_end = time.time()
        time_spend = time_end - time_strt

        for k in range(2, 11, 2):
            name_worker_model = type(workers).__name__
            name_allocator_model = type(bonus_allocator).__name__
            name_decmaker_model = type(decision_maker).__name__
            precision_recall = cal_precision_recall(rslt_seq, range(len(decision_maker.all_nodes)), k)
            rslts.append((name_worker_model, name_allocator_model, name_decmaker_model,
                          selection_rate, k, cost, util, precision_recall[0], precision_recall[1],
                          time_spend))

    while cmp_pair is not None:
        if (len(decision_maker.used_edges)) % num_periter == 0 and len(matrix) != 0:
            record_result()

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

            print 'spend: %d\n' % sum(spend)
            cost += sum(spend)

            workers.publish_questions(workers.available_workers(), cmp_pair, spend)  # publish questions to workers

            answers = workers.collect_answers()  # collect answers from workers

            majority_vote = get_majority(answers)  # calculate the majority answer

            update_hist(hist_qlt_bns, workers.available_workers(), answers, spend, majority_vote)
            update_answers(matrix, cmp_pair, answers)

            decision_maker.update(cmp_pair, answers)
            bonus_allocator.update(workers.available_workers(), answers,
                                   spend, majority_vote)  # train new iohmm model to evaluate workers
    return rslts


def gen_train_data(base_cost, bns, workers, num_tasks):
    _train_data = [[] for _ in range(len(workers.available_workers))]
    for _i in range(num_tasks):
        spend = map(lambda x: base_cost + bns * x,
                    np.random.choice(2, len(workers.available_workers()), p=[0.8, 0.2]))
        workers.publish_questions(workers.available_workers(), (0, 1), spend)  # publish questions to workers

        answers = workers.collect_answers()  # collect answers from workers

        majority_vote = get_majority(answers)  # calculate the majority answer

        for worker_id in workers.available_workers():
            _train_data[worker_id].append((int(answers[worker_id] == majority_vote), spend[worker_id]))
    return _train_data

if __name__ == '__main__':

    num_nd = 1000
    num_workers = 20
    base_costs = 5
    bonus = 2
    t = 10

    worker_models = list()     # 3 worker models: uniform distribution, beta distribution, iohmm distribution
    bonus_allocators = list()  # 5 bonus models(actually 7): baseline, mls-mdp, nstep-lookahead, qlearning
    decision_makers = list()   # 2 decision models: apolling, crowdbt
    num_questions = num_nd     # set to be 10 for now

    worker_models = [lambda:UniformWorkers(num_workers, base_cost=base_costs, bns=bonus),
                     lambda:BetaWorkers(num_workers, base_cost=base_costs, bns=bonus),
                     lambda:IOHmmWorkers(num_workers, base_cost=base_costs, bns=bonus)]
    bonus_allocators = [lambda:IOHMMBaseline(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda:MLSAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda:NStepAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda:QLearningAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t),
                        lambda:RandomAllocator(num_workers, base_cost=base_costs, bns=bonus, t=t)]
    decision_makers = [lambda:Crowdbt(num_workers, num_nd),
                       lambda:Apolling(num_workers, num_nd)]

    worker_model = IOHmmWorkers(3000, base_cost=base_costs, bns=bonus)
    train_data = gen_train_data(base_costs, bonus, worker_model, 50)

    with open('rslt log', 'w') as log_file:
        log_file.write('WorkerType\tBonusType\tSelect&InferenceType\tSelectionRate\t'
                       'K\tTotalCost\tUtility\tPrecision\tRecall\tTimeSpend\n')
        for i in range(len(bonus_allocators)):
            bns_allocator = bonus_allocators[i]()
            bns_allocator.train(train_data=train_data)
            results = do_experiment(worker_models[2](), bonus_allocators[i](),
                                    decision_makers[1](), base_costs, bonus)
            for rslt in results:
                str_rslt = reduce(lambda x, y: str(x) + '\t' + str(y), rslt)
                log_file.write(str_rslt + '\n')
