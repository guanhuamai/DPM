import numpy as np
import sys
from PythonSource.BonusAllocatorLib.IOHmmModel import IOHmmModel
from PythonSource.Workers import UniformWorkers


def get_majority(answers):
    cnt1 = sum(answers)
    return int(cnt1 >= (len(answers) / 2.0))


def gen_train_data(base_cost, bns, workers, num_tasks, prob):
    print 'bonus with %.2lf' % prob
    _train_data = [[] for _ in workers.available_workers()]
    for _i in range(num_tasks):
        spend = map(lambda x: base_cost + bns * x,
                    np.random.choice(2, len(workers.available_workers()), p=[(1 - prob), prob]))
        workers.publish_questions(workers.available_workers(), (0, 1), spend)  # publish questions to workers

        answers = workers.collect_answers()  # collect answers from workers

        majority_vote = get_majority(answers)  # calculate the majority answer

        for worker_id in workers.available_workers():
            _train_data[worker_id].append((int(answers[worker_id] == majority_vote), spend[worker_id]))
    return _train_data


if __name__ == '__main__':
    num_worker = int(sys.argv[1])
    p1 = float(sys.argv[2])

    base_costs = 45
    bonus = 5

    worker_model = UniformWorkers(num_worker, base_cost=base_costs, bns=bonus)
    train_data = gen_train_data(base_costs, bonus, worker_model, 50, p1)
    iohmmmodel = IOHmmModel()
    iohmmmodel.train(train_data, base_costs)
    iohmmmodel.write_model('iohmm.model.' + str(num_worker) + '.%d%%' % int(p1 * 100))

    if len(sys.argv) == 4:
        p2 = float(sys.argv[3])
        train_data = gen_train_data(base_costs, bonus, worker_model, 50, p2)
        iohmmmodel = IOHmmModel()
        iohmmmodel.train(train_data, base_costs)
        iohmmmodel.write_model('iohmm.model.' + str(num_worker) + '.%d%%' % int(p2 * 100))
