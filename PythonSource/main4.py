from DecisionMakerLib.Crowdbt import Crowdbt
import sys


def find_pair_by_id(pair_id, num_nodes):
    for m in range(num_nodes):
        id_last_col = m * (m + 1) / 2 - 1
        if id_last_col >= pair_id:
            n = pair_id - id_last_col + m - 1
            return m, n


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


if __name__ == '__main__':
    fpath = sys.argv[1]

    num_nd = 1000
    num_workers = 5
    base_cost = 45
    bns = 3

    num_ans_pair = 0
    bns_time = 0
    total_num_correct_ans = 0
    pair_list = []
    cnt_list = []

    statistics = {}
    with open(fpath, 'r') as f:

        lines = f.readlines()
        for line in lines:

            line = line.split('\t')
            if line[0] == 'WorkerType':

                num_ans_pair = 0
                bns_time = 0
                total_num_correct_ans = 0
                pair_list = []
                cnt_list = []
            else:
                worker_type = line[0]
                bonus_type = line[1]
                cmp_pair = find_pair_by_id(int(line[2]), num_nd)
                bns_time += (int(line[3]) - num_workers * base_cost) / bns
                num_ans_pair += 1
                num_correct_ans = int(line[4])
                time_spend = float(line[5])

                total_num_correct_ans += num_correct_ans

                pair_list.append(cmp_pair)
                pair_list.append((cmp_pair[1], cmp_pair[0]))
                cnt_list.append(num_workers - num_correct_ans)
                cnt_list.append(num_correct_ans)

            if num_ans_pair % 49950 == 0 and num_ans_pair != 0:
                cbt_eng = Crowdbt(5, num_nd)
                cbt_eng.matrix = dict(zip(pair_list, cnt_list))
                print 'inferencing %.2lf%%...\n' % (float(num_ans_pair) / 4995)
                # seq = cbt_eng.result_inference()
                # print seq[0:10]

                # prec, recal = cal_precision_recall(seq, range(num_nd), 10)

                util = 0 * (num_ans_pair - total_num_correct_ans) + \
                    0.15 * total_num_correct_ans - \
                    0.05 * bns_time

                cost = bns_time * bns

                s_id = num_ans_pair / 49950
                try:
                    statistics[s_id].append((num_ans_pair, total_num_correct_ans, bns_time, cost, util))
                except KeyError:
                    statistics[s_id] = []
                    statistics[s_id].append((num_ans_pair, total_num_correct_ans, bns_time, cost, util))

            # elif num_ans_pair % 5000 == 0:
            #     print num_ans_pair

    with open(fpath + '.result', 'w') as f:
        # statistics = map(lambda stats: reduce(lambda x, y: str(x) + '\t' + str(y), stats) + '\n',  statistics)
        for stat in statistics:
            stat = reduce(lambda x, y: [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4]],
                          statistics[stat])
            stat = map(lambda x: x / 30, stat)
            stat = reduce(lambda x, y: str(x) + '\t' + str(y), stat) + '\n'
            f.write(stat)
