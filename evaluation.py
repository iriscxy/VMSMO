import sys

from metrics import recall_2at1, recall_at_k_new, precision_at_k, MRR, MAP
import numpy as np
import codecs


def evaluation(pred_scores, true_scores, samples=10):
    '''
    :param pred_scores:     list of scores predicted by model
    :param true_scores:     list of ground truth labels, 1 or 0
    :return:
    '''

    num_sample = int(len(pred_scores) / samples) # 1 positive and 9 negative
    # score_list = np.argmax(np.split(np.array(pred_scores), num_sample, axis=0), 1)
    # logit_list = np.argmax(np.split(np.array(true_scores), num_sample, axis=0), 1)
    recall_2_1 = recall_2at1(np.array(true_scores), np.array(pred_scores))
    recall_at_1 = recall_at_k_new(np.array(true_scores), np.array(pred_scores), 1)
    recall_at_2 = recall_at_k_new(np.array(true_scores), np.array(pred_scores), 2)
    recall_at_5 = recall_at_k_new(np.array(true_scores), np.array(pred_scores), 5)
    _mrr = MRR(np.array(true_scores), np.array(pred_scores))
    _map = MAP(np.array(true_scores), np.array(pred_scores))
    precision_at_1 = precision_at_k(np.array(true_scores), np.array(pred_scores), k=1)
    # ndcg_at_1 = NDCG(np.array(true_scores), np.array(pred_scores), 1)
    # ndcg_at_2 = NDCG(np.array(true_scores), np.array(pred_scores), 2)
    # ndcg_at_5 = NDCG(np.array(true_scores), np.array(pred_scores), 5)


    print("**********************************")
    print("results..........")
    print('pred_scores: ', len(pred_scores))
    print("MAP: %.3f" % (_map))
    print("MRR: %.3f" % (_mrr))
    print("precision_at_1:  %.3f" % (precision_at_1))
    print("recall_2_1:  %.3f" % (recall_2_1))
    print("recall_at_1: %.3f" % (recall_at_1))
    print("recall_at_2: %.3f" % (recall_at_2))
    print("recall_at_5: %.3f" % (recall_at_5))
    print("**********************************")
    return {
        'MAP': _map,
        'MRR': _mrr,
        'p@1': precision_at_1,
        'r2@1': recall_2_1,
        'r@1': recall_at_1,
        'r@2': recall_at_2,
        'r@5': recall_at_5,
    }


if __name__ == '__main__':
    # pred_scores = []
    # true_scores = []
    # with codecs.open('/home1/liuchang/projects/sticker_chat/code/early.txt', 'r', 'utf-8') as f:
    #     for line in f:
    #         parts = line.strip().split("\t")
    #         pred_scores.append(float(parts[0]))
    #         true_scores.append(int(parts[1]))
    #
    # print(len(pred_scores))
    # print(len(true_scores))
    # k = 10
    # if len(sys.argv) > 2 and sys.argv[2] is not None and sys.argv[2] != '':
    #     k = int(sys.argv[2])
    # evaluation(pred_scores, true_scores, k)
    pred_scores = [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5,
                   0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5]
    true_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    evaluation(pred_scores, true_scores)

