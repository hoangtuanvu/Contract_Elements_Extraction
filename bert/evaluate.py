""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
# from data_tools.create_dataset import normalize_output_text
# from data_tools.constants import MAP_QUESTIONS


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ''.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction['text']) == normalize_answer(ground_truth[0])
    # return normalize_answer(prediction['text']) == normalize_answer(ground_truth[0]) \
    #        and prediction['start_index'] == ground_truth[1]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def pre_process(a):
    def validate_index(indexes, start_index, end_index, text):
        for [_start_idx, _end_idx, _text] in indexes:
            if _start_idx <= start_index <= _end_idx or _start_idx <= end_index <= _end_idx or _text in text:
                return False

        return True

    b = []
    indexes = []
    for i in range(0, len(a)):
        if i == 0:
            b.append(a[i])
            indexes.append([a[i]['start_index'], a[i]['end_index'], a[i]['text']])
        else:
            if validate_index(indexes, a[i]['start_index'], a[i]['end_index'], a[i]['text']):
                b.append(a[i])
                indexes.append([a[i]['start_index'], a[i]['end_index'], a[i]['text']])

    return b


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0

    ground_truth_map = {}
    prediction_map = {}

    map_result = {}
    for article in dataset:
        # map_ques = {}
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:

                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: [x['text'], x['answer_start']], qa['answers']))
                n_prediction = predictions[qa['id']]
                n_prediction = [item for item in n_prediction if '。' not in item['text']]
                question = qa['question']

                if qa['is_impossible']:
                    continue

                if question in ['実施機関', '質問書提出方法']:
                    continue

                # if question not in map_ques:
                #     map_ques[question] = [[ground_truths, prediction, qa['id']]]
                # else:
                #     map_ques[question].append([ground_truths, prediction, qa['id']])

                if qa['question'] in ['*質問箇所TEL/FAX', '*入札書送付先', '*資格申請送付先', '入札書締切日時',
                                      '質問票締切日時', '開札日時', '資格申請締切日時', '仕様書交付期限']:
                    if qa['question'] not in ground_truth_map:
                        ground_truth_map[qa['question']] = [ground_truths[0]]
                    else:
                        ground_truth_map[qa['question']].append(ground_truths[0])

                    prediction_map[qa['question']] = n_prediction
                else:
                    total += 1
                    acc = metric_max_over_ground_truths(
                        exact_match_score, n_prediction[0], ground_truths)
                    # f1_scr = metric_max_over_ground_truths(
                    #     f1_score, n_prediction, ground_truths)

                    # if acc != 1:
                    #     print("======================================================")
                    #     print(qa['id'])
                    #     print(n_prediction)
                    #     print(ground_truths)
                    exact_match += acc

                    # if acc != 1.0:
                    #     if qa['question'] == '*質問箇所所属/担当者':
                    #         print("====================")
                    #         print(qa['id'])
                    #         print(n_prediction)
                    #         print(ground_truths)

                    if question not in map_result:
                        map_result[question] = [acc]
                    else:
                        map_result[question].append(acc)
                # f1 += f1_scr

                # if qa['question'] in ['仕様書交付期限'] and acc == 1.0:
                #     print("Prediction: {}, Ground_truth: {}".format(predictions[qa['id']], ground_truths[0]))
                #
                # if qa['question'] in ['質問票締切日時', '開札日時', '入札書締切日時', '仕様書交付期限', '資格申請締切日時'] and acc != 1.0:
                #     print("Prediction: {}, Ground_truth: {}".format(predictions[qa['id']], ground_truths[0]))
                #
                # if acc != 1.0:
                #     print("==========================")
                #     print(qa['id'])
                #     print("Prediction: {}, Ground_truth: {}".format(predictions[qa['id']], ground_truths[0]))

        for ques in ground_truth_map:
            n_ground_truths = ground_truth_map[ques]
            n_ground_truths = sorted(n_ground_truths, key=lambda x: x[1])
            n_prediction = prediction_map[ques]
            n_prediction = pre_process(n_prediction)[: len(n_ground_truths)]

            total += 1
            acc = 0.0
            if ques in ["*質問箇所TEL/FAX", "*入札書送付先", "*資格申請送付先"]:
                n_prediction = sorted(n_prediction, key=lambda x: x['start_index'])
                flag = True
                # print(n_prediction)
                # print(n_ground_truths)
                # if len(n_ground_truths) != len(n_prediction):
                #     print(n_prediction)
                #     print(n_ground_truths)

                for i in range(len(n_prediction)):
                    if not (n_ground_truths[i][0] == n_prediction[i]['text']):
                    #     if not (n_ground_truths[i][0] == n_prediction[i]['text'] and n_ground_truths[i][1] ==
                    #             n_prediction[i]['start_index']):
                        flag = False
                        break

                if flag:
                    exact_match += 1.0
                    acc = 1.0
                # else:
                #     if ques == '*質問箇所TEL/FAX':
                #         print("====================")
                #         print(article['title'])
                #         print(prediction_map[ques])
                #         print(n_ground_truths)
            else:
                flag = True

                is_all_in_one = True
                for i in range(len(n_ground_truths)):
                    if n_ground_truths[i][0] not in n_prediction[0]['text']:
                    # if n_ground_truths[i][0] not in n_prediction[0]['text'] or (
                    #         n_ground_truths[0][1] != n_prediction[0]['start_index']):
                        is_all_in_one = False
                        break

                if not is_all_in_one:
                    n_prediction = sorted(n_prediction, key=lambda x: x['start_index'])
                    for i in range(len(n_ground_truths)):
                        if ques == "仕様書交付期限":
                            if (n_ground_truths[i][0] not in n_prediction[i]['text']) and \
                                    (n_prediction[i]['text'] not in n_ground_truths[i][0]):
                                # if not (n_ground_truths[i][0] == n_prediction[i]['text']):
                                # if not (n_ground_truths[i][0] == n_prediction[i]['text'] and n_ground_truths[i][1] ==
                                #         n_prediction[i]['start_index']):
                                flag = False
                                break
                        else:
                            if not (n_ground_truths[i][0] == n_prediction[i]['text']):
                            # if not (n_ground_truths[i][0] == n_prediction[i]['text'] and n_ground_truths[i][1] ==
                            #         n_prediction[i]['start_index']):
                                flag = False
                                break
                # else:
                #     print("====================")
                #     print(n_prediction)
                #     print(n_ground_truths)

                if flag:
                    exact_match += 1.0
                    acc = 1.0
                else:
                    if ques == "質問票締切日時":
                        print("====================")
                        print(prediction_map[ques])
                        print(n_prediction)
                        print(n_ground_truths)

            if ques not in map_result:
                map_result[ques] = [acc]
            else:
                map_result[ques].append(acc)

        ground_truth_map.clear()
        prediction_map.clear()

        # for question, val in map_ques.items():
        #     # if question == "*質問箇所TEL/FAX":
        #     # if question == normalize_output_text("仕様書交付期限"):
        #     #     print("====================================")
        #     #     for item in val:
        #     #         print("Ground-truth: {}, prediction: {}, qaid: {}".format(item[0][0], item[1], item[2]))
        #     if len(val) == 1:
        #         [ground_truths, prediction, _] = val[0]
        #         acc = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        #         if acc != 1.0:
        #             print("Question: {}, Prediction: {}, Ground_truth: {}".format(_, prediction, ground_truths[0]))
        #
        #         exact_match += acc
        #         f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        #     else:
        #         ground_truths = []
        #         predicts = []
        #         for item in val:
        #             ground_truths.append(item[0][0])
        #             predicts.append(item[1])
        #
        #         if len(intersect(ground_truths, predicts)) == len(ground_truths):
        #             exact_match += 1
        #             f1 += 1
    print(total)
    exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total

    for ques, acc_lst in map_result.items():
        acc = sum(acc_lst) / len(acc_lst)
        map_result[ques] = acc

    map_result = sorted(map_result.items(), key=lambda x: x[1])
    print(map_result)

    return {'exact_match': exact_match, 'f1': f1}


def predict(dataset, predictions):
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                n_prediction = predictions[qa['id']]
                n_prediction = [item for item in n_prediction if '。' not in item['text']]

                if qa['is_impossible']:
                    continue

                if qa['question'] in ['*質問箇所TEL/FAX', '*入札書送付先', '*資格申請送付先', '入札書締切日時',
                                      '質問票締切日時', '開札日時', '資格申請締切日時', '仕様書交付期限']:
                    n_prediction = pre_process(n_prediction)[:3]
                    n_prediction = sorted(n_prediction, key=lambda x: x['start_index'])

                    text_field = []
                    for i in range(len(n_prediction)):
                        text_field.append(n_prediction[i]['text'])

                    qa['answers'][0]['text'] = ';'.join(text_field)
                    qa['answers'][0]['answer_start'] = n_prediction[0]['start_index']
                else:
                    # update answer and index
                    qa['answers'][0]['text'] = n_prediction[0]['text']
                    qa['answers'][0]['answer_start'] = n_prediction[0]['start_index']
    return dataset


def intersect(a, b):
    return list(set(a) & set(b))


if __name__ == '__main__':
    expected_version = 1.0
    # parsing = argparse.ArgumentParser(
    #     description='Evaluation for SQuAD ' + str(expected_version))
    # parsing.add_argument('dataset_file', help='Dataset file')
    # parsing.add_argument('prediction_file', help='Prediction File')
    # args = parsing.parse_args()
    # with open(args.dataset_file) as dataset_file:
    #     dataset_json = json.load(dataset_file)
    #     if dataset_json['version'] != expected_version:
    #         print('Evaluation expects v-' + str(expected_version) +
    #               ', but got dataset with v-' + dataset_json['version'],
    #               file=sys.stderr)
    #     dataset = dataset_json['data']
    # with open(args.prediction_file) as prediction_file:
    #     predictions = json.load(prediction_file)
    # print(json.dumps(evaluate(dataset, predictions)))

    with open('../data_tools/new_dev_9.json') as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + str(expected_version) +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open('./test9/nbest_predictions.json') as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))

    # with open('../data_tools/new_dev_8_test.json') as dataset_file:
    #     dataset_json = json.load(dataset_file)
    #     if dataset_json['version'] != expected_version:
    #         print('Evaluation expects v-' + str(expected_version) +
    #               ', but got dataset with v-' + dataset_json['version'],
    #               file=sys.stderr)
    #     dataset = dataset_json['data']
    # with open('./test8/nbest_predictions.json_2_test') as prediction_file:
    #     predictions = json.load(pre  diction_file)
    # print(json.dumps(predict(dataset, predictions)))
