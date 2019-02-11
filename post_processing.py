from __future__ import print_function
from collections import Counter
from parsing.address_parser import AddressParser
import json
import sys
import argparse
import re
import copy


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, matching_flag):
    """
    Metric for evaluation
    :param prediction: prediction text from model
    :param ground_truth: ground truth text for evaluation
    :param matching_flag: True: Only matching text, False: Both matching text and position
    :return: True/False
    """
    if matching_flag:
        return prediction['text'].lower() == ground_truth[0].lower()
    else:
        return prediction['text'].lower() == ground_truth[0].lower() and prediction['start_index'] == ground_truth[1]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, matching_flag):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, matching_flag)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def pre_process(a, question):
    """
    Ignore un-valid predictions from result of model
    :param a: model output
    :return: processed predictions
    """

    def validate_index(indexes, start_index, end_index, text):
        for [_start_idx, _end_idx, _text] in indexes:
            if _start_idx <= start_index <= _end_idx or _start_idx <= end_index <= _end_idx or _text in text:
                return False

        return True

    b = []
    indexes = []
    for i in range(0, len(a)):
        if a[i]['text'].startswith("日時平成"):
            a[i]['start_index'] += 2
            a[i]['text'] = a[i]['text'][2:]

        if re.search("か", a[i]['text']):
            if question == "開札日時":
                ch_idx = a[i]['text'].index("か")
                a[i]['text'] = a[i]['text'][:ch_idx]
                a[i]['end_index'] = a[i]['start_index'] + ch_idx

        if question == "施設名":
            a[i]['text'] = a[i]['text'].replace("で使用する電気", "")
            a[i]['text'] = a[i]['text'].replace("で使用する電", "")

        if i == 0:
            b.append(a[i])
            indexes.append([a[i]['start_index'], a[i]['end_index'], a[i]['text']])
        else:
            if validate_index(indexes, a[i]['start_index'], a[i]['end_index'], a[i]['text']):
                b.append(a[i])
                indexes.append([a[i]['start_index'], a[i]['end_index'], a[i]['text']])
            else:
                if question in ["開札日時", "*資格申請送付先部署/担当者名", "需要場所(住所)", "施設名", "入札件名"]:
                    tmp_idx = -1
                    cur_idx = -1
                    for [_, __, _text] in indexes:
                        tmp_idx += 1
                        if _text in a[i]['text']:
                            cur_idx = tmp_idx
                            break

                    if cur_idx >= 0:
                        # Update start index, end index and prediction text
                        b[cur_idx]['start_index'] = a[i]['start_index']
                        b[cur_idx]['end_index'] = a[i]['end_index']
                        b[cur_idx]['text'] = a[i]['text']

                        indexes[cur_idx][0] = a[i]['start_index']
                        indexes[cur_idx][1] = a[i]['end_index']
                        indexes[cur_idx][2] = a[i]['text']
    return b


def intersect(a, b):
    return list(set(a) & set(b))


def evaluate(dataset, predictions, matching_flag=True):
    """
    Evaluate model output
    :param dataset: contains ground truth text and its position
    :param predictions: model output of N-best results
    :param matching_flag: True/False
    :return: model accuracy
    """
    exact_match = total = 0
    ground_truth_map = {}
    prediction_map = {}

    map_result = {}
    for article in dataset:
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

                tmp = ground_truths[0]
                tmp.append(qa['id'])
                if qa['question'] not in ground_truth_map:
                    ground_truth_map[qa['question']] = [tmp]
                else:
                    ground_truth_map[qa['question']].append(tmp)

                prediction_map[qa['question']] = n_prediction

        for ques in ground_truth_map:
            n_ground_truths = ground_truth_map[ques]
            n_ground_truths = sorted(n_ground_truths, key=lambda x: x[1])
            n_prediction = prediction_map[ques]
            n_prediction = copy.deepcopy(pre_process(n_prediction, ques)[: len(n_ground_truths)])

            total += 1
            acc = 0.0

            flag = True
            is_all_in_one = True

            # Update value's tag5, ta6 by parsing data
            if ques in ['施設名', '需要場所(住所)']:
                for i in range(len(n_prediction)):
                    orig_text = n_prediction[i]['text']
                    addr = AddressParser(orig_text)
                    if ques == '施設名':
                        n_prediction[i]['text'] = addr.get_output_components()['other']
                    else:
                        n_prediction[i]['text'] = addr.get_output_components()['address']
                    n_prediction[i]['start_index'] += orig_text.index(n_prediction[i]['text'])
                    n_prediction[i]['text'] = n_prediction[i]['text'].replace('"', "")
                    n_prediction[i]['text'] = n_prediction[i]['text'].replace('\n', "")
                    n_prediction[i]['text'] = n_prediction[i]['text'].replace('\\n', "")
            elif ques in ["資格申請締切日時", "質問票締切日時"]:
                for i in range(len(n_prediction)):
                    if "\\n" in n_prediction[i]['text']:
                        n_prediction[i]['text'] = n_prediction[i]['text'][n_prediction[i]['text'].index("\\n") + 2:]
            elif ques in ["入札書締切日時"]:
                for i in range(len(n_prediction)):
                    n_prediction[i]['text'] = n_prediction[i]['text'].replace("\\n", "")

            for i in range(len(n_ground_truths)):
                if matching_flag:
                    if n_ground_truths[i][0] not in n_prediction[0]['text']:
                        is_all_in_one = False
                        break
                else:
                    if n_ground_truths[i][0] not in n_prediction[0]['text'] or \
                            n_ground_truths[0][1] != n_prediction[0]['start_index']:
                        is_all_in_one = False
                        break

            if not is_all_in_one:
                n_prediction = sorted(n_prediction, key=lambda x: x['start_index'])
                for i in range(len(n_ground_truths)):
                    if matching_flag:
                        if not (n_ground_truths[i][0] == n_prediction[i]['text']):
                            flag = False
                            break
                    else:
                        if not (n_ground_truths[i][0] == n_prediction[i]['text'] and n_ground_truths[i][1] ==
                                n_prediction[i]['start_index']):
                            flag = False
                            break
            if flag:
                exact_match += 1.0
                acc = 1.0
            else:
                if ques in ["資格申請締切日時", "質問票締切日時"]:
                    if len(n_ground_truths) == 1:
                        candidate = ""
                        start_idx = -1
                        for i in range(len(prediction_map[ques])):
                            if not re.search("^(午後)(\d{1,2})(時)((\d{1,2})(分|)|)", prediction_map[ques][i]['text']):
                                candidate = prediction_map[ques][i]['text']
                                start_idx = prediction_map[ques][i]['start_index']
                                break

                        if matching_flag:
                            if n_ground_truths[0][0] == candidate:
                                exact_match += 1.0
                                acc = 1.0
                        else:
                            if n_ground_truths[0][0] == candidate and n_ground_truths[0][1] == start_idx:
                                exact_match += 1.0
                                acc = 1.0

            if ques not in map_result:
                map_result[ques] = [acc]
            else:
                map_result[ques].append(acc)

        ground_truth_map.clear()
        prediction_map.clear()

    exact_match = 100.0 * exact_match / total

    for ques, acc_lst in map_result.items():
        acc = sum(acc_lst) / len(acc_lst)
        map_result[ques] = acc

    map_result = sorted(map_result.items(), key=lambda x: x[1])
    print(map_result)
    return {'acc': exact_match}


def predict(dataset, predictions):
    """
    This function serves for API
    :param dataset: contains ground truth text and its position
    :param predictions: model output of N-best results
    :return: API output
    """
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
                n_prediction = pre_process(n_prediction, qa['question'])

                if qa['question'] in ['資格申請締切日時', '*質問箇所TEL/FAX']:
                    n_prediction = n_prediction[:3]
                elif qa['question'] in ['仕様書交付期限', '質問票締切日時', '入札書締切日時', '開札日時', '*資格申請送付先',
                                        '*資格申請送付先部署/担当者名', '*入札書送付先', '*入札書送付先部署/担当者名',
                                        '*開札場所']:
                    if qa['question'] in ['*資格申請送付先', '*入札書送付先']:
                        addr = AddressParser(n_prediction[0]['text'])
                        if len(addr.get_output_components()['other']) > 0 and len(
                                addr.get_output_components()['address']) > 0:
                            n_prediction = n_prediction[:1]
                        else:
                            n_prediction = n_prediction[:2]
                    else:
                        n_prediction = n_prediction[:2]
                else:
                    n_prediction = n_prediction[:1]

                # Update value's tag5, ta6 by parsing data
                if qa['question'] in ['施設名', '需要場所(住所)']:
                    for i in range(len(n_prediction)):
                        orig_text = n_prediction[i]['text']
                        addr = AddressParser(orig_text)
                        if qa['question'] == '施設名':
                            n_prediction[i]['text'] = addr.get_output_components()['other']
                        else:
                            n_prediction[i]['text'] = addr.get_output_components()['address']
                        n_prediction[i]['start_index'] += orig_text.index(n_prediction[i]['text'])
                        n_prediction[i]['text'] = n_prediction[i]['text'].replace('"', "")
                        n_prediction[i]['text'] = n_prediction[i]['text'].replace('\n', "")
                        n_prediction[i]['text'] = n_prediction[i]['text'].replace('\\n', "")
                elif qa['question'] in ["資格申請締切日時", "質問票締切日時"]:
                    for i in range(len(n_prediction)):
                        if "\\n" in n_prediction[i]['text']:
                            n_prediction[i]['text'] = n_prediction[i]['text'][n_prediction[i]['text'].index("\\n") + 2:]
                elif qa['question'] in ["入札書締切日時"]:
                    for i in range(len(n_prediction)):
                        n_prediction[i]['text'] = n_prediction[i]['text'].replace("\\n", "")

                tmp_n_prediction = []
                start_idx = -1
                end_idx = -1
                for i in range(len(n_prediction)):
                    if i == 0:
                        tmp_n_prediction.append(n_prediction[i])
                    else:
                        if 0 <= n_prediction[i]['start_index'] - end_idx <= 2 \
                                or 0 <= start_idx - n_prediction[i]['end_index'] <= 2:
                            tmp_n_prediction.append(n_prediction[i])
                            tmp_n_prediction = sorted(tmp_n_prediction, key=lambda x: x['start_index'])
                        else:
                            break
                    start_idx = tmp_n_prediction[0]['start_index']
                    end_idx = tmp_n_prediction[-1]['end_index']

                text_field = []
                for i in range(len(tmp_n_prediction)):
                    text_field.append(tmp_n_prediction[i]['text'])

                if qa['question'] == "*質問箇所TEL/FAX":
                    qa['answers'][0]['text'] = ';'.join(text_field)
                else:
                    qa['answers'][0]['text'] = ''.join(text_field)
                qa['answers'][0]['answer_start'] = tmp_n_prediction[0]['start_index']

    return dataset


def test_evaluate():
    expected_version = 1.0
    with open('bert/contract/dev.json') as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + str(expected_version) +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        _dataset = dataset_json['data']
    with open('bert/nbest_predictions.json') as prediction_file:
        _predictions = json.load(prediction_file)
    print(json.dumps(evaluate(_dataset, _predictions, True)))


def test_prediction():
    expected_version = 1.0
    with open('bert/contract/dev.json') as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + str(expected_version) +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open('bert/contract_output/nbest_predictions.json') as prediction_file:
        predictions = json.load(prediction_file)
    with open('predicted_file.json', 'w') as f:
        json.dump(predict(dataset, predictions), f)


if __name__ == '__main__':
    expected_version = 1.0
    parsing = argparse.ArgumentParser(description='Evaluation for Elements Contract Extraction ' + str(expected_version))
    parsing.add_argument('dataset_file', help='Dataset file')
    parsing.add_argument('prediction_file', help='Prediction file')
    args = parsing.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + str(expected_version) +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        _dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        _predictions = json.load(prediction_file)
    print(json.dumps(evaluate(_dataset, _predictions)))
