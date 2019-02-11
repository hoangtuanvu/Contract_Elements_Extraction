import pandas as pd
import unicodedata
import jaconv
import glob
import os
import uuid
import json
import copy
import argparse
import errno
import sys

sys.path.append('../')
from data_tools.constants import MAP_QUESTIONS
from data_tools.constants import MAP_QUESTIONS, MAP_QUESTIONS_2, SUB_TAG_15, SUB_TAG_16, SUB_TAG_17, SUB_TAG_18, \
    SUB_TAG_19, SUB_TAG_22, \
    SUB_TAG_24, SUB_TAG_26, MAP


def _read_excel(data_path):
    """
    Read data from excel files and create dictionary format file to save this information
    :param data_path: path contains excel files
    :return: dictionary with reusable format
    """
    df = pd.read_excel(data_path)

    paragraph = {'context': '', 'qas': []}
    qas = []
    context = []

    for i in df.index:
        raw_sent = df['Sentence'][i]
        normalized_sent = _normalize_text(raw_sent)

        raw_tag = df['Tag'][i]
        normalized_tag = _normalize_text(raw_tag)

        raw_value = df['Value'][i]
        normalized_value = _normalize_text(raw_value)

        if len(normalized_sent) > 0:
            if not normalized_sent.endswith('。'):
                normalized_sent = ''.join([normalized_sent, '。'])

        if len(normalized_tag) > 0 and len(normalized_value) == 0:
            raise ValueError('No answer for question {}'.format(normalized_tag))

        if len(normalized_tag) == 0 and len(normalized_value) > 0:
            raise ValueError('No question for value {}'.format(normalized_value))

        if len(normalized_tag) > 0:
            questions = normalized_tag.split(';')
            answers = normalized_value.split(';')

            if len(questions) == len(answers):
                map_qas = [[questions[i], answers[i]] for i in range(len(questions))]
            elif len(questions) > len(answers):
                if len(answers) != 1:
                    raise ValueError('Only one answer {} in multiple questions'.format(normalized_value))
                map_qas = [[questions[i], answers[0]] for i in range(len(questions))]
            else:
                if len(questions) != 1:
                    raise ValueError('Only one question {} in multiple answers'.format(normalized_tag))
                map_qas = [[questions[0], answers[i]] for i in range(len(answers))]

            for ques, ans in map_qas:
                if ans not in normalized_sent:
                    print('Answer: {} not in the sentence {} of {}'.format(ans, normalized_sent, data_path))
                    continue

                if ques in ['*入札書送付先']:
                    if i >= 1:
                        # print(data_path, normalize_output_text(df['Sentence'][i - 1]) + '。' + normalized_sent, ans)
                        print(_normalize_text(df['Sentence'][i - 1]) + '。' + normalized_sent, ans)
                    else:
                        # print(ques, normalized_sent, ans, data_path)
                        print(normalized_sent, ans, data_path)

                start_pos = normalized_sent.index(ans)

                if len(context) > 0:
                    start_pos += context[-1][1]

                sub_qas = {'answers': [{'answer_start': start_pos, 'text': ans}], 'question': ques,
                           'id': str(uuid.uuid4()), 'is_impossible': False}
                qas.append(sub_qas)

        if len(context) > 0:
            context.append([normalized_sent, context[-1][1] + len(normalized_sent)])
        else:
            context.append([normalized_sent, len(normalized_sent)])

    paragraph['context'] = ''.join([txt[0] for txt in context])
    paragraph['qas'] = qas
    return paragraph


def _normalize_text(text):
    """Remove unusual characters from the ocr results
   :param text: ocr result need to be fix
   :returns: normalized string
   :rtype: string
   """

    if pd.isna(text):
        return ""

    text = unicodedata.normalize('NFKC', str(text))
    text = jaconv.normalize(text, 'NFKC')
    text = "".join(text.split())
    return text


def _create_dataset(input, samples=None, output_file=None):
    doc_all = {'data': [], 'version': 1.0}

    for folder_name in glob.glob(input):
        seg = {'title': folder_name, 'paragraphs': []}
        if os.path.basename(folder_name) not in samples:
            continue

        questions = []
        merged_qas = []
        merged_context = []
        combine = []
        for f_name in os.listdir(folder_name):
            if f_name.endswith('xlsx'):
                paragraph = _read_excel(os.path.join(folder_name, f_name))
                merged_context.append(paragraph['context'])
                qas = paragraph['qas']
                combine.append(copy.deepcopy(paragraph))

                count = -1
                for sub_qas in qas:
                    count += 1
                    ques_txt = sub_qas['question']
                    # ignore tag4 and tag23
                    if ques_txt in ['実施機関', '*質問書提出方法']:
                        del qas[count]
                        continue

                    if ques_txt in SUB_TAG_15:
                        sub_qas['question'] = '仕様書交付期限'
                    elif ques_txt in SUB_TAG_16:
                        sub_qas['question'] = '質問票締切日時'
                    elif ques_txt in SUB_TAG_17:
                        sub_qas['question'] = '資格申請締切日時'
                    elif ques_txt in SUB_TAG_18:
                        sub_qas['question'] = '入札書締切日時'
                    elif ques_txt in SUB_TAG_19:
                        sub_qas['question'] = '開札日時'
                    elif ques_txt in SUB_TAG_22:
                        sub_qas['question'] = '*質問箇所TEL/FAX'
                    elif ques_txt in SUB_TAG_24:
                        sub_qas['question'] = '*資格申請送付先'
                    elif ques_txt in SUB_TAG_26:
                        sub_qas['question'] = '*入札書送付先'

                    if sub_qas['question'] in ['施設名', '仕様書交付期限', '質問票締切日時', '需要場所(住所)', '電力量kWh(調達期間通し)']:
                        sub_qas['question'] = MAP[sub_qas['question']]

                    tmp = copy.deepcopy(sub_qas)
                    tmp['id'] = str(uuid.uuid4())
                    if len(merged_context) > 1:
                        tmp['answers'][0]['answer_start'] += len(''.join(merged_context[:-1]))
                    merged_qas.append(tmp)
                    questions.append(tmp['question'])

                seg['paragraphs'].append(paragraph)

        # for tmp in questions:
        #     if tmp not in list(MAP_QUESTIONS.values()):
        #         print("Question {} is unvalid".format(tmp))

        merged_paragraph = {'context': ''.join(merged_context), 'qas': merged_qas}
        seg['paragraphs'].append(merged_paragraph)
        doc_all['data'].append(seg)

    # json.dump(doc_all, open(output_file, 'w'), ensure_ascii=False, indent=2)


def _create_dataset_2(input, samples=None, output_file=None):
    doc_all = {'data': [], 'version': 1.0}

    for folder_name in glob.glob(input):
        seg = {'title': folder_name, 'paragraphs': []}
        if os.path.basename(folder_name) not in samples:
            continue

        merged_context = []
        merged_qas = []
        questions = []
        for f_name in os.listdir(folder_name):

            if f_name.endswith('xlsx'):
                paragraph = _read_excel(os.path.join(folder_name, f_name))
                merged_context.append(paragraph['context'])
                qas = paragraph['qas']

                for sub_qas in qas:
                    ques_txt = sub_qas['question']
                    # ignore tag4 and tag23
                    if ques_txt in ['実施機関', '*質問書提出方法']:
                        continue

                    if ques_txt in SUB_TAG_15:
                        sub_qas['question'] = '仕様書交付期限'
                    elif ques_txt in SUB_TAG_16:
                        sub_qas['question'] = '質問票締切日時'
                    elif ques_txt in SUB_TAG_17:
                        sub_qas['question'] = '資格申請締切日時'
                    elif ques_txt in SUB_TAG_18:
                        sub_qas['question'] = '入札書締切日時'
                    elif ques_txt in SUB_TAG_19:
                        sub_qas['question'] = '開札日時'
                    elif ques_txt in SUB_TAG_22:
                        sub_qas['question'] = '*質問箇所TEL/FAX'
                    elif ques_txt in SUB_TAG_24:
                        sub_qas['question'] = '*資格申請送付先'
                    elif ques_txt in SUB_TAG_26:
                        sub_qas['question'] = '*入札書送付先'

                    if sub_qas['question'] in ['施設名', '仕様書交付期限', '質問票締切日時', '需要場所(住所)', '電力量kWh(調達期間通し)']:
                        sub_qas['question'] = MAP[sub_qas['question']]

                    if len(merged_context) > 1:
                        sub_qas['answers'][0]['answer_start'] += len(''.join(merged_context[:-1]))
                    merged_qas.append(sub_qas)
                    questions.append(sub_qas['question'])

        # print("Folder {} has {} questions".format(folder_name, len(questions)))

        # Update Questions for Impossible cases
        # for tmp in questions:
        #     if tmp not in list(MAP_QUESTIONS.values()):
        #         print("Question {} is unvalid".format(tmp))

        for ques in list(MAP_QUESTIONS_2.values()):
            if ques not in questions:
                sub_qas = {'answers': [{'answer_start': -1, 'text': ''}], 'question': ques,
                           'id': str(uuid.uuid4()), 'is_impossible': True}
                merged_qas.append(sub_qas)

        merged_paragraph = {'context': ''.join(merged_context), 'qas': merged_qas}
        seg['paragraphs'].append(merged_paragraph)
        doc_all['data'].append(seg)

    # json.dump(doc_all, open(output_file, 'w'), ensure_ascii=False, indent=2)


def _create_train_set(folder, dev):
    """
    Create training set
    :param folder: All folder names of dataset
    :param dev: folder names of dev set
    :return: Folder names of training set
    """
    filenames = []
    for fname in os.listdir(folder):
        if fname in ['九州0686', '九州0003', '九州0189', '九州0217', '九州0183', '中部0310']:
            continue
        filenames.append(fname)

    train = []
    for fname in filenames:
        if fname not in dev:
            train.append(fname)
    return train


def check_n_make_folder(file_path):
    """
    Make path if does not exists
    :param file_path: Path contains file
    :return: `
    """
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    # create train and dev set
    dev = ['中部0124', '中部0486', '中部0465', '中部0093', '九州0028', '中国0051', '中国0054', '中国0152', '九州0376',
           '九州0347', '中部0186', '中部0033', '九州0553', '中国0040', '九州0377', '中国0042', '中部0484', '九州0041',
           '中国0050', '中部0371', '中国0041']
    train = _create_train_set('../data/kepco', dev)

    parsing = argparse.ArgumentParser(description='Create dataset for training BERT_QA model')
    parsing.add_argument('train_file', help='Directory contains training file in json format')
    parsing.add_argument('dev_file', help='Directory contains testing file in json format')

    args = parsing.parse_args()

    check_n_make_folder(args.train_file)
    check_n_make_folder(args.dev_file)

    _create_dataset('../data/kepco/*', train, args.train_file)
    print("========================================================================================")
    _create_dataset_2('../data/kepco/*', dev, args.dev_file)
