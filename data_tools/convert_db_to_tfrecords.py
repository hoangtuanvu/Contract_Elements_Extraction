from bert.run_squad import FeatureWriter
from bert.run_squad import convert_examples_to_features
from bert.run_squad import read_squad_examples
from bert import tokenization
import os


def create_tf_records(input_file, output_dir):
    examples = read_squad_examples(
        input_file=input_file, is_training=True)

    tokenizer = tokenization.FullTokenizer(
        vocab_file='../bert/bert_based/vocab.txt', do_lower_case=True)

    writer = FeatureWriter(
        filename=os.path.join(output_dir, "train.tf_record"),
        is_training=True)
    convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=20,
        is_training=True,
        output_fn=writer.process_feature)
    writer.close()


if __name__ == '__main__':
    create_tf_records('train.json', '.')
