from run_contract_qa import read_contract_examples
from run_contract_qa import convert_examples_to_features
from run_contract_qa import create_model
from run_contract_qa import RawResult
from run_contract_qa import write_predictions
from bert import modeling
from bert import tokenization
import os
import tensorflow as tf
import post_processing
import json
import configparser
import tensorflow.contrib.tensorrt as trt


class BertQA:

    def __init__(self, config_path=None):
        super().__init__()

        config = configparser.ConfigParser()
        config.read(config_path)

        if config is None:
            raise ValueError("Config is None.")

        default = config['DEFAULT']
        obligatory = config['OBLIGATORY']

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        gpu_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=gpu_config)

        ckpt_path = obligatory['checkpoint']

        if ckpt_path is None or len(ckpt_path) == 0:
            raise ValueError("Checkpoint path is None")

        try:
            self.max_seq_length = int(default['max_seq_length'])
            self.doc_stride = int(default['doc_stride'])
            self.max_query_length = int(default['max_query_length'])
            self.max_answer_length = int(default['max_answer_length'])
            self.n_best_size = int(default['n_best_size'])
            self.input_ids_p = tf.placeholder(tf.int32, [None, self.max_seq_length], name="input_ids")
            self.input_mask_p = tf.placeholder(tf.int32, [None, self.max_seq_length], name="input_mask")
            self.unique_ids_p = tf.placeholder(tf.int32, [None, ], name="unique_ids")
            self.segment_ids_p = tf.placeholder(tf.int32, [None, self.max_seq_length], name="segment_ids")
            self.sess = sess
            bert_config = modeling.BertConfig.from_json_file(default['bert_config'])
            self.start_logits, self.end_logits = create_model(bert_config, False, self.input_ids_p, self.input_mask_p,
                                                              self.segment_ids_p, False)
            self.graph = self.load(ckpt_path)
            self.tokenizer = tokenization.FullTokenizer(default['bert_vocab'])
            self.batch_size = int(default['batch_size'])
            os.environ['CUDA_VISIBLE_DEVICES'] = default['gpu_device']

        except ValueError:
            raise ValueError("Wrong config")

    def load(self, checkpoint):
        # saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(checkpoint)
        saver.restore(self.sess, os.path.join(os.path.dirname(checkpoint), os.path.basename(checkpoint)[:-5]))
        graph = tf.get_default_graph()
        return graph

    @staticmethod
    def get_iterator(data):
        """Wrap numpy data in a dataset."""
        dataset = tf.data.Dataset.from_tensors(data).repeat()
        return dataset.make_one_shot_iterator()

    def process(self, predict_file):
        import time
        start_time = time.time()
        eval_examples = read_contract_examples(input_file=predict_file, is_training=False)
        eval_features = []

        def append_feature(_feature):
            eval_features.append(_feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            output_fn=append_feature)

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))

        predictions = {
            "unique_ids": self.unique_ids_p,
            "start_logits": self.start_logits,
            "end_logits": self.end_logits,
        }

        all_results = []

        with self.graph.as_default():
            count = 0
            _input_ids_batch, _input_mask_batch, _segment_ids_batch, _unique_ids_batch = [], [], [], []
            no_batchs = len(eval_features) // self.batch_size
            count_batch = 0

            output_node_names = ["unique_ids", "start_logits", "end_logits"]
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                self.sess,
                self.graph.as_graph_def(),
                output_node_names=["cls/squad/output_bias/adam_v", "cls/squad/output_weights/adam_v"])

            # print(frozen_graph)
            trt_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph,
                outputs=output_node_names,
                max_batch_size=self.batch_size,
                max_workspace_size_bytes=2 << 10,
                precision_mode="FP16")

            _input_ids = []
            _input_mask = []
            _segment_ids = []
            _unique_ids = []
            for feature in eval_features:
                _input_ids.append(feature.input_ids)
                _input_mask.append(feature.input_mask)
                _segment_ids.append(feature.segment_ids)
                _unique_ids.append(feature.unique_id)

            output_node = tf.import_graph_def(
                graph_def=trt_graph,
                input_map={"input_ids": self.get_iterator(_input_ids).get_next(),
                           "input_mask": self.get_iterator(_input_mask).get_next(),
                           "unique_ids": self.get_iterator(_unique_ids).get_next(),
                           "segment_ids": self.get_iterator(_segment_ids).get_next()},
                return_elements=output_node_names
            )

            result = self.sess.run(output_node)

            for i in range(len(result["unique_ids"])):
                unique_id = result["unique_ids"][i]
                start_logits = [float(x) for x in result["start_logits"][i].flat]
                end_logits = [float(x) for x in result["end_logits"][i].flat]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))

            # for feature in eval_features:
            #     _input_ids_batch.append(feature.input_ids)
            #     _input_mask_batch.append(feature.input_mask)
            #     _segment_ids_batch.append(feature.segment_ids)
            #     _unique_ids_batch.append(feature.unique_id)
            #     count += 1
            #     if count % self.batch_size == 0 or (count_batch == no_batchs and count == len(eval_features)):
            #         count_batch += 1
            #         feed_dict = {self.input_ids_p: _input_ids_batch,
            #                      self.input_mask_p: _input_mask_batch,
            #                      self.segment_ids_p: _segment_ids_batch,
            #                      self.unique_ids_p: _unique_ids_batch}
            #
            #         # run session get current feed_dict result
            #         result = self.sess.run(predictions, feed_dict)
            #
            #         for i in range(len(result["unique_ids"])):
            #             unique_id = result["unique_ids"][i]
            #             start_logits = [float(x) for x in result["start_logits"][i].flat]
            #             end_logits = [float(x) for x in result["end_logits"][i].flat]
            #             all_results.append(
            #                 RawResult(
            #                     unique_id=unique_id,
            #                     start_logits=start_logits,
            #                     end_logits=end_logits))
            #
            #         # clear after batch processing
            #         _input_ids_batch.clear()
            #         _input_mask_batch.clear()
            #         _segment_ids_batch.clear()
            #         _unique_ids_batch.clear()

        with tf.gfile.Open(predict_file, "r") as reader:
            input_data = json.load(reader)["data"]

        predictions = write_predictions(eval_examples, eval_features, all_results, self.n_best_size,
                                        self.max_answer_length, True, None, None, None, False)

        result = post_processing.predict(input_data, predictions)
        print('Time to process: {}'.format(time.time() - start_time))
        return result


def main():
    # Load config
    bert_model = BertQA('config.ini')
    res = bert_model.process('uploads/new_dev_8_test.json')
    print(res)


if __name__ == "__main__":
    main()
