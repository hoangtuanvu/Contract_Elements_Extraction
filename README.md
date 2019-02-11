# BERT-Question_Answering
Tensorflow solution of Question Answering task Using BERT model with Google BERT Embeddings

The Elements Contract Extraction training data(bert/contract) 

Try to implement Question Answering work based on google's BERT code!

## How to train
#### 1. Download BERT for multilingual model (include Japanese):
https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip

#### 2. Extract the above zip file and put it into the following path
```
bert
```

#### 3. Create dataset for training and evaluating model
You can change the paths contain train and test set
```
  cd data_tools
  python create_dataset.py '../bert/contract/train.json' '../bert/contract/dev.json'
```

#### 4. Train model
You can change the "train_file" path and "predict_file" path by your own paths (Default: train_file=bert/contract/train.json, predict_file=bert/contract/dev.json)
```
  python run_contract_qa.py   \   
                  --vocab_file=bert/bert_based/vocab.txt   \
                  --bert_config_file=bert/bert_based/bert_config.json   \
                  --init_checkpoint=bert/bert_based/bert_model.ckpt   \
                  --do_train=True   \
                  --train_file=[path_to_train_file]   \
                  --do_predict=True   \
                  --predict_file=[path_to_test_file]   \
                  --train_batch_size=8  \
                  --predict_batch_size=8   \
                  --learning_rate=3e-5   \
                  --num_train_epochs=100.0   \
                  --max_seq_length=384   \
                  --doc_stride=128   \
                  --output_dir=bert/contract_output    \
                  --max_query_length=15     \
                  --max_answer_length=70    \
                  --version_2_with_negative=False   \
                  --n_best_size=10
 ```       

## How to evaluate 
```
  python post_processing.py [path_to_test_file] [path_to_n_best_output_json]
```

## How to run API service
If you train your own model, ignore step1, step2

#### 1. Download trained model from the following link:

#### 2. Extract the above zip file and put it into the root path

#### 3. Run API
All model's paramters are in config path (default: config_file_path = config.ini), you can re-configure by yourself.
```

  python run_contract_service.py [config_file_path] [port]
```

You have to check the configured port which does exist or not by the following command:
```bash

  netstat -anp|grep [port]
```


## reference:
+ [https://github.com/google-research/bert](https://github.com/google-research/bert)

Note: Currently, this model is run on 1 GPU GTX and can process one document per minute. In the future work, we can speed up the process by experimenting model with less layers and run on multi-gpu.