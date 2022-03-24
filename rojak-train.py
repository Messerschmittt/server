#!/usr/bin/env python

# combination of
# cross-encoder-training_nli.py


"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "contradiction": 0, "entailment": 1, "neutral": 2.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_nli.py
"""
from audioop import cross
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime
import os
import gzip
import csv
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


# #As dataset, we use SNLI + MultiNLI
# #Check if dataset exsist. If not, download and extract  it
# nli_dataset_path = 'datasets/AllNLI.tsv.gz'

# if not os.path.exists(nli_dataset_path):
#     util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)


# # Read the AllNLI.tsv.gz file and create the training dataset
# logger.info("Read AllNLI train dataset")

# label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
# train_samples = []
# dev_samples = []
# with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         label_id = label2int[row['label']]
#         if row['split'] == 'train':
#             train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
#         else:
#             dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

df_train_update = pd.read_csv("https://raw.githubusercontent.com/silentmusix/slavaukraini/main/train_update.csv")
df_train_update.head()


test_phase_1_update = pd.read_csv("https://raw.githubusercontent.com/silentmusix/slavaukraini/main/test_phase_1_update.csv")
test_phase_1_update.head()
test_phase_1_update_list = test_phase_1_update[['Claim', 'Evidence']].to_records(index=False)

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}

cross_validation_k = 10


# In https://huggingface.co/cross-encoder/nli-distilroberta-base
# 0 = contradiction
# 1 = entailment
# 2 = neutral



# In df:
# 0 means irrelevant to the claim (i.e., NOINFO)
# 1 means the evidence sentence supports the claim (i.e., SUPPORT)
# 2 means the evidence sentence refutes the claim (i.e., REFUTE). 

# Swap df so that
# 2 means irrelevant to the claim (i.e., NOINFO)
# 1 means the evidence sentence supports the claim (i.e., SUPPORT)
# 0 means the evidence sentence refutes the claim (i.e., REFUTE). 

print(df_train_update.head())
df_train_update.loc[df_train_update['Label'] == 0, 'Label'] = 999
df_train_update.loc[df_train_update['Label'] == 2, 'Label'] = 0
df_train_update.loc[df_train_update['Label'] == 999, 'Label'] = 2
print(df_train_update.head())


all_samples = []
for index, row in df_train_update.iterrows():
    all_samples.append(InputExample(texts=[row['Claim'], row['Evidence']], label=row['Label']))


num_samples = len(all_samples)
# num_samples = 1000
dev_samples_size = int(num_samples / cross_validation_k)
print(f"num_samples = {num_samples}, block size = {dev_samples_size}")

# for i in range(0, cross_validation_k):
#     dev_samples_start = i * dev_samples_size
#     dev_samples_end = dev_samples_start + dev_samples_size - 1
#     print(f"{start} to {end}")
# exit()

for i in range(0, cross_validation_k):

    # dev_samples = train_samples[i*cross_validation_k:cross_validation_k*num]

    dev_samples_start = i * dev_samples_size
    dev_samples_end = dev_samples_start + dev_samples_size - 1
    dev_samples = all_samples[dev_samples_start:dev_samples_end]
    print(f"dev_samples_start = {dev_samples_start}, dev_samples_end = {dev_samples_end}")

    # train_samples = all_samples[0:dev_samples_start] + all_samples[dev_samples_end:num_samples]
    train_samples = all_samples

    
    train_batch_size = 32
    num_epochs = 10

    # model_name = "gsarti/biobert-nli"
    # model_name = "distilroberta-base"
    # model_name = "cross-encoder/nli-distilroberta-base"
    # model_name = "cross-encoder/nli-deberta-v3-base"
    # model_name = "gsarti/scibert-nli"
    # model_name = "razent/SciFive-large-Pubmed_PMC"
    model_save_path = f"output/rojak-{model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


    # model = CrossEncoder('distilroberta-base', num_labels=3)
    model = CrossEncoder(model_name, num_labels=3)
    # model = CrossEncoder('gsarti/biobert-nli', num_labels=3)
    
    # model = CrossEncoder('cross-encoder/nli-distilroberta-base', num_labels=3)
    # model = CrossEncoder('cross-encoder/nli-deberta-v3-base', num_labels=3)
    

    # tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")
    # model = AutoModel.from_pretrained("gsarti/biobert-nli")

    #We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    # Our training loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    #During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples)


    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            use_amp=True
            )


# scores = model.predict(test_phase_1_update_list)
# print("=== scores ===")
# # print(scores)
# label_mapping = ['2', '1', '0']
# labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
# # print(labels)
# print(*labels,sep='\n')