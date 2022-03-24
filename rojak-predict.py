#!/usr/bin/env python

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
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
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

# model_name="output/rojak-gsarti/scibert-nli-2022-03-19_20-11-37"
model_name = "output/rojak2-gsarti/biobert-nli-2022-03-24_14-34-32"

# model = CrossEncoder('output/training_allnli-2022-03-19_15-20-38', num_labels=3)
# model = CrossEncoder('output/training_allnli-2022-03-19_16-19-54', num_labels=3)
# model = CrossEncoder('output/training_allnli-2022-03-19_17-07-21', num_labels=3)
model = CrossEncoder(model_name, num_labels=3)



test_phase_1_update = pd.read_csv("https://raw.githubusercontent.com/silentmusix/slavaukraini/main/test_phase_1_update.csv")
test_phase_1_update.head()
test_phase_1_update_list = test_phase_1_update[['Claim', 'Evidence']].to_records(index=False)

scores = model.predict(test_phase_1_update_list)
labels = [score_max for score_max in scores.argmax(axis=1)]

# In https://huggingface.co/cross-encoder/nli-distilroberta-base
# 0 = contradiction
# 1 = entailment
# 2 = neutral

# In codalab
# 0 means irrelevant to the claim (i.e., NOINFO)
# 1 means the evidence sentence supports the claim (i.e., SUPPORT)
# 2 means the evidence sentence refutes the claim (i.e., REFUTE). 
label_mapping = ['2', '1', '0']

labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

print(*labels,sep='\n')

with open('p110.txt', 'w') as f:
    print(*labels, sep='\n', file=f)
    f.close()
