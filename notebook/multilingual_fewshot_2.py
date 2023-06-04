from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



import json
from transformers import AutoTokenizer, AdapterConfig, AutoAdapterModel, AutoConfig
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, TrainerCallback
from transformers import AutoModelForSequenceClassification, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict

from tqdm import tqdm

import numpy as np
from datasets import concatenate_datasets, load_metric
import evaluate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)




df_en = pd.read_csv("../data/subtask_2/en/train.tsv", sep='\t')
df_en=df_en.drop(df_en.columns[0], axis=1)

df_es = pd.read_csv("../data/subtask_2/es/train.tsv", sep='\t')
df_es=df_es.drop(df_es.columns[0], axis=1)


mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5
}
df_en["label"] = df_en['label'].map(mapping)
df_es["label"] = df_es['label'].map(mapping)


from sklearn.model_selection import train_test_split
from datasets import Dataset, concatenate_datasets

# Split your data into train and test sets
dataset_train_en, dataset_test_en = train_test_split(df_en, test_size=0.1, random_state=42)
dataset_train_es, dataset_test_es = train_test_split(df_es, test_size=0.1, random_state=42)

# Further split your train data into train and validation sets
dataset_train_en, dataset_valid_en = train_test_split(dataset_train_en, test_size=0.1, random_state=42)
dataset_train_es, dataset_valid_es = train_test_split(dataset_train_es, test_size=0.1, random_state=42)


shots = [500, 1000, 1500, 2000, 2500]

list_train_datasets = {}

for shot in shots:
    # Concatenate the datasets
    dataset_train_compl = pd.concat([dataset_train_en.iloc[:(shot//2)], dataset_train_es.iloc[:(shot//2)]])
    
    list_train_datasets[str(shot)] = dataset_train_compl

dataset_valid_compl = pd.concat([dataset_valid_en, dataset_valid_es])
dataset_test_compl = pd.concat([dataset_test_en, dataset_test_es])


language_model = "xlm-roberta-base"
# language_model = "bert-base-multilingual-cased"
# language_model = "microsoft/deberta-v3-base"
# language_model = "prajjwal1/bert-tiny"
# language_model = "distilbert-base-cased"
# language_model = "roberta-base-openai-detector"
# language_model = "Hello-SimpleAI/chatgpt-detector-roberta"




dataset_test_en = Dataset.from_pandas(dataset_test_en)
dataset_test_es = Dataset.from_pandas(dataset_test_es)


tokenizer = AutoTokenizer.from_pretrained(language_model)


def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

dataset_test_en = dataset_test_en.map(encode_batch, batched=True)
dataset_test_en = dataset_test_en.rename_column("label", "labels")
dataset_test_en.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])


dataset_test_es = dataset_test_es.map(encode_batch, batched=True)
dataset_test_es = dataset_test_es.rename_column("label", "labels")
dataset_test_es.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])



t_metrics_en = {}
t_metrics_es = {}


for shot in shots:

    dataset_train = Dataset.from_pandas(list_train_datasets[str(shot)])
    dataset_valid = Dataset.from_pandas(dataset_valid_compl)



    dataset_train = dataset_train.rename_column("label", "labels")
    dataset_train = dataset_train.map(encode_batch, batched=True)
    dataset_train.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    dataset_valid = dataset_valid.rename_column("label", "labels")
    dataset_valid = dataset_valid.map(encode_batch, batched=True)
    dataset_valid.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])



    model = AutoModelForSequenceClassification.from_pretrained(language_model, num_labels=len(dataset_train_compl.label.unique()), ignore_mismatched_sizes=True)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    early_stop = EarlyStoppingCallback(3)

    training_args = TrainingArguments(
        learning_rate=1e-5,
        num_train_epochs=10,
        seed = 42,
        output_dir="./training_output_multilingual1",
        # label_names=["generated", "human"]
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        dataloader_num_workers=32,
        logging_steps=100,
        save_total_limit = 2,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to='tensorboard',
        metric_for_best_model='f1'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        compute_metrics=compute_metrics,
        callbacks = [early_stop]
    )

    trainer.train()



    t_metrics_en[str(shot)] = trainer.evaluate(dataset_test_en)
    t_metrics_es[str(shot)] = trainer.evaluate(dataset_test_es)

    print(t_metrics_en)
    print(t_metrics_es)
pd.DataFrame([t_metrics]).to_csv("./result/fewshot_subtask1.csv")
