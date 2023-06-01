from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)




parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--subtask', type=str, required=True)
parser.add_argument('--lang', type=str, required=True)
args = vars(parser.parse_args())


language_model = args["model"]
subtask = args["subtask"]
lang = args["lang"]


df_subtask_1_en = pd.read_csv("../data/subtask_1/en/train.tsv", sep='\t')
df_subtask_1_en=df_subtask_1_en.drop(df_subtask_1_en.columns[0], axis=1)

mapping = {
    "generated":0,
    "human":1
}
df_subtask_1_en["label"] = df_subtask_1_en['label'].map(mapping)


dataset_train, dataset_test = train_test_split(df_subtask_1_en, test_size=0.1, random_state=42)

# language_model = "xlm-roberta-base"
# language_model = "bert-base-multilingual-cased"
# language_model = "microsoft/deberta-v3-base"
# language_model = "distilbert-base-cased"
# language_model = "prajjwal1/bert-tiny"
# language_model = "roberta-base-openai-detector"
# language_model = "Hello-SimpleAI/chatgpt-detector-roberta"


tokenizer = AutoTokenizer.from_pretrained(language_model)

dataset_train = Dataset.from_pandas(dataset_train)
dataset_test = Dataset.from_pandas(dataset_test)

def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")


dataset_train = dataset_train.rename_column("label", "labels")
dataset_train = dataset_train.map(encode_batch, batched=True)
dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

dataset_test = dataset_test.map(encode_batch, batched=True)
dataset_test = dataset_test.rename_column("label", "labels")
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


model = AutoModelForSequenceClassification.from_pretrained(language_model, num_labels=len(df_subtask_1_en.label.unique()), ignore_mismatched_sizes=True)
  
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
    learning_rate=1e-6,
    num_train_epochs=10,
    seed = 42,
    output_dir="./training_output",
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
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
    callbacks = [early_stop]
)

trainer.train()

t_metrics = trainer.evaluate(dataset_test)
print(pd.DataFrame([t_metrics]))