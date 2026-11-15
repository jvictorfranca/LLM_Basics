# Demo: Fine-Tuning with LoRA on a Custom Dataset

# Scenario:
# You are a Natural Language Processing Engineer at an e-commerce company
# building a sentiment analysis system to automatically classify customer
# product reviews as positive, neutral, or negative.
#
# Your company has collected 800 custom reviews specific to your product
# domain, but no pre-trained model exists for your exact use case.
#
# You need to adapt a pre-trained DistilBERT model to your specific review
# distribution and writing style.

# Learning Objectives:
#
# - Understand How to Load and Prepare Custom Data for Fine-Tuning
# - Master the Full Fine-Tuning Workflow with Hugging Face Trainer
# - Implement LoRA for Parameter-Efficient Fine-Tuning
# - Compare Full Fine-Tuning vs. LoRA on Real Custom Dataset



import time
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

CSV_PATH = 'reviews.csv'
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f'{CSV_PATH} not found. Please place reviews.csv in this directory.')

df = pd.read_csv(CSV_PATH)
assert 'review_text' in df.columns and 'sentiment' in df.columns, \
    'CSV must contain review_text and sentiment columns.'

df = df.dropna(subset=['review_text', 'sentiment']).reset_index(drop=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

split_idx = int(0.8 * len(df))
df_train = df.iloc[:split_idx].reset_index(drop=True)
df_val = df.iloc[split_idx:].reset_index(drop=True)

print(f'Total samples: {len(df)}, train: {len(df_train)}, val: {len(df_val)}')

train_ds = Dataset.from_pandas(df_train[['review_text', 'sentiment']])
val_ds = Dataset.from_pandas(df_val[['review_text', 'sentiment']])

train_ds = train_ds.rename_columns({'review_text': 'text', 'sentiment': 'label'})
val_ds = val_ds.rename_columns({'review_text': 'text', 'sentiment': 'label'})


#Tokenization

import pandas as pd
import numpy as np

# Create a dummy dataset
dummy_data = {
    'review_text': [
        'This product is amazing! I love it.',
        'It was okay, not great, not terrible.',
        'Absolutely terrible, a waste of money.',
        'Good quality for the price, very satisfied.',
        'Disappointing experience, will not buy again.',
        'Highly recommend this item!',
        'Mediocre at best.',
        'The best purchase I made this year!',
        'Could be better, average product.',
        'Terrible customer service, but the product is fine.'
    ],
    'sentiment': [
        'positive',
        'neutral',
        'negative',
        'positive',
        'negative',
        'positive',
        'neutral',
        'positive',
        'neutral',
        'negative'
    ]
}

df_dummy = pd.DataFrame(dummy_data)

# Save the dummy dataset to reviews.csv
df_dummy.to_csv('reviews.csv', index=False)

print('Dummy reviews.csv created successfully with the following content:')
display(df_dummy)

checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Create label mapping based on unique sentiments in the DataFrame
# The dataframe `df` is available from cell `2e33654`
unique_labels = sorted(df['sentiment'].unique().tolist())  # Ensure consistent order for label IDs
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for i, label in enumerate(unique_labels)}
num_labels = len(unique_labels)

print(f"Label mapping: {label_to_id}")
print(f"Number of labels: {num_labels}")


def preprocess(batch):
    tokenized_batch = tokenizer(
        batch['text'],
        truncation=True,
        padding=False,
        max_length=128
    )

    # Map string labels to integer IDs
    tokenized_batch['label'] = [
        label_to_id[label] for label in batch['label']
    ]

    return tokenized_batch


encoded_train = train_ds.map(preprocess, batched=True)
encoded_val = val_ds.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

encoded_train.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'label']
)

encoded_val.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'label']
)


#Parameters:
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


def format_millions(n):
    return f"{n/1e6:.2f}M"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': acc,
        'f1': f1
    }


from dataclasses import dataclass


@dataclass
class RunStats:
    name: str
    total_params: int
    trainable_params: int
    train_time_sec: float
    val_accuracy: float
    val_f1: float


# Full parameter tunning:

print('=== FULL FINE-TUNING (all parameters trainable) ===')

full_model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=num_labels
)

full_total, full_trainable = count_parameters(full_model)

print(f'Full FT - Total params:     {format_millions(full_total)}')
print(f'Full FT - Trainable params: {format_millions(full_trainable)}')

full_epochs = 2

training_args_full = TrainingArguments(
    output_dir='./ft_full',
    eval_strategy='epoch',
    save_strategy='no',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=full_epochs,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_steps=50,
    report_to='none',
)

full_trainer = Trainer(
    model=full_model.to(device),
    args=training_args_full,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

start_time = time.time()

full_trainer.train()

full_train_time = time.time() - start_time

full_eval = full_trainer.evaluate()

full_acc = full_eval['eval_accuracy']
full_f1 = full_eval['eval_f1']

print(f'Full FT - Train time: {full_train_time:.2f} s')
print(f'Full FT - Val accuracy: {full_acc:.4f}, F1: {full_f1:.4f}')

full_stats = RunStats(
    name='Full Fine-Tuning',
    total_params=full_total,
    trainable_params=full_trainable,
    train_time_sec=full_train_time,
    val_accuracy=full_acc,
    val_f1=full_f1,
)

# LORA:
 print('=== LoRA FINE-TUNING (PEFT: only adapter weights trainable) ===')

lora_base_model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=num_labels
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='none',
    target_modules=['q_lin', 'k_lin', 'v_lin', 'out_lin'],
)

lora_model = get_peft_model(lora_base_model, lora_config)

lora_total, lora_trainable = count_parameters(lora_model)

print(f'LoRA FT - Total params: {format_millions(lora_total)}')
print(f'LoRA FT - Trainable params: {format_millions(lora_trainable)}')

reduction_factor = full_trainable / lora_trainable
print(f'Trainable parameter reduction: ~{reduction_factor:.1f}x fewer trainable params')

lora_epochs = 2

training_args_lora = TrainingArguments(
    output_dir='./ft_lora',
    eval_strategy='epoch',
    save_strategy='no',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=lora_epochs,
    weight_decay=0.01,
    learning_rate=2e-4,
    logging_steps=50,
    report_to='none',
)

lora_trainer = Trainer(
    model=lora_model.to(device),
    args=training_args_lora,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

start_time = time.time()

lora_trainer.train()

lora_train_time = time.time() - start_time

lora_eval = lora_trainer.evaluate()

lora_acc = lora_eval['eval_accuracy']
lora_f1 = lora_eval['eval_f1']

print(f'\nLoRA FT - Train time: {lora_train_time:.2f} s')
print(f'LoRA FT - Val accuracy: {lora_acc:.4f}, F1: {lora_f1:.4f}')

lora_stats = RunStats(
    name='LoRA Fine-Tuning',
    total_params=lora_total,
    trainable_params=lora_trainable,
    train_time_sec=lora_train_time,
    val_accuracy=lora_acc,
    val_f1=lora_f1,
)