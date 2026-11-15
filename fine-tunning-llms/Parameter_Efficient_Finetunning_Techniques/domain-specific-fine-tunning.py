# Demo: Domain-Specific Classification Fine-Tuning
#
# Scenario:
#
# You are a data scientist at a news aggregation platform that automatically
# categorizes articles into four news categories: World, Sports, Business,
# and Sci/Tech. Your platform currently uses a pre-trained DistilBERT model
# that was trained on general English text but has never seen news articles
# before. The base model performs poorly on news classification because it
# lacks domain-specific knowledge about news terminology, context, and writing
# style — a phenomenon called "topic drift" where a pre-trained model
# misclassifies domain-specific content.
#
# Learning Objectives:
#
# • Understand the Concept of Domain Shift and Topic Drift in Pre-Trained Models
# • Implement End-to-End Fine-Tuning Workflows with Hugging Face Trainer
# • Evaluate Model Performance with Comprehensive Metrics Beyond Accuracy
# • Understanding classification report and accuracy metrics.


# Setup and Install Dependencies

!pip install datasets transformers scikit-learn torch -q

import copy
import math
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

print("Setup complete!")
print("All libraries imported successfully")


# AG NEWS DATASET

dataset = load_dataset("ag_news")

print("\nAG NEWS DATASET")
print(dataset)

# Get label names
label_names = dataset["train"].features["label"].names
print(f"\nLabel names: {label_names}")
print(f"  Number of classes: {len(label_names)}")

# Keep original texts for potential error analysis later
test_texts = dataset["test"]["text"]

# Create train/validation split from training set
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds = split["test"]
test_ds = dataset["test"]

print(f"\nDataset splits:")
print(f"  Training:   {len(train_ds):,} samples")
print(f"  Validation: {len(val_ds):,} samples")
print(f"  Test:       {len(test_ds):,} samples")

# Show a few examples
print("Sample articles:")
for i in range(3):
    print(f"\n{i+1}. [{label_names[train_ds[i]['label']]}]")
    print(f"   {train_ds[i]['text'][:150]}...")


# Tokenizer and Preprocessing

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def preprocess(examples):
    """Tokenize text with padding and truncation."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


print("\n" + "=" * 60)
print("TOKENIZING DATASETS")
print("=" * 60)
print(f"Model checkpoint: {checkpoint}")
print("Max length: 128 tokens\n")

# Apply tokenization
encoded_train = train_ds.map(preprocess, batched=True)
encoded_val = val_ds.map(preprocess, batched=True)
encoded_test = test_ds.map(preprocess, batched=True)

# Set format for PyTorch (so Trainer can use it)
encoded_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
encoded_val.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
encoded_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

num_labels = len(label_names)

print("Tokenization complete!")
print("Format: PyTorch tensors ready for training")

# Data Subsampling

max_train_samples = 20000
max_val_samples = 2000
max_test_samples = 2000

encoded_train_small = encoded_train.shuffle(seed=42).select(
    range(min(max_train_samples, len(encoded_train)))
)

encoded_val_small = encoded_val.shuffle(seed=42).select(
    range(min(max_val_samples, len(encoded_val)))
)

encoded_test_small = encoded_test.shuffle(seed=42).select(
    range(min(max_test_samples, len(encoded_test)))
)

print("\nUsing subset of data for faster training/evaluation:")
print(f"  Training subset:   {len(encoded_train_small):,} samples")
print(f"  Validation subset: {len(encoded_val_small):,} samples")
print(f"  Test subset:       {len(encoded_test_small):,} samples")


# Load Base Model

print("\nLOADING BASE MODEL")
print(f"Checkpoint: {checkpoint}")
print(f"Number of labels: {num_labels}")
print("Note: The classifier head is randomly initialized!")
print("This model has NOT been trained on AG News yet.\n")

# Base model (pretrained LM, NOT fine-tuned on AG News)
base_model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=num_labels,
)

# Clone for fine-tuning (same starting weights)
fine_model = copy.deepcopy(base_model)

print("Models loaded!")
print("  - Base model: Ready for evaluation (no training)")
print("  - Fine-tune model: Cloned from base (ready for training)")


# Metrics Function

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1-score."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


print("\nMetrics function defined!")
print("  Will compute: accuracy, precision, recall, F1-score")


# Evaluate BASE Model

eval_args = TrainingArguments(
    output_dir="./agnews_eval",
    per_device_eval_batch_size=32,
    do_train=False,
    do_eval=True,
    logging_steps=50,
    report_to="none",
)

base_trainer = Trainer(
    model=base_model,
    args=eval_args,
    eval_dataset=encoded_test_small,
    compute_metrics=compute_metrics,
)

print("\nEVALUATING BASE MODEL (NO FINE-TUNING)")
print("  Expected: Low accuracy (near random guessing)\n")

base_eval = base_trainer.evaluate()

base_loss = base_eval["eval_loss"]
base_perplexity = math.exp(base_loss)

print("BASE MODEL METRICS")

for k, v in base_eval.items():
    print(f"  {k}: {v:.4f}")

print(f"  approx perplexity: {base_perplexity:.4f}")

# Get predictions for confusion matrix & error analysis
base_pred_output = base_trainer.predict(encoded_test_small)

base_logits = base_pred_output.predictions
base_y_pred = np.argmax(base_logits, axis=-1)
base_y_true = base_pred_output.label_ids

print("\nBase model evaluation complete!")

# Fine-Tune Model on AG News

train_args = TrainingArguments(
    output_dir="./agnews_finetuned",
    eval_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none",
)

fine_trainer = Trainer(
    model=fine_model,
    args=train_args,
    train_dataset=encoded_train_small,
    eval_dataset=encoded_val_small,
    compute_metrics=compute_metrics,
)

print("\nFINE-TUNING MODEL ON AG NEWS (DOMAIN-SPECIFIC)")
print(f"Epochs: {train_args.num_train_epochs}")
print(f"Batch size: {train_args.per_device_train_batch_size}")
print(f"Weight decay: {train_args.weight_decay}")
print("\nWatch the loss decrease as the model learns news topics.\n")

fine_trainer.train()

print("\nFine-tuning complete!")


# Evaluate Fine-Tuned Model on Test Set

print("\nEVALUATING FINE-TUNED MODEL ON TEST SET")

fine_eval = fine_trainer.evaluate(encoded_test_small)
fine_loss = fine_eval["eval_loss"]
fine_perplexity = math.exp(fine_loss)

print("\nFINE-TUNED MODEL METRICS")

for k, v in fine_eval.items():
    print(f"  {k}: {v:.4f}")

print(f"  approx_perplexity: {fine_perplexity:.4f}")


# Predictions for fine-tuned model

fine_pred_output = fine_trainer.predict(encoded_test_small)
fine_logits = fine_pred_output.predictions
fine_y_pred = np.argmax(fine_logits, axis=-1)
fine_y_true = fine_pred_output.label_ids


# Improvement summary

base_acc = base_eval["eval_accuracy"]
fine_acc = fine_eval["eval_accuracy"]

print("\nIMPROVEMENT SUMMARY")
print(f"Base accuracy: {base_acc:.4f}")
print(f"Fine-tuned accuracy: {fine_acc:.4f}")
print(f"Absolute improvement: {(fine_acc - base_acc):.4f}")
print(
    f"Relative improvement: "
    f"{((fine_acc - base_acc) / max(base_acc, 1e-6) * 100):.2f}%"
)
print("\nEvaluation complete!")


# Classification Reports

print("\nCLASSIFICATION REPORT: BASE MODEL")
print(
    classification_report(
        base_y_true,
        base_y_pred,
        target_names=label_names,
        digits=4,
    )
)

print("\nCLASSIFICATION REPORT: FINE-TUNED MODEL")
print(
    classification_report(
        fine_y_true,
        fine_y_pred,
        target_names=label_names,
        digits=4,
    )
)

print("\nNotice how fine-tuning improves precision, recall, and F1 for each topic!")


# Accuracy

labels_chart = [
    "Base Model\n(No Fine-Tuning)",
    "Fine-Tuned Model\n(Domain-Specific)",
]

accuracies = [base_acc, fine_acc]

plt.figure(figsize=(10, 7))

bars = plt.bar(
    labels_chart,
    accuracies,
    color=["lightcoral", "seagreen"],
    alpha=0.85,
    width=0.5,
)

plt.ylim(0.0, 1.0)
plt.ylabel("Accuracy", fontweight="bold", fontsize=13)
plt.title(
    "Base vs Fine-Tuned Model Accuracy on AG News",
    fontweight="bold",
    fontsize=15,
    pad=15,
)
plt.grid(axis="y", alpha=0.3, linestyle="--")


# Annotate bars with accuracy values

for bar in bars:
    h = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        h + 0.02,
        f"{h:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.4",
        ),
    )

plt.tight_layout()
plt.show()


print("\nAccuracy Analysis:")
print(
    f"  Base model:      {base_acc:.4f} "
    f"(essentially random guessing for 4 classes)"
)

print(
    f"  Fine-tuned model: {fine_acc:.4f} "
    f"(learned domain-specific patterns)"
)

print(
    f"  Improvement:     {(fine_acc - base_acc):.4f} absolute "
    f"({((fine_acc - base_acc) / max(base_acc, 1e-6) * 100):.2f}% relative)"
)

print("\nFine-tuning transforms a generic LM into a domain expert.")
