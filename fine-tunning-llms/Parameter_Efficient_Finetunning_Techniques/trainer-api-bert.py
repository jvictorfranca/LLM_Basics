# Demo: Configure Trainer API for BERT

# Scenario:
#
# You're a machine learning engineer tasked with building a
# sentiment analysis system for movie reviews.
#
# The goal is to classify movie reviews as positive or negative
# with high accuracy while keeping the implementation simple
# and maintainable.
#
# Instead of writing complex training loops from scratch,
# you need to leverage Hugging Face's Trainer API to quickly
# fine-tune a pre-trained DistilBERT model on the IMDb dataset.


# Learning Objectives:
#
# - Understand the Trainer API workflow
# - Master text preprocessing for transformers
# - Implement automated evaluation
# - Apply transfer learning


# Install required packages
# !pip install transformers datasets scikit-learn accelerate -q

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

print("All imports successful")

# Install required packages
!pip install transformers datasets scikit-learn accelerate -q

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

print("All imports successful")


# Loading IMDb dataset...
print("Loading IMDb dataset...")

# Load the full IMDb dataset
raw_datasets = load_dataset("imdb")

print("Dataset loaded:")
print(f"Train samples: {len(raw_datasets['train'])}")
print(f"Test samples: {len(raw_datasets['test'])}")

# Create a small subset for faster training (demo purposes)
print("\nCreating small subset for demo...")

small_train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(2000))
small_test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(1000))

print("Subset created:")
print("Train: 2,000 samples")
print("Test: 1,000 samples")


# Show a sample
print("\nSample review:")

sample = small_train_dataset[0]

print(f"Text: {sample['text'][:200]}...")
print(
    f"Label: {sample['label']} "
    f"({'Positive' if sample['label'] == 1 else 'Negative'})"
)

print("Loading tokenizer and model...")

model_name = "distilbert-base-uncased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded: {model_name}")

# Load model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Binary classification: positive or negative
)

print(f"Model loaded: {model_name}")
print("Model type: Sequence Classification")
print("Number of labels: 2 (Positive/Negative)")
print(f"Parameters: {model.num_parameters():,}")


def tokenize_function(batch):
    """
    Tokenize a batch of text samples.
    """
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,  # Maximum sequence length
    )


print("Tokenizing datasets...")

# Tokenize training and test sets
tokenized_train = small_train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_test = small_test_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenization complete!")

# Remove the original text column (no longer needed)
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

# Convert to PyTorch format
tokenized_train = tokenized_train.with_format("torch")
tokenized_test = tokenized_test.with_format("torch")

print("Datasets prepared for training")
print(f"  Train dataset columns: {tokenized_train.column_names}")
print(f"  Test dataset columns: {tokenized_test.column_names}")


# Show tokenized example
print("\nTokenized sample:")

sample_tokens = tokenized_train[0]

print(f"  Input IDs shape: {sample_tokens['input_ids'].shape}")
print(
    f"  First 10 tokens: "
    f"{sample_tokens['input_ids'][:10].tolist()}"
)
print(f"  Label: {sample_tokens['label']}")

# Compute Metrics

def compute_metrics(eval_pred):
    """Compute metrics from predictions and labels."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")

    return {
        "accuracy": acc,
        "f1": f1,
    }


print("Metrics function defined!")
print("  Will compute: Accuracy and F1 Score")


# Training Arguments

training_args = TrainingArguments(
    output_dir="bert_imdb_trainer_demo",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="bert_imdb_trainer_demo/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none",
)

print("Training arguments configured!")
print("\nTraining Configuration:")
print(f"  Output directory: bert_imdb_trainer_demo")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Evaluation: {training_args.eval_strategy}")
print(f"  Best model metric: accuracy")
print(f"  Reporting: {training_args.report_to}")


# Create the Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Trainer created successfully!")
print(f"Trainer Summary:")
print(f"  Model: {model}")
print(f"  Train samples: {len(tokenized_train):,}")
print(f"  Eval samples: {len(tokenized_test):,}")
print(
    f"  Total training steps: "
    f"{training_args.num_train_epochs * "
    f"(len(tokenized_train) // training_args.per_device_train_batch_size)}"
)


# Train the Model

print("-" * 70)
print("STARTING TRAINING")
print("-" * 70)
print("This may take a few minutes...\n")

# Start training
train_result = trainer.train()

print("\n" + "-" * 70)
print("TRAINING COMPLETE!")
print("-" * 70)

# Show training metrics
print(f"\nTraining Summary:")
print(
    f"  Total steps: "
    f"{train_result.metrics.get('train_steps', 'N/A')}"
)
print(
    f"  Training time: "
    f"{train_result.metrics.get('train_runtime', 0):.2f} seconds"
)
print(
    f"  Final loss: "
    f"{train_result.metrics.get('train_loss', 'N/A'):.4f}"
)

# Plotando as métricas de treinamento
import matplotlib.pyplot as plt

# Extrair métricas do histórico do trainer
train_losses = []
eval_losses = []
eval_accuracies = []

for log in trainer.state.log_history:
    if 'loss' in log:  # Training loss
        train_losses.append(log['loss'])

    if 'eval_loss' in log:  # Evaluation loss
        eval_losses.append(log['eval_loss'])

    if 'eval_accuracy' in log:  # Evaluation accuracy
        eval_accuracies.append(log['eval_accuracy'])


# Plot Loss
fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))

ax.plot(train_losses, label='Training Loss')

# Assuming one eval_loss point per epoch,
# if multiple epochs, use range(1, num_train_epochs + 1)
if eval_losses:
    ax.plot(
        [len(train_losses) - 1],
        [eval_losses[-1]],
        'rx',
        markersize=10,
        label='Validation Loss (End of Epoch)'
    )

    ax.annotate(
        f'{eval_losses[-1]:.4f}',
        (len(train_losses) - 1, eval_losses[-1]),
        textcoords='offset points',
        xytext=(0, 10),
        ha='center'
    )

ax.set_title('Loss over Training Steps')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


# Avaliando no conjunto de teste
print("=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

# Evaluate on test set
eval_metrics = trainer.evaluate()

print("\nEvaluation Complete!")
print("\nTest Set Metrics:")
print(
    f"  Accuracy: {eval_metrics['eval_accuracy']:.4f} "
    f"({eval_metrics['eval_accuracy'] * 100:.2f}%)"
)
print(f"  F1 Score: {eval_metrics['eval_f1']:.4f}")
print(f"  Loss: {eval_metrics['eval_loss']:.4f}")
print(
    f"  Samples/second: "
    f"{eval_metrics.get('eval_samples_per_second', 'N/A'):.2f}"
)
print("=" * 70)


# Fine-tuned Model
print("Saving fine-tuned model...")
save_path = "bert_imdb_bert_demo/final_model"

# Save model
trainer.save_model(save_path)

# Save tokenizer
tokenizer.save_pretrained(save_path)

print(f"Model saved to: {save_path}")
print("  Saved files:")
print("    - pytorch_model.bin (model weights)")
print("    - config.json (model configuration)")
print("    - tokenizer files")
print("\nYou can now load this model with:")
print(
    f"    model = AutoModelForSequenceClassification.from_pretrained("
    f"'{save_path}')"
)
print(
    f"    tokenizer = AutoTokenizer.from_pretrained('{save_path}')"
)
