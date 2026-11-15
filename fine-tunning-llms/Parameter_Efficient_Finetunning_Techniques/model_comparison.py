!pip install transformers datasets torch scikit-learn -q

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")


# Helper functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_gpu_memory():
    return (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else 0.0
    )


# Load SST-2 validation subset (500 samples for speed)
dataset = load_dataset("glue", "sst2")
test_subset = dataset["validation"].select(range(500))

print(f" Test subset loaded: {len(test_subset)} samples")

#Compare models

models_to_compare = [
    {"name": "DistilBERT", "id": "distilbert-base-uncased-finetuned-sst-2-english"},
    {"name": "BERT Base", "id": "textattack/bert-base-uncased-SST-2"},
    {"name": "RoBERTa Base", "id": "textattack/roberta-base-SST-2"},
]

print(" Models to compare:")
for m in models_to_compare:
    print(f"  • {m['name']}: {m['id']}")


def evaluate_model(model, tokenizer, dataset, batch_size=32):
    """Evaluate model: accuracy + speed + GPU memory."""
    model.eval()
    model.to(device)

    texts = dataset["sentence"]
    labels = np.array(dataset["label"])
    n = len(texts)
    all_preds = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    for i in range(0, n, batch_size):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds)

    total_time = time.time() - start
    speed = n / total_time if total_time > 0 else float("inf")
    acc = accuracy_score(labels, all_preds)
    gpu_mem = measure_gpu_memory()

    return acc, speed, gpu_mem


# Evaluate all models
results = []

for cfg in models_to_compare:
    print(f"\n{'=' * 60}")
    print(f" Evaluating {cfg['name']}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["id"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["id"])

    total_params = count_parameters(model)
    print(f" Total params: {total_params / 1e6:.2f}M")

    acc, speed, gpu_mem = evaluate_model(
        model,
        tokenizer,
        test_subset
    )

    print(f" Accuracy: {acc:.4f}")
    print(f" Speed: {speed:.2f} samples/sec")

    if torch.cuda.is_available():
        print(f" Peak GPU memory: {gpu_mem:.2f} MB")

    results.append({
        "name": cfg["name"],
        "model_id": cfg["id"],
        "params": total_params,
        "accuracy": acc,
        "speed": speed,
        "gpu_mem": gpu_mem,
    })

print("\nAll evaluations complete!")



#Visualization 

names = [r["name"] for r in results]
params_m = [r["params"] / 1e6 for r in results]
accs = [r["accuracy"] for r in results]
speeds = [r["speed"] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Chart 1: Parameter counts
bars1 = axes[0].bar(
    names,
    params_m,
    color=["steelblue", "coral", "seagreen"],
    alpha=0.85
)
axes[0].set_ylabel("Parameters (Millions)", fontweight="bold")
axes[0].set_title("Model Size (Total Parameters)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=15)
axes[0].grid(axis="y", alpha=0.3)

for bar, val in zip(bars1, params_m):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        val + 2,
        f"{val:.1f}M",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

# Chart 2: Speed
bars2 = axes[1].bar(
    names,
    speeds,
    color=["steelblue", "coral", "seagreen"],
    alpha=0.85
)
axes[1].set_ylabel("Samples per Second", fontweight="bold")
axes[1].set_title("Inference Speed (500 Samples)", fontweight="bold")
axes[1].tick_params(axis="x", rotation=15)
axes[1].grid(axis="y", alpha=0.3)

for bar, val in zip(bars2, speeds):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        val + 5,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

# Chart 3: Accuracy
bars3 = axes[2].bar(
    names,
    accs,
    color=["steelblue", "coral", "seagreen"],
    alpha=0.85
)
axes[2].set_ylabel("Accuracy", fontweight="bold")
axes[2].set_title("Accuracy on SST-2 Validation", fontweight="bold")
axes[2].set_ylim(0, 1.0)
axes[2].tick_params(axis="x", rotation=15)
axes[2].grid(axis="y", alpha=0.3)


sample_texts = [
    "I absolutely loved this movie, it was fantastic!",
    "This was the worst product I have ever bought.",
    "The performance was okay, nothing special but not terrible.",
]

print("=" * 60)
print("SAMPLE PREDICTION COMPARISON")
print("=" * 60)

for cfg in results:
    print(f"\nModel: {cfg['name']}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"]
    ).to(device)

    model.eval()

    enc = tokenizer(
        sample_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

    for i, (text, pred, prob) in enumerate(
        zip(sample_texts, preds, probs), 1
    ):
        label = "positive" if pred == 1 else "negative"
        confidence = prob[pred]

        print(f"  [{i}] {text}")
        print(
            f"      → {label.upper()} ({confidence:.3f} confidence)"
        )

print(
    "\nNotice: All models agree on clear cases, "
    "may differ on neutral/ambiguous text"
)