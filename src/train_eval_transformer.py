"""
Fine-tune DistilBERT
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data import get_splits, SEED, LABELS
from src.plots import plot_confusion

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
for _d in (MODEL_DIR, OUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
EPOCHS = 3
BATCH = 16

LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                             padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    torch.manual_seed(SEED)
    train, val, test = get_splits(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Label encoding and dataset preparation
    ds_train = TextDataset(train["statement"], train["status"].map(LABEL2ID), tok)
    ds_val = TextDataset(val["statement"], val["status"].map(LABEL2ID), tok)
    ds_test = TextDataset(test["statement"], test["status"].map(LABEL2ID), tok)

    weights = compute_class_weight(
        "balanced", classes=np.arange(len(LABELS)),
        y=train["status"].map(LABEL2ID).values)
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Load the pre-trained model and prepare the Trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    out_dir = MODEL_DIR / "distilbert"
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=SEED,
        save_total_limit=1,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_val,
        compute_metrics=_metrics, class_weights=class_weights,
        data_collator=DataCollatorWithPadding(tokenizer=tok),  # dynamic padding
    )
    trainer.train()

    # Test-set evaluation, comparable to the classic models.
    logits = trainer.predict(ds_test).predictions
    preds = np.argmax(logits, axis=1)
    y_true = test["status"].map(LABEL2ID).values
    macro = f1_score(y_true, preds, average="macro")
    acc = accuracy_score(y_true, preds)
    pred_labels = [ID2LABEL[p] for p in preds]
    report = classification_report(test["status"], pred_labels,
                                   output_dict=True, zero_division=0)

    # Plot the confusion matrix
    plot_confusion(test["status"], pred_labels, LABELS,
                   "Confusion matrix — DistilBERT",
                   FIG_DIR / "confusion_distilbert.png")

    metrics = {"model": MODEL_NAME, "test_accuracy": round(acc, 4),
               "test_macro_f1": round(macro, 4), "per_class": report}
    (OUT_DIR / "transformer_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDistilBERT test: accuracy={acc:.3f}  macro-F1={macro:.3f}")
    print(f"Wrote {OUT_DIR / 'transformer_metrics.json'}")


if __name__ == "__main__":
    main()
