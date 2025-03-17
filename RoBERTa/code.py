import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#  Define dataset path
DATASET_DIR = "C:/Users/yashi/OneDrive/Desktop/fasttext/dataset"

#  Auto-detect language files
language_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".txt")]

#  Extract language codes from filenames (before "_")
languages = sorted(set(f.split("_")[0] for f in language_files))

#  Read text files
data = []
for file in language_files:
    lang = file.split("_")[0]  # Extract language code
    file_path = os.path.join(DATASET_DIR, file)

    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
        
        for sentence in sentences:
            data.append({"text": sentence, "language": lang})

#  Convert to DataFrame
df = pd.DataFrame(data)

#  Ensure we have data
if df.empty:
    raise ValueError("🚨 No data loaded! Check if text files have content.")

#  Create Label Mapping
label_map = {lang: i for i, lang in enumerate(languages)}
df["label"] = df["language"].map(label_map)

#  Split into Train, Validation, and Test
train_texts, temp_texts, train_labels, temp_labels = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

#  Convert to Hugging Face Dataset
def create_hf_dataset(texts, labels):
    return Dataset.from_dict({"text": list(texts), "label": list(labels)})

hf_dataset = DatasetDict({
    "train": create_hf_dataset(train_texts, train_labels),
    "validation": create_hf_dataset(val_texts, val_labels),
    "test": create_hf_dataset(test_texts, test_labels),
})

#  Load XLM-Roberta Model & Tokenizer
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#  Tokenization Function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

#  Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(languages))

#  Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    fp16=True,
    load_best_model_at_end=True
)

#  Compute Metrics Function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#  Train the Model
trainer.train()

# Evaluate on Test Set
test_results = trainer.evaluate(tokenized_datasets["test"])
print("Test Results:", test_results)

#  Generate Classification Report
y_true = test_labels.tolist()
y_pred = np.argmax(trainer.predict(tokenized_datasets["test"]).predictions, axis=1)

print("\nDetailed Per-Language Report:")
print(classification_report(y_true, y_pred, target_names=languages))
