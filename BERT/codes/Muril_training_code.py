import os
import argparse
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.cuda.amp import autocast

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(description="Train a BERT-based language classification model using MuRIL.")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the folder containing text files for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--output_dir", type=str, default="./muril_language_classifier", help="Directory to save the trained model.")
    
    return parser.parse_args()


def load_data(folder_path):
    """Load text data from the specified folder and return as a Pandas DataFrame."""
    data, labels = [], []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            language = os.path.splitext(file_name)[0]
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                sentences = file.readlines()

            data.extend(sentences)
            labels.extend([language] * len(sentences))

    df = pd.DataFrame({"text": data, "label": labels})

    # Ensure each language has at least 10K samples
    df = df.groupby("label").filter(lambda x: len(x) >= 10000)

    return df


def tokenize_function(examples):
    """Tokenize text data using the BERT tokenizer."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


def predict_languages(sentences, batch_size=32):
    """Perform batched inference for language classification."""
    predictions = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad(), autocast():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predictions.extend(batch_predictions)

    return [id_to_label[pred] for pred in predictions]


def compute_metrics(eval_pred):
    """Compute classification metrics including accuracy and F1-score."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=id_to_label.values(), output_dict=True)
    
    return {"accuracy": accuracy, "classification_report": report}


def generate_language_reports(dataset, dataset_name):
    """Generate classification reports and confusion matrix for a dataset."""
    y_true = dataset["label"].tolist()
    y_pred = predict_languages(dataset["text"].tolist())

    # Generate Classification Report
    class_report = classification_report(y_true, [label_to_id[pred] for pred in y_pred], target_names=id_to_label.values())

    # Save classification report to a text file
    report_path = f"{dataset_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(class_report)

    # Generate and save Confusion Matrix
    conf_matrix = confusion_matrix(y_true, [label_to_id[pred] for pred in y_pred])

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=id_to_label.values(), yticklabels=id_to_label.values())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {dataset_name.capitalize()}")
    plt.savefig(f"{dataset_name}_confusion_matrix.png")
    plt.show()

    print(f"{dataset_name.capitalize()} report saved as {report_path}")
    print(f"{dataset_name.capitalize()} confusion matrix saved as {dataset_name}_confusion_matrix.png")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load dataset
    df = load_data(args.dataset_path)

    # Convert labels to numeric
    label_to_id = {label: i for i, label in enumerate(df["label"].unique())}
    id_to_label = {v: k for k, v in label_to_id.items()}
    df["label"] = df["label"].map(label_to_id)

    # Split dataset into train (80%), validation (10%), and test (10%)
    train_data, temp_data = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42)

    # Convert data to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)

    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("google/muril-base-cased")

    # Tokenize dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1)

    # Load pre-trained MuRIL model for sequence classification
    model = BertForSequenceClassification.from_pretrained("google/muril-base-cased", num_labels=len(label_to_id)).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir="./logs",
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,
        fp16=True,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Set model to evaluation mode
    model.eval()

    # Generate evaluation reports
    generate_language_reports(train_data, "train")
    generate_language_reports(val_data, "validation")
    generate_language_reports(test_data, "test")

    print(" All reports generated successfully!")
    
