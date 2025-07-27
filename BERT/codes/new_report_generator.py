import os
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

#  Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Define Functions
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)  # ⚠️ Reduced max_length to avoid OOM

def predict_languages(sentences, batch_size=32):
    predictions = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad(), autocast():  
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predictions.extend(batch_predictions)

    return [id_to_label[pred] for pred in predictions]

#  Wrap the script inside __main__
if __name__ == "__main__":
    folder_path = r"C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\data_set"

    #  Load data from text files
    data, labels = [], []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            language = os.path.splitext(file_name)[0]
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                sentences = file.readlines()

            data.extend(sentences)
            labels.extend([language] * len(sentences))

    #  Convert to DataFrame
    df = pd.DataFrame({"text": data, "label": labels})

    #  Ensure each language has at least 10K samples
    df = df.groupby("label").filter(lambda x: len(x) >= 10000)

    #  Convert labels to numeric
    label_to_id = {label: i for i, label in enumerate(df["label"].unique())}
    id_to_label = {v: k for k, v in label_to_id.items()}
    df["label"] = df["label"].map(label_to_id)

    #  Split into Train (80%), Validation (10%), Test (10%)
    train_data, temp_data = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42)

    #  Convert to Hugging Face dataset
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)

    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

    #  Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("google/muril-base-cased")

    #  Tokenize dataset (Fix multiprocessing issue)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1)  # ⚠️ Set num_proc=1 for Windows

    #  Load BERT (MuRIL) model
    model = BertForSequenceClassification.from_pretrained("google/muril-base-cased", num_labels=len(label_to_id)).to(device)

    #  Define Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, target_names=id_to_label.values(), output_dict=True)
        return {"accuracy": accuracy, "classification_report": report}

    #  Training arguments (Optimized)
    training_args = TrainingArguments(
        output_dir="./muril_language_classifier",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,
        fp16=True,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    #  Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    #  Train the model
    trainer.train()

    #  Save model
    model.save_pretrained("./muril_language_classifier")
    tokenizer.save_pretrained("./muril_language_classifier")

    #  Set model to evaluation mode
    model.eval()

    #  Compute Accuracy per Language for Train, Validation, and Test Sets
    def generate_language_reports(dataset, dataset_name):
        y_true = dataset["label"].tolist()
        y_pred = predict_languages(dataset["text"].tolist())

        #  Generate Classification Report
        class_report = classification_report(y_true, [label_to_id[pred] for pred in y_pred], target_names=id_to_label.values())

        #  Save classification report
        report_path = f"{dataset_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(class_report)

        #  Generate Confusion Matrix
        conf_matrix = confusion_matrix(y_true, [label_to_id[pred] for pred in y_pred])

        #  Plot Confusion Matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=id_to_label.values(), yticklabels=id_to_label.values())
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {dataset_name.capitalize()}")
        plt.savefig(f"{dataset_name}_confusion_matrix.png")  #  Save the figure
        plt.show()

        print(f"{dataset_name.capitalize()} report saved as {report_path}")
        print(f"{dataset_name.capitalize()} confusion matrix saved as {dataset_name}_confusion_matrix.png")

    #  Generate reports for Train, Validation, and Test Sets
    generate_language_reports(train_data, "train")
    generate_language_reports(val_data, "validation")
    generate_language_reports(test_data, "test")

    print(" All reports generated successfully!")
