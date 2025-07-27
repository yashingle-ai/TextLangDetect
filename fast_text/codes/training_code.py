import fasttext
import os
import pandas as pd
import random
from collections import defaultdict
from itertools import product
import argparse

# Argument Parser
# Allows setting file paths and output from command line
parser = argparse.ArgumentParser(description="Train and evaluate FastText for language identification")

parser.add_argument("--train_file", type=str, default="C:\\Users\\Lenovo\\OneDrive\\Desktop\\votting algo\\fastext\\train_ft.txt", help="Path to the training file")
parser.add_argument("--valid_file", type=str, default="C:\\Users\\Lenovo\\OneDrive\\Desktop\\votting algo\\fastext\\dev_ft.txt", help="Path to the validation file")
parser.add_argument("--test_file", type=str, default="C:\\Users\\Lenovo\\OneDrive\\Desktop\\votting algo\\fastext\\test_ft.txt", help="Path to the test file")
parser.add_argument("--output_model", type=str, default="fasttext_best_model_final_2.bin", help="Filename for saving best model")
parser.add_argument("--output_report", type=str, default="C:\\Users\\Lenovo\\OneDrive\\Desktop\\votting algo\\fastext\\fasttext_per_label_accuracy_2.csv", help="CSV output report path")

args = parser.parse_args()

# Hyperparameter Grid
param_grid = {
    "lr": [0.3],          # Learning rate
    "epoch": [25],        # Number of training epochs
    "wordNgrams": [2],    # Use of n-grams
    "minCount": [1]       # Minimum word count threshold
}

# Evaluate Accuracy 
def evaluate_model(model, file_path):
    """
    Evaluate FastText model on a given dataset.
    Returns overall accuracy.
    """
    result = model.test(file_path)
    return result[1]  # Return validation accuracy

# Train with Grid Search 
best_model = None
best_accuracy = 0
best_params = None

for lr, epoch, wordNgrams, minCount in product(param_grid["lr"], param_grid["epoch"], param_grid["wordNgrams"], param_grid["minCount"]):
    print(f"🔹 Training model with lr={lr}, epoch={epoch}, wordNgrams={wordNgrams}, minCount={minCount}")
    
    try:
        model = fasttext.train_supervised(
            input=args.train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            minCount=minCount,
            loss="softmax",  # Better for multi-class classification
            verbose=0
        )
    except RuntimeError as e:
        print(f" Training failed. Error: {e}")
        continue

    val_accuracy = evaluate_model(model, args.valid_file)
    print(f" Validation Accuracy: {val_accuracy * 100:.2f}%\n")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model
        best_params = (lr, epoch, wordNgrams, minCount)

# Save Best Model
best_model.save_model(args.output_model)
print(f"\n Best Model Saved to: {args.output_model}")
print(f" Best Hyperparameters: lr={best_params[0]}, epoch={best_params[1]}, wordNgrams={best_params[2]}, minCount={best_params[3]}")
print(f" Best Validation Accuracy: {best_accuracy * 100:.2f}%")

#Function: Accuracy Per Language 
def evaluate_per_label(model, file_path):
    """
    Evaluate the model accuracy per language label.
    Returns total samples and a dictionary of accuracy per label.
    """
    label_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        true_label = parts[0]
        text = " ".join(parts[1:])
        predicted_label, _ = model.predict(text, k=1)

        label_counts[true_label] += 1
        if predicted_label[0] == true_label:
            correct_counts[true_label] += 1

    # Calculate accuracy per label
    label_accuracy = {
        label: (correct_counts[label] / label_counts[label]) * 100 if label_counts[label] > 0 else 0
        for label in label_counts
    }

    return len(lines), label_accuracy

#  Evaluate Model on All Splits
train_samples, train_acc = evaluate_per_label(best_model, args.train_file)
valid_samples, valid_acc = evaluate_per_label(best_model, args.valid_file)
test_samples, test_acc = evaluate_per_label(best_model, args.test_file)

#  Create CSV Report 
labels = sorted(set(train_acc.keys()) | set(valid_acc.keys()) | set(test_acc.keys()))

report = []
for label in labels:
    report.append({
        "Label": label,
        "Train Accuracy (%)": train_acc.get(label, 0),
        "Validation Accuracy (%)": valid_acc.get(label, 0),
        "Test Accuracy (%)": test_acc.get(label, 0),
    })

df = pd.DataFrame(report)

print("\n Per-Label Accuracy Report:")
print(df)

# Save to CSV
df.to_csv(args.output_report, index=False)
print(f"\n Report saved to: {args.output_report}")
