import fasttext
import os
import pandas as pd
import random
from collections import defaultdict
from itertools import product

# Define file paths
train_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\label_data_10k\\train.txt"
valid_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\label_data_10k\\valid.txt"
test_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\label_data_10k\\test.txt"

# Hyperparameter grid
param_grid = {
    "lr": [0.05, 0.1, 0.3],      # Reduced max learning rate for stability
    "epoch": [10, 25, 50],       
    "wordNgrams": [1, 2, 3],     
    "minCount": [1, 2, 5]        
}

# Function to clean data (removes empty or corrupted lines)
def clean_data(file_path, output_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    cleaned_lines = [line for line in lines if line.strip() and line.startswith("__label__")]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f" Cleaned data saved to {output_path}")

# Clean datasets before training
clean_data(train_file, "clean_train.txt")
clean_data(valid_file, "clean_valid.txt")
clean_data(test_file, "clean_test.txt")

# Update paths to cleaned files
train_file = "clean_train.txt"
valid_file = "clean_valid.txt"
test_file = "clean_test.txt"

# Function to evaluate model
def evaluate_model(model, file_path):
    result = model.test(file_path)
    return result[1]  # Validation accuracy

# Train models with different hyperparameters
best_model = None
best_accuracy = 0
best_params = None

for lr, epoch, wordNgrams, minCount in product(param_grid["lr"], param_grid["epoch"], param_grid["wordNgrams"], param_grid["minCount"]):
    print(f"🔹 Training model with lr={lr}, epoch={epoch}, wordNgrams={wordNgrams}, minCount={minCount}")
    
    try:
        model = fasttext.train_supervised(
            input=train_file, 
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            minCount=minCount,
            loss="softmax",  # Use softmax loss for better classification
            verbose=0
        )
    except RuntimeError as e:
        print(f" Training failed for lr={lr}, epoch={epoch}, wordNgrams={wordNgrams}, minCount={minCount}. Error: {e}")
        continue

    val_accuracy = evaluate_model(model, valid_file)
    print(f" Validation Accuracy: {val_accuracy * 100:.2f}%\n")
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model
        best_params = (lr, epoch, wordNgrams, minCount)

# Save the best model
best_model_path = "fasttext_best_model_final.bin"
best_model.save_model(best_model_path)
print(f"🎯 Best Model Saved: {best_model_path}")
print(f"🏆 Best Hyperparameters: lr={best_params[0]}, epoch={best_params[1]}, wordNgrams={best_params[2]}, minCount={best_params[3]}")
print(f"🔥 Best Validation Accuracy: {best_accuracy * 100:.2f}%")

# Function to compute per-label accuracy
def evaluate_per_label(model, file_path):
    label_counts = defaultdict(int)   # Stores count of each label
    correct_counts = defaultdict(int) # Stores correctly predicted counts
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        true_label = parts[0]  # First word is the label (e.g., __label__english)
        text = " ".join(parts[1:])  # Remaining is the sentence
        
        # Predict label
        predicted_label, _ = model.predict(text, k=1)
        predicted_label = predicted_label[0]
        
        # Update counts
        label_counts[true_label] += 1
        if predicted_label == true_label:
            correct_counts[true_label] += 1
    
    # Calculate accuracy per label
    label_accuracy = {
        label: (correct_counts[label] / label_counts[label]) * 100 if label_counts[label] > 0 else 0
        for label in label_counts
    }
    
    return total_samples, label_accuracy

# Evaluate the best model
train_samples, train_acc = evaluate_per_label(best_model, train_file)
valid_samples, valid_acc = evaluate_per_label(best_model, valid_file)
test_samples, test_acc = evaluate_per_label(best_model, test_file)

# Create a summary report
labels = sorted(set(train_acc.keys()) | set(valid_acc.keys()) | set(test_acc.keys()))

report = []
for label in labels:
    report.append({
        "Label": label,
        "Train Accuracy (%)": train_acc.get(label, 0),
        "Validation Accuracy (%)": valid_acc.get(label, 0),
        "Test Accuracy (%)": test_acc.get(label, 0),
    })

# Convert to DataFrame
df = pd.DataFrame(report)

# Display the report
print("\n🔹 Per-Label Accuracy Report:")
print(df)

# Save to CSV
csv_path = "fasttext_per_label_accuracy.csv"
df.to_csv(csv_path, index=False)
print(f"\nReport saved to '{csv_path}'.")
