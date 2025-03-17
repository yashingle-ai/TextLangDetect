"""
Function: evaluate_per_label(model, file_path)
Purpose: Computes the accuracy of the FastText model for each label.

Parameters:
    - model: Trained FastText model
    - file_path: Path to the labeled dataset (train, validation, or test)

Returns:
    - total_samples: Total number of samples in the dataset
    - label_accuracy: Dictionary containing accuracy for each label

Usage Example:
    total_samples, accuracy_report = evaluate_per_label(model, "fasttext_test.txt")
"""


import argparse
import fasttext
import os
import pandas as pd
from collections import defaultdict

# Function to train FastText model
def train_model(train_file, model_path):
    model = fasttext.train_supervised(
        input=train_file, 
        epoch=25,        
        lr=0.5,          
        wordNgrams=2,    
        verbose=2,       
        minCount=1       
    )
    model.save_model(model_path)
    print(f"\u2705 Model training complete. Saved as '{model_path}'.")
    return model

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a FastText language identification model and evaluate it.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--model_path", type=str, default="fasttext_lang_identifier.bin", help="Path to save the trained model.")
    parser.add_argument("--output_csv", type=str, default="fasttext_per_label_accuracy.csv", help="Path to save the accuracy report.")
    
    args = parser.parse_args()
    
    # Train the model
    model = train_model(args.train_file, args.model_path)
    
    # Load trained model
    model = fasttext.load_model(args.model_path)
    
    # Evaluate on train, validation, and test sets
    train_samples, train_acc = evaluate_per_label(model, args.train_file)
    valid_samples, valid_acc = evaluate_per_label(model, args.valid_file)
    test_samples, test_acc = evaluate_per_label(model, args.test_file)
    
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
    print("\n\U0001F539 Per-Label Accuracy Report:")
    print(df)
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"\n\u2705 Report saved to '{args.output_csv}'.")
