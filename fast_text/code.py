import fasttext
import os
import pandas as pd
from collections import defaultdict

# Define file paths
train_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\label data\\fasttext_train.txt"
valid_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\label data\\fasttext_valid.txt"
test_file = "C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\label data\\fasttext_test.txt"

# Train FastText model
model = fasttext.train_supervised(
    input=train_file, 
    epoch=25,        
    lr=0.5,          
    wordNgrams=2,    
    verbose=2,       
    minCount=1       
)

# Save the trained model
model_path = "fasttext_lang_identifier.bin"
model.save_model(model_path)
print(f"✅ Model training complete. Saved as '{model_path}'.")

# Load the trained model
model = fasttext.load_model(model_path)

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

# Evaluate on train, validation, and test sets
train_samples, train_acc = evaluate_per_label(model, train_file)
valid_samples, valid_acc = evaluate_per_label(model, valid_file)
test_samples, test_acc = evaluate_per_label(model, test_file)

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
print(f"\n✅ Report saved to '{csv_path}'.")
