import fasttext
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Load FastText model
model = fasttext.load_model("C:\\Users\\Lenovo\\OneDrive\\Desktop\\votting algo\\fasttext_best_model_final_2.bin")

# Load test data
with open("C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\shuffle_test_train_data\\shuffled_test_sentences.txt", "r", encoding="utf-8") as f:
    test_sentences = [line.strip() for line in f if line.strip()]

with open("C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\shuffle_test_train_data\\shuffed_test_labels.txt" , "r", encoding="utf-8") as f:
    true_labels = [line.strip() for line in f if line.strip()]

# Extract coarse labels (e.g., mni_Beng → mni, snd_Deva → snd)
def extract_coarse_label(label: str) -> str:
    return label.replace("__label__", "").split("_")[0]

# Convert to coarse labels
y_true_coarse = [extract_coarse_label(label) for label in true_labels]
y_predicted_raw = [model.predict(sentence)[0][0] for sentence in test_sentences]
y_pred_coarse = [extract_coarse_label(label) for label in y_predicted_raw]

# Collect all unique coarse labels from both true and predicted
all_coarse_labels = sorted(set(y_true_coarse + y_pred_coarse))

# Print classification report
print("\n--- Classification Report (Coarse Labels) ---")
print(classification_report(y_true_coarse, y_pred_coarse, labels=all_coarse_labels))

# Individual metrics
precision = precision_score(y_true_coarse, y_pred_coarse, average='weighted', zero_division=0)
recall = recall_score(y_true_coarse, y_pred_coarse, average='weighted', zero_division=0)
f1 = f1_score(y_true_coarse, y_pred_coarse, average='weighted', zero_division=0)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
