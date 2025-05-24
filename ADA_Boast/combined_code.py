import os
import joblib
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Load data
def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return sentences, labels

# File paths
train_sentence_file = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\train.txt"
train_label_file = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\train_labels.txt"
dev_sentence_file = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\valid.txt"
dev_label_file = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\valid_labels.txt"
test_sentence_file = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\test.txt"
test_label_file = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\test_labels.txt"

# Load data
train_sentences, train_labels = load_data(train_sentence_file, train_label_file)
dev_sentences, dev_labels = load_data(dev_sentence_file, dev_label_file)
test_sentences, test_labels = load_data(test_sentence_file, test_label_file)

# Create combined TF-IDF vectorizer for char and word n-grams
char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
tfidf_combined = FeatureUnion([("char", char_vectorizer), ("word", word_vectorizer)])

# Fit and transform
tfidf_combined.fit(train_sentences)
X_dev = tfidf_combined.transform(dev_sentences)
X_test = tfidf_combined.transform(test_sentences)
y_dev = np.array(dev_labels)
y_test = np.array(test_labels)

# Classifier file mapping
classifier_files = {
    "adaboost": "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\ada_boast\\models\\adaboost-char-word-est400-lr1.0-tfidf.pkl",
    "decision_tree": "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\decision_tree\\models\\dt-depthNone-crit-gini---combined-word(1,2)+char(2,6).pkl",
    "knn": "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\knn\\models\\knn-classifier---char-2-6-word-1-2.pkl",
    "logistic":"C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\logisitc regression\\models\\logreg-C10-penaltyl2---word(1-2)+char(2-6).pkl",
    "naive_bayes":"C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\mulinomail naive bayes\\models\\mnb-alpha-1.0---combined-char2-6-word1-2.pkl",
    "random_forest":"C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\random forest\\models\\rf-n100-depthNone-crit-gini---word(1-2)+char(2-6).pkl",
    "sgd": "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\SGD\\models\\sgd-best-hyperparams---char-2-6-word-1-2.pkl",
    "svc":"C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\SVM\\models\\svm-C-1-kernel-sigmoid-word-1-2.pkl"
}

# Load classifiers
classifiers = {}
for name, file in classifier_files.items():
    try:
        model = joblib.load(file)
        classifiers[name] = model
    except Exception as e:
        print(f"Could not load {name}: {e}")

# Ensure we have at least 5 models
if len(classifiers) < 5:
    raise ValueError("Not enough models loaded to create combinations of 5.")

# Generate all 10C5 combinations
combos = list(itertools.combinations(classifiers.items(), 5))

# Output directory
output_dir = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\Voting_Results"
os.makedirs(output_dir, exist_ok=True)

# Evaluate each 5-model combination
for combo in combos:
    names = [name for name, _ in combo]
    models = [(name, model) for name, model in combo]
    combo_name = "_".join(names)

    voting_type = 'soft' if all(hasattr(model, "predict_proba") for _, model in models) else 'hard'
    print(f"\nEvaluating combination: {combo_name} | Voting: {voting_type}")

    try:
        voting_clf = VotingClassifier(estimators=models, voting=voting_type)
        voting_clf.fit(X_dev, y_dev)

        # Dev predictions
        pred_dev = voting_clf.predict(X_dev)
        report_dev = classification_report(y_dev, pred_dev, zero_division=0)

        # Test predictions
        pred_test = voting_clf.predict(X_test)
        report_test = classification_report(y_test, pred_test, zero_division=0)

        # Save reports
        with open(os.path.join(output_dir, f"dev_report_{combo_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Classification Report - Dev Set ({voting_type} voting)\n\n")
            f.write(report_dev)

        with open(os.path.join(output_dir, f"test_report_{combo_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Classification Report - Test Set ({voting_type} voting)\n\n")
            f.write(report_test)

        # Save prediction CSVs
        pd.DataFrame({
            "sentence": dev_sentences,
            "true_label": dev_labels,
            "predicted_label": pred_dev
        }).to_csv(os.path.join(output_dir, f"dev_preds_{combo_name}.csv"), index=False)

        pd.DataFrame({
            "sentence": test_sentences,
            "true_label": test_labels,
            "predicted_label": pred_test
        }).to_csv(os.path.join(output_dir, f"test_preds_{combo_name}.csv"), index=False)

    except Exception as e:
        print(f"Skipping combo {combo_name} due to error: {e}")

print("\nAll 5-model combinations evaluated and saved.")
