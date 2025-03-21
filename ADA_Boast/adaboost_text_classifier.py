"""
AdaBoost Text Classifier with Grid Search and Evaluation

This script trains an AdaBoost text classification model using TF-IDF features.
It performs hyperparameter tuning using GridSearchCV and evaluates the model on 
development and test datasets.

### Features:
- Loads sentence and label data from text files.
- Converts text into TF-IDF representations.
- Uses AdaBoost with a Decision Tree (max_depth=1) as the base estimator.
- Performs hyperparameter tuning with GridSearchCV to find the best model.
- Evaluates the model on test and development datasets.
- Reports precision, recall, and F1-score with macro, micro, and weighted averaging.

### How to Use:
Run the script with the required file paths as command-line arguments:

```bash
python adaboost_text_classifier.py --train_sent "path/to/train_sentences.txt" \
--train_label "path/to/train_labels.txt" \
--dev_sent "path/to/dev_sentences.txt" \
--dev_label "path/to/dev_labels.txt" \
--test_sent "path/to/test_sentences.txt" \
--test_label "path/to/test_labels.txt"
"""

import argparse
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

# Function to load data
def load_data(sentence_file, label_file):
    """Load sentences and labels from files."""
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    # Remove extra whitespace/newlines
    sentences = [line.strip() for line in sentences]
    labels = [line.strip() for line in labels]

    return sentences, labels

# Argument parser
parser = argparse.ArgumentParser(description="Train and evaluate an AdaBoost text classifier.")
parser.add_argument("--train_sent", required=True, help="Path to training sentences file")
parser.add_argument("--train_label", required=True, help="Path to training labels file")
parser.add_argument("--dev_sent", required=True, help="Path to development sentences file")
parser.add_argument("--dev_label", required=True, help="Path to development labels file")
parser.add_argument("--test_sent", required=True, help="Path to test sentences file")
parser.add_argument("--test_label", required=True, help="Path to test labels file")
args = parser.parse_args()

# Load data
train_sentences, train_labels = load_data(args.train_sent, args.train_label)
dev_sentences, dev_labels = load_data(args.dev_sent, args.dev_label)
test_sentences, test_labels = load_data(args.test_sent, args.test_label)

# Initialize Label Encoder and TF-IDF Vectorizer
label_encoder = LabelEncoder()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))

# Transform data
X_train = tfidf_vectorizer.fit_transform(train_sentences)
y_labels = label_encoder.fit_transform(train_labels)

X_test = tfidf_vectorizer.transform(test_sentences)
y_test = label_encoder.transform(test_labels)

X_dev = tfidf_vectorizer.transform(dev_sentences)
y_dev = label_encoder.transform(dev_labels)

# AdaBoost Classifier with Decision Tree Base Estimator
base_clf = DecisionTreeClassifier(max_depth=1)
adaboost_clf = AdaBoostClassifier(estimator=base_clf)

# Parameter Grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of weak learners
    'learning_rate': [0.01, 0.1, 1.0]  # Learning rate for boosting
}

# GridSearchCV
grid_search = GridSearchCV(adaboost_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_labels)

# Best Parameters and Evaluation
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluation on Test Data
y_pred_test = best_model.predict(X_test)
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Evaluation on Dev Data
y_pred_dev = best_model.predict(X_dev)
print("\nDev Set Classification Report:")
print(classification_report(y_dev, y_pred_dev, target_names=label_encoder.classes_))

# Precision, Recall, and F1-Score
print("Evaluation Metrics (Macro, Micro, Weighted):")

# Macro Averaging
dev_precision, dev_recall, dev_fscore, _ = precision_recall_fscore_support(y_dev, y_pred_dev, average="macro")
test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_test, y_pred_test, average="macro")
print("\nMacro Averaging:")
print("Dev Precision:", dev_precision, "Dev Recall:", dev_recall, "Dev F1-Score:", dev_fscore)
print("Test Precision:", test_precision, "Test Recall:", test_recall, "Test F1-Score:", test_fscore)

# Micro Averaging
dev_precision, dev_recall, dev_fscore, _ = precision_recall_fscore_support(y_dev, y_pred_dev, average="micro")
test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_test, y_pred_test, average="micro")
print("\nMicro Averaging:")
print("Dev Precision:", dev_precision, "Dev Recall:", dev_recall, "Dev F1-Score:", dev_fscore)
print("Test Precision:", test_precision, "Test Recall:", test_recall, "Test F1-Score:", test_fscore)

# Weighted Averaging
dev_precision, dev_recall, dev_fscore, _ = precision_recall_fscore_support(y_dev, y_pred_dev, average="weighted")
test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_test, y_pred_test, average="weighted")
print("\nWeighted Averaging:")
print("Dev Precision:", dev_precision, "Dev Recall:", dev_recall, "Dev F1-Score:", dev_fscore)
print("Test Precision:", test_precision, "Test Recall:", test_recall, "Test F1-Score:", test_fscore)
