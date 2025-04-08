from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from joblib import parallel_backend
import numpy as np

# Function to load data
def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return sentences, labels

# File paths
train_sentence_file = "C:\\Users\\Lenovo\\Desktop\\train.txt"
train_label_file = "C:\\Users\\Lenovo\\Desktop\\train_labels.txt"
dev_sentence_file = "C:\\Users\\Lenovo\\Desktop\\valid.txt"
dev_label_file = "C:\\Users\\Lenovo\\Desktop\\valid_labels.txt"
test_sentence_file = "C:\\Users\\Lenovo\\Desktop\\test.txt"
test_label_file = "C:\\Users\\Lenovo\\Desktop\\test_labels.txt"

# Load data
train_sentences, train_labels = load_data(train_sentence_file, train_label_file)
dev_sentences, dev_labels = load_data(dev_sentence_file, dev_label_file)
test_sentences, test_labels = load_data(test_sentence_file, test_label_file)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)
y_dev = label_encoder.transform(dev_labels)
y_test = label_encoder.transform(test_labels)

# TF-IDF with feature limit & n-grams (use float32 to reduce memory)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
X_train = tfidf_vectorizer.fit_transform(train_sentences).astype(np.float32)
X_dev = tfidf_vectorizer.transform(dev_sentences).astype(np.float32)
X_test = tfidf_vectorizer.transform(test_sentences).astype(np.float32)

# GridSearchCV with KNN
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [1, 3, 5, 7]}  # reduced range

with parallel_backend('threading', n_jobs=-1):
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

# Best model
print("Best Parameters:", grid_search.best_params_)
best_knn = grid_search.best_estimator_

# Predictions
y_pred_dev = best_knn.predict(X_dev)
y_pred_test = best_knn.predict(X_test)

# Report
print("\n========= Classification Report =========")
print("\n[DEV SET]")
print(classification_report(y_dev, y_pred_dev, target_names=label_encoder.classes_))
print("\n[TEST SET]")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Averaging Metrics
def print_metrics(name, y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=name)
    print(f"{name.capitalize()} Avg -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

print("\n========= Averaging Metrics =========")
print("[DEV SET]")
print_metrics("macro", y_dev, y_pred_dev)
print_metrics("micro", y_dev, y_pred_dev)
print_metrics("weighted", y_dev, y_pred_dev)

print("\n[TEST SET]")
print_metrics("macro", y_test, y_pred_test)
print_metrics("micro", y_test, y_pred_test)
print_metrics("weighted", y_test, y_pred_test)
