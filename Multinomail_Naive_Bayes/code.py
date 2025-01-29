# Import required libraries 
from sklearn.naive_bayes import MultinomialNB
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

# File paths
train_sentence_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_train_sent_all.txt'
train_label_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_train_label_all.txt'
dev_sentence_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_dev_sent_all.txt'
dev_label_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_dev_label_all.txt'
test_sentence_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_test_sent_all.txt'
test_label_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_test_label_all.txt'

# Load data
train_sentences, train_labels = load_data(train_sentence_file, train_label_file)
dev_sentences, dev_labels = load_data(dev_sentence_file, dev_label_file)
test_sentences, test_labels = load_data(test_sentence_file, test_label_file)

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

# Multinomial Naive Bayes Classifier
mnb_clf = MultinomialNB()

# Parameter Grid for GridSearchCV
param_grid = {
    'alpha': [0.01,0.001,1,2,3,4,5],  # Smoothing parameter for Naive Bayes
}

# GridSearchCV
grid_search = GridSearchCV(mnb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
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
