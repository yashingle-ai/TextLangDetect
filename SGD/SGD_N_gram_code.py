from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to load data
def load_data(sentence_file, label_file):
    """Load sentences and labels from files."""
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
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

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)
y_dev = label_encoder.transform(dev_labels)
y_test = label_encoder.transform(test_labels)

# Feature Extraction - TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
X_train = tfidf_vectorizer.fit_transform(train_sentences)
X_dev = tfidf_vectorizer.transform(dev_sentences)
X_test = tfidf_vectorizer.transform(test_sentences)

# Initialize SGDClassifier
sgd_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, penalty='elasticnet')

# Hyperparameter Grid for RandomizedSearchCV
param_grid = {
    'alpha': np.logspace(-4, 0, 10),  # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': np.linspace(0, 1, 5),  # Used for elasticnet
    'learning_rate': ['constant', 'optimal', 'adaptive'],
    'eta0': np.logspace(-3, 0, 5),  # Initial learning rate for 'constant' schedule
}

# RandomizedSearchCV for best parameters
grid_search = RandomizedSearchCV(sgd_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, n_iter=20, random_state=42)
grid_search.fit(X_train, y_train)

# Best model selection
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate on Test Data
y_pred_test = best_model.predict(X_test)
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Evaluate on Dev Data
y_pred_dev = best_model.predict(X_dev)
print("\nDev Set Classification Report:")
print(classification_report(y_dev, y_pred_dev, target_names=label_encoder.classes_))

# Precision, Recall, F1-Score Calculation
for avg in ['macro', 'micro', 'weighted']:
    dev_precision, dev_recall, dev_fscore, _ = precision_recall_fscore_support(y_dev, y_pred_dev, average=avg)
    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_test, y_pred_test, average=avg)
    print(f"\n{avg.capitalize()} Averaging:")
    print(f"Dev - Precision: {dev_precision:.4f}, Recall: {dev_recall:.4f}, F1-Score: {dev_fscore:.4f}")
    print(f"Test - Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_fscore:.4f}")
