# Import required libraries
import itertools
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion

# Function to load data
def load_data(sentence_file, label_file):
    """Load sentences and labels from files."""
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()

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

# Initialize Label Encoder
label_encoder = LabelEncoder()

# Create TF-IDF vectorizers for word and character n-grams
word_tfidf = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')  # Word-level bigrams
char_tfidf = TfidfVectorizer(ngram_range=(2, 5), analyzer='char')  # Character-level 2 to 5-grams

# Combine both TF-IDF vectorizers
combined_tfidf = FeatureUnion([
    ('word_tfidf', word_tfidf),
    ('char_tfidf', char_tfidf)
])

# Transform data
X_train = combined_tfidf.fit_transform(train_sentences)
y_train = label_encoder.fit_transform(train_labels)

X_test = combined_tfidf.transform(test_sentences)
y_test = label_encoder.transform(test_labels)

X_dev = combined_tfidf.transform(dev_sentences)
y_dev = label_encoder.transform(dev_labels)

# Define classifiers with hyperparameter grids
param_grids = {
    "SVM": (SVC(probability=True), {'C': [10], 'kernel': ['rbf']}),  # Ensure probability=True
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [1]}),
    "RandomForest": (RandomForestClassifier(), {'n_estimators': [200], 'min_samples_split': [2], 'min_samples_leaf': [1], 'max_depth': [None]}),
    "LogisticRegression": (LogisticRegression(), {'C': [10]}),
    "SGD": (SGDClassifier(loss='log_loss'), {'penalty': ['l2'], 'alpha': [0.0001], 'learning_rate': ['optimal'], 'eta0': [0.03]}),
    "AdaBoost": (AdaBoostClassifier(), {'n_estimators': [400]}),
    "DecisionTree": (DecisionTreeClassifier(), {'max_depth': [5, 10, 20, None]}),
    "NaiveBayes": (MultinomialNB(), {'alpha': [0.001, 0.01, 0.1, 1]})
}

# Perform GridSearchCV for each classifier
best_classifiers = {}
for name, (clf, params) in param_grids.items():
    print(f"Hyperparameter tuning for {name}...")
    grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_classifiers[name] = grid_search.best_estimator_
    print(f"Best Parameters for {name}: {grid_search.best_params_}\n")

# Generate all possible 5-classifier combinations
combinations = list(itertools.combinations(best_classifiers.keys(), 5))

# Evaluate each ensemble combination
for combo in combinations:
    print(f"\nEvaluating combination: {combo}")

    selected_models = [(name, best_classifiers[name]) for name in combo]
    
    # Check if all classifiers in the combination support predict_proba()
    if all(hasattr(best_classifiers[name], "predict_proba") for name in combo):
        ensemble = VotingClassifier(estimators=selected_models, voting='soft')  # Soft Voting needs predict_proba()
    else:
        ensemble = VotingClassifier(estimators=selected_models, voting='hard')  # Hard Voting for non-probabilistic models

    ensemble.fit(X_train, y_train)

    # Predict on test set
    y_pred_test = ensemble.predict(X_test)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    # Predict on dev set
    y_pred_dev = ensemble.predict(X_dev)
    print("\nDev Set Classification Report:")
    print(classification_report(y_dev, y_pred_dev, target_names=label_encoder.classes_))

print("Evaluation completed for all classifier combinations!")
