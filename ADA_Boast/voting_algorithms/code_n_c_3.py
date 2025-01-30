# Import required libraries
import itertools
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Function to load data
def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    sentences = [line.strip() for line in sentences]
    labels = [line.strip() for line in labels]
    
    return sentences, labels

# File paths (update these paths accordingly)
train_sentence_file = 'train_sentences.txt'
train_label_file = 'train_labels.txt'
test_sentence_file = 'test_sentences.txt'
test_label_file = 'test_labels.txt'

# Load data
train_sentences, train_labels = load_data(train_sentence_file, train_label_file)
test_sentences, test_labels = load_data(test_sentence_file, test_label_file)

# Preprocessing
label_encoder = LabelEncoder()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using bigrams for richer feature extraction

X_train = tfidf_vectorizer.fit_transform(train_sentences)
y_train = label_encoder.fit_transform(train_labels)

X_test = tfidf_vectorizer.transform(test_sentences)
y_test = label_encoder.transform(test_labels)

# Define classifiers and hyperparameter grids
classifiers = {
    "SVM": (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "RandomForest": (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}),
    "LogisticRegression": (LogisticRegression(), {'C': [0.1, 1, 10]}),
    "SGD": (SGDClassifier(), {'alpha': [0.0001, 0.001, 0.01], 'loss': ['hinge', 'log']}),
    "AdaBoost": (AdaBoostClassifier(), {'n_estimators': [50, 100]}),
    "DecisionTree": (DecisionTreeClassifier(), {'max_depth': [5, 10, 20, None]}),
    "NaiveBayes": (MultinomialNB(), {'alpha': [0.1, 1, 10]}),
}

# Tune hyperparameters using GridSearchCV
best_models = {}
for name, (model, params) in classifiers.items():
    print(f"Training {name} with hyperparameter tuning...")
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best params for {name}: {grid_search.best_params_}\n")

# Generate all possible 3-classifier combinations
combinations = list(itertools.combinations(best_models.keys(), 3))

# Evaluate each ensemble
for combo in combinations:
    print(f"\nEvaluating combination: {combo}")
    
    selected_models = [(name, best_models[name]) for name in combo]
    ensemble = VotingClassifier(estimators=selected_models, voting='soft')
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Evaluation completed for all classifier combinations!")
