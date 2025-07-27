"""Identify Language of Texts in English and Indian Languages using Decision Tree Classifier (Char-level TF-IDF)"""

import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import dump
import numpy as np

# Load sentences and corresponding labels from provided files
def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return sentences, labels

# Write text (e.g., classification report) to a file
def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write(text + '\n')

# Save a Python object (e.g., model) to disk using pickle
def dump_object_into_pickle_file(data_object, pickle_file):
    with open(pickle_file, 'wb') as pickle_dump:
        dump(data_object, pickle_dump)

def main():
    # Command-line arguments to accept file paths for train/dev/test
    parser = argparse.ArgumentParser(description="Decision Tree Classifier with Char-level TF-IDF")
    parser.add_argument('--train_sentences', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--dev_sentences', type=str, required=True)
    parser.add_argument('--dev_labels', type=str, required=True)
    parser.add_argument('--test_sentences', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    args = parser.parse_args()

    # Load the datasets
    train_sentences, train_labels = load_data(args.train_sentences, args.train_labels)
    dev_sentences, dev_labels = load_data(args.dev_sentences, args.dev_labels)
    test_sentences, test_labels = load_data(args.test_sentences, args.test_labels)

    # Configure TF-IDF vectorizer for char-level bigrams to 5-grams
    analyzer = 'char'
    ngram_range = (2, 6)
    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    print(f"Using TF-IDF: analyzer='{analyzer}', ngram_range={ngram_range}")

    # Convert text to TF-IDF feature vectors
    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    y_train = np.array(train_labels)

    X_dev = tfidf_vectorizer.transform(dev_sentences)
    y_dev = np.array(dev_labels)

    X_test = tfidf_vectorizer.transform(test_sentences)
    y_test = np.array(test_labels)

    # Parameter grid for Decision Tree
    param_grid = {
        'max_depth': [None],
        'criterion': ['gini', 'entropy']
    }

    tfidf_prefix = f"{analyzer}-{'-'.join(map(str, ngram_range))}"

    # Train a model for each parameter combination
    for max_depth in param_grid['max_depth']:
        for criterion in param_grid['criterion']:
            model_prefix = f"dt-depth{max_depth}-crit-{criterion}"
            print(f"Training model: {model_prefix}")

            # Train Decision Tree
            clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
            clf.fit(X_train, y_train)

            # Predict on dev and test data
            pred_dev = clf.predict(X_dev)
            pred_test = clf.predict(X_test)

            # Generate classification reports
            report_dev = classification_report(y_dev, pred_dev)
            report_test = classification_report(y_test, pred_test)

            # Save reports and model
            write_text_to_file(report_dev, f'class-report-dev-{model_prefix}-{tfidf_prefix}.txt')
            write_text_to_file(report_test, f'class-report-test-{model_prefix}-{tfidf_prefix}.txt')
            dump_object_into_pickle_file(clf, f'{model_prefix}---{tfidf_prefix}.pkl')

if __name__ == "__main__":
    main()


#for running the code the example command
# python char_tfidf_dt.py \
#   --train_sentences train.txt \
#   --train_labels train_labels.txt \
#   --dev_sentences dev.txt \
#   --dev_labels dev_labels.txt \
#   --test_sentences test.txt \
#   --test_labels test_labels.txt

