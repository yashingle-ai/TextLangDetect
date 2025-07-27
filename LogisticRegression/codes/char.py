
"""Identify Language of Texts in English and Indian Languages"""

import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import dump
import numpy as np


def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return sentences, labels


def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write(text + '\n')


def dump_object_into_pickle_file(data_object, pickle_file):
    with open(pickle_file, 'wb') as pickle_dump:
        dump(data_object, pickle_dump)


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression Classifier with Parameter Loop")
    parser.add_argument('--train_sentences', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--dev_sentences', type=str, required=True)
    parser.add_argument('--dev_labels', type=str, required=True)
    parser.add_argument('--test_sentences', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    args = parser.parse_args()

    # Load datasets
    train_sentences, train_labels = load_data(args.train_sentences, args.train_labels)
    dev_sentences, dev_labels = load_data(args.dev_sentences, args.dev_labels)
    test_sentences, test_labels = load_data(args.test_sentences, args.test_labels)

    # TF-IDF configuration
    analyzer = 'char'
    ngram_range = (2, 6)
    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    print(f"Using TF-IDF: analyzer='{analyzer}', ngram_range={ngram_range}")

    # Transform data
    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    X_dev = tfidf_vectorizer.transform(dev_sentences)
    X_test = tfidf_vectorizer.transform(test_sentences)

    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)

    # Hyperparameters
    C_values = [0.01, 0.1, 1, 10]
    penalty = 'l2'
    solver = 'saga'

    tfidf_prefix = f"{analyzer}-{'-'.join(map(str, ngram_range))}"

    for C in C_values:
        model_prefix = f"logreg-C{C}-penalty{penalty}"
        print(f"\nTraining model: {model_prefix}")

        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, multi_class='multinomial', max_iter=1000)
        clf.fit(X_train, y_train)

        pred_dev = clf.predict(X_dev)
        pred_test = clf.predict(X_test)

        report_dev = classification_report(y_dev, pred_dev)
        report_test = classification_report(y_test, pred_test)

        write_text_to_file(report_dev, f'class-report-dev-{model_prefix}-{tfidf_prefix}.txt')
        write_text_to_file(report_test, f'class-report-test-{model_prefix}-{tfidf_prefix}.txt')
        dump_object_into_pickle_file(clf, f'{model_prefix}---{tfidf_prefix}.pkl')


if __name__ == "__main__":
    main()
