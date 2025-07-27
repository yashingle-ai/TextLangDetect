"""Identify Language of Texts using Combined Char + Word TF-IDF with Logistic Regression"""

import argparse
import numpy as np
from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack


def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return sentences, labels


def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')


def dump_object_into_pickle_file(data_object, pickle_file):
    with open(pickle_file, 'wb') as f:
        dump(data_object, f)


def main():
    parser = argparse.ArgumentParser(description="Combined Char + Word TF-IDF Logistic Regression")
    parser.add_argument('--train_sentences', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--dev_sentences', type=str, required=True)
    parser.add_argument('--dev_labels', type=str, required=True)
    parser.add_argument('--test_sentences', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    args = parser.parse_args()

    # Load data
    train_sentences, train_labels = load_data(args.train_sentences, args.train_labels)
    dev_sentences, dev_labels = load_data(args.dev_sentences, args.dev_labels)
    test_sentences, test_labels = load_data(args.test_sentences, args.test_labels)

    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)

    # TF-IDF: Word-level (1,2) and Char-level (2,6)
    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))

    print("Fitting Word-level TF-IDF (1,2)...")
    X_train_word = tfidf_word.fit_transform(train_sentences)
    X_dev_word = tfidf_word.transform(dev_sentences)
    X_test_word = tfidf_word.transform(test_sentences)

    print("Fitting Char-level TF-IDF (2,6)...")
    X_train_char = tfidf_char.fit_transform(train_sentences)
    X_dev_char = tfidf_char.transform(dev_sentences)
    X_test_char = tfidf_char.transform(test_sentences)

    # Combine features
    X_train_combined = hstack([X_train_word, X_train_char])
    X_dev_combined = hstack([X_dev_word, X_dev_char])
    X_test_combined = hstack([X_test_word, X_test_char])

    # Train Logistic Regression with different C values
    C_values = [0.01, 0.1, 1, 10]
    penalty = 'l2'
    solver = 'saga'

    feature_info = "word(1-2)+char(2-6)"

    for C in C_values:
        model_name = f"logreg-C{C}-penalty{penalty}"
        print(f"\nTraining model: {model_name} with features: {feature_info}")

        clf = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                 multi_class='multinomial', max_iter=1000)
        clf.fit(X_train_combined, y_train)

        pred_dev = clf.predict(X_dev_combined)
        pred_test = clf.predict(X_test_combined)

        report_dev = classification_report(y_dev, pred_dev)
        report_test = classification_report(y_test, pred_test)

        write_text_to_file(report_dev, f'class-report-dev-{model_name}-{feature_info}.txt')
        write_text_to_file(report_test, f'class-report-test-{model_name}-{feature_info}.txt')
        dump_object_into_pickle_file(clf, f'{model_name}---{feature_info}.pkl')


if __name__ == "__main__":
    main()

