"""Identify Language of Texts in English and Indian Languages"""
import argparse
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import dump
import numpy as np


# Load sentence-label data
def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return sentences, labels


def write_text_to_file(text, file_path):
    """Write text to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write(text + '\n')


def dump_object_into_pickle_file(data_object, pickle_file):
    """Dump an object into a pickle file."""
    with open(pickle_file, 'wb') as pickle_dump:
        dump(data_object, pickle_dump)


def main():
    parser = argparse.ArgumentParser(description="SVM Classifier with GridSearchCV")
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

    analyzer = 'word'
    ngram_range = (1, 1)
    # Label Encoding and TF-IDF
    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    print("TF-IDF word ngram_range = (1,1)")

    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    y_train = np.array(train_labels)

    X_dev = tfidf_vectorizer.transform(dev_sentences)
    y_dev = np.array(dev_labels)

    X_test = tfidf_vectorizer.transform(test_sentences)
    y_test = np.array(test_labels)
    # Define SVM & Grid Search parameters
    param_grid = {
        'C': [0, 0.001, 0.02, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # 'C': [1.],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
    tfidf_prefix = analyzer + '-' + '-'.join(map(str, ngram_range))
    dump_object_into_pickle_file(tfidf_vectorizer, tfidf_prefix + '-vectorizer.pkl')
    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
            model_prefix = '-'.join(['svm', 'C', str(C), 'kernel', kernel])
            svm_classifier = SVC(C=C, kernel=kernel)
            svm_classifier.fit(X_train, y_train)
            pred_dev = svm_classifier.predict(X_dev)
            pred_test = svm_classifier.predict(X_test)
            report_dev = classification_report(y_dev, pred_dev)
            report_test = classification_report(y_test, pred_test)
            write_text_to_file(report_dev, 'class-report-dev-' + model_prefix + '-' + tfidf_prefix + '.txt')
            write_text_to_file(report_test, 'class-report-test-' + model_prefix + '-' + tfidf_prefix + '.txt')
            dump_object_into_pickle_file(svm_classifier, model_prefix + '-' + tfidf_prefix + '.pkl')


if __name__ == "__main__":
    main()
