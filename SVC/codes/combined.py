"""
Identify Language of Texts in English and Indian Languages using SVM 
with Combined Word and Char TF-IDF features.
"""

import os
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from pickle import dump


def load_data(sentence_file, label_file):
    """
    Load text data and labels from two separate files.
    Each line in the file corresponds to one sample.
    """
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return sentences, labels


def write_text_to_file(text, file_path):
    """
    Save a string of text (e.g., classification report) to a file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')


def dump_object_into_pickle_file(data_object, pickle_file):
    """
    Pickle and save a Python object (model, vectorizer) to file.
    """
    with open(pickle_file, 'wb') as f:
        dump(data_object, f)


def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(" Loading datasets...")
    train_sentences, train_labels = load_data(args.train_sentences, args.train_labels)
    dev_sentences, dev_labels = load_data(args.dev_sentences, args.dev_labels)
    test_sentences, test_labels = load_data(args.test_sentences, args.test_labels)

    print(" Vectorizing text with TF-IDF (word and character n-grams)...")
    # Word-level TF-IDF: unigrams and bigrams
    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    # Character-level TF-IDF: 2- to 6-grams
    tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))

    # Combine both vectorizers using FeatureUnion
    combined_vectorizer = FeatureUnion([
        ("word", tfidf_word),
        ("char", tfidf_char)
    ])

    # Fit and transform training data, transform dev/test data
    X_train = combined_vectorizer.fit_transform(train_sentences)
    X_dev = combined_vectorizer.transform(dev_sentences)
    X_test = combined_vectorizer.transform(test_sentences)

    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)

    print(" Training SVM Classifier")
    model_prefix = f"svm-C{args.C}-kernel-{args.kernel}"
    tfidf_prefix = "word(1,2)_char(2,6)"

    # Train the SVM model
    clf = SVC(C=args.C, kernel=args.kernel)
    clf.fit(X_train, y_train)

    print("🔍 Generating predictions")
    pred_dev = clf.predict(X_dev)
    pred_test = clf.predict(X_test)

    print(" Saving classification reports")
    report_dev = classification_report(y_dev, pred_dev)
    report_test = classification_report(y_test, pred_test)

    # Save reports to output directory
    write_text_to_file(report_dev, os.path.join(args.output_dir, f'class-report-dev-{model_prefix}-{tfidf_prefix}.txt'))
    write_text_to_file(report_test, os.path.join(args.output_dir, f'class-report-test-{model_prefix}-{tfidf_prefix}.txt'))

    print(" Saving model and vectorizer")
    model_path = os.path.join(args.output_dir, f'{model_prefix}---{tfidf_prefix}.pkl')
    vectorizer_path = os.path.join(args.output_dir, 'combined_tfidf_vectorizer.pkl')

    # Save both model and vectorizer
    dump_object_into_pickle_file(clf, model_path)
    dump_object_into_pickle_file(combined_vectorizer, vectorizer_path)

    print(f"\n All done!\n→ Model saved to:{model_path}\n→ Vectorizer saved to: {vectorizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM for Language Identification using Word+Char TF-IDF")

    # Required file paths
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save reports and models')
    parser.add_argument('--train_sentences', type=str, required=True, help='Path to training sentences file')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels file')
    parser.add_argument('--dev_sentences', type=str, required=True, help='Path to development sentences file')
    parser.add_argument('--dev_labels', type=str, required=True, help='Path to development labels file')
    parser.add_argument('--test_sentences', type=str, required=True, help='Path to test sentences file')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to test labels file')

    # Optional SVM parameters
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter for SVM')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel type (e.g., linear, rbf)')

    args = parser.parse_args()
    main(args)
