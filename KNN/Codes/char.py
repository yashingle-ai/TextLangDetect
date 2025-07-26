import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from pickle import dump
import os
import argparse

#functions 

def load_data(sentence_file, label_file):
    """Load sentences and labels from corresponding files."""
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return sentences, labels

def write_text_to_file(text, file_path):
    """Save a text string to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')

def dump_object_into_pickle_file(data_object, pickle_file):
    """Save a Python object to a pickle file."""
    with open(pickle_file, 'wb') as f:
        dump(data_object, f)


def main(args):
    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load train, dev, and test data
    train_sentences, train_labels = load_data(args.train_sentences, args.train_labels)
    dev_sentences, dev_labels = load_data(args.dev_sentences, args.dev_labels)
    test_sentences, test_labels = load_data(args.test_sentences, args.test_labels)

    # We're using character-level TF-IDF with n-grams from size 2 to 6
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))

    print("👉 Fitting TF-IDF vectorizer on training data...")
    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    y_train = np.array(train_labels)

    # Transform dev and test data using the same vectorizer
    X_dev = tfidf_vectorizer.transform(dev_sentences)
    y_dev = np.array(dev_labels)

    X_test = tfidf_vectorizer.transform(test_sentences)
    y_test = np.array(test_labels)

    # File/model naming convention
    tfidf_prefix = "char-2-6"
    model_prefix = "knn-classifier"

    # Initialize and train the KNN classifier
    print(f" Training KNN model ({model_prefix}) with cosine distance...")
    clf = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate model on dev and test sets
    pred_dev = clf.predict(X_dev)
    pred_test = clf.predict(X_test)

    report_dev = classification_report(y_dev, pred_dev)
    report_test = classification_report(y_test, pred_test)

    # Save classification reports
    write_text_to_file(report_dev, os.path.join(args.output_dir, f'class-report-dev-{model_prefix}-{tfidf_prefix}.txt'))
    write_text_to_file(report_test, os.path.join(args.output_dir, f'class-report-test-{model_prefix}-{tfidf_prefix}.txt'))

    # Save the trained model and vectorizer
    dump_object_into_pickle_file(clf, os.path.join(args.output_dir, f'{model_prefix}---{tfidf_prefix}.pkl'))
    dump_object_into_pickle_file(tfidf_vectorizer, os.path.join(args.output_dir, f'vectorizer---{tfidf_prefix}.pkl'))

    print(f"\n All done! Model, vectorizer, and reports saved in: {args.output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a KNN classifier on text using character-level TF-IDF.")

    # File path arguments
    parser.add_argument('--train_sentences', type=str, required=True, help="Path to the training sentences file.")
    parser.add_argument('--train_labels', type=str, required=True, help="Path to the training labels file.")
    parser.add_argument('--dev_sentences', type=str, required=True, help="Path to the dev sentences file.")
    parser.add_argument('--dev_labels', type=str, required=True, help="Path to the dev labels file.")
    parser.add_argument('--test_sentences', type=str, required=True, help="Path to the test sentences file.")
    parser.add_argument('--test_labels', type=str, required=True, help="Path to the test labels file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save model, vectorizer, and reports.")

    args = parser.parse_args()
    main(args)
