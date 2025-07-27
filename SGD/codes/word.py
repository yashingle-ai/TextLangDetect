"""Identify Language of Texts using SGDClassifier with best parameters"""
import argparse
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from pickle import dump


def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return sentences, labels


def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')


def dump_object_into_pickle_file(data_object, pickle_file):
    with open(pickle_file, 'wb') as f:
        dump(data_object, f)


def main():
    parser = argparse.ArgumentParser(description="SGDClassifier with Best Hyperparameters")
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

    # TF-IDF Char N-gram Setup
    analyzer = 'word'
    ngram_range = (1,2)
    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    print(f"TF-IDF Config: analyzer={analyzer}, ngram_range={ngram_range}")

    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    y_train = np.array(train_labels)

    X_dev = tfidf_vectorizer.transform(dev_sentences)
    y_dev = np.array(dev_labels)

    X_test = tfidf_vectorizer.transform(test_sentences)
    y_test = np.array(test_labels)

    tfidf_prefix = f"{analyzer}-{'-'.join(map(str, ngram_range))}"
    model_prefix = "sgd-best-hyperparams"

    # Best Parameters as per your tuning
    clf = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        learning_rate='optimal',
        alpha=0.0001,
        eta0=0.03162277660168379,
        l1_ratio=1.0,  # Used only for 'elasticnet', but fine to include
        random_state=42,
        max_iter=1000,
        tol=1e-3
    )

    print(f"\nTraining model: {model_prefix}")
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
