"""Identify Language of Texts in English and Indian Languages using Multinomial Naive Bayes with Char + Word TF-IDF"""
import argparse
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
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
    parser = argparse.ArgumentParser(description="Multinomial Naive Bayes with Combined Char + Word TF-IDF")
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

    # TF-IDF Combined Char + Word Setup
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    combined_vectorizer = FeatureUnion([
        ('char_tfidf', char_vectorizer),
        ('word_tfidf', word_vectorizer)
    ])

    print("TF-IDF Config: Char ngram=(2,6) + Word ngram=(1,2)")

    X_train = combined_vectorizer.fit_transform(train_sentences)
    y_train = np.array(train_labels)

    X_dev = combined_vectorizer.transform(dev_sentences)
    y_dev = np.array(dev_labels)

    X_test = combined_vectorizer.transform(test_sentences)
    y_test = np.array(test_labels)

    alpha_values = [1.0, 0.01, 0.001]
    tfidf_prefix = "combined-char2-6-word1-2"

    for alpha in alpha_values:
        model_prefix = f"mnb-alpha-{alpha}"
        print(f"\nTraining model: {model_prefix}")

        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train, y_train)

        pred_dev = clf.predict(X_dev)
        pred_test = clf.predict(X_test)

        report_dev = classification_report(y_dev, pred_dev)
        report_test = classification_report(y_test, pred_test)

        write_text_to_file(report_dev, f'class-report-dev-{model_prefix}-{tfidf_prefix}.txt')
        write_text_to_file(report_test, f'class-report-test-{model_prefix}-{tfidf_prefix}.txt')
        dump_object_into_pickle_file(clf, f'{model_prefix}---{tfidf_prefix}.pkl')

        # Save vectorizer once
        if alpha == alpha_values[0]:
            dump_object_into_pickle_file(combined_vectorizer, f'word_char_combined_vectorizer---{tfidf_prefix}.pkl')


if __name__ == "__main__":
    main()
