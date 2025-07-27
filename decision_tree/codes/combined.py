"""Language Identification using Word + Char TF-IDF and Decision Tree Classifier"""
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
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
    parser = argparse.ArgumentParser(description="Decision Tree with Combined Word + Char TF-IDF")
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

    # TF-IDF Vectors
    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))

    print("Fitting word & char TF-IDF vectorizers...")
    X_train_word = tfidf_word.fit_transform(train_sentences)
    X_train_char = tfidf_char.fit_transform(train_sentences)
    X_train = hstack([X_train_word, X_train_char])
    y_train = np.array(train_labels)

    X_dev_word = tfidf_word.transform(dev_sentences)
    X_dev_char = tfidf_char.transform(dev_sentences)
    X_dev = hstack([X_dev_word, X_dev_char])
    y_dev = np.array(dev_labels)

    X_test_word = tfidf_word.transform(test_sentences)
    X_test_char = tfidf_char.transform(test_sentences)
    X_test = hstack([X_test_word, X_test_char])
    y_test = np.array(test_labels)

    # Params for Decision Tree
    param_grid = {
        'max_depth': [10, 20,None],
        'criterion': ['gini', 'entropy']
    }

    tfidf_prefix = "combined-word(1,2)+char(2,6)"

    for max_depth in param_grid['max_depth']:
        for criterion in param_grid['criterion']:
            model_prefix = f"dt-depth{max_depth}-crit-{criterion}"
            print(f"Training model: {model_prefix}")

            clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
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
