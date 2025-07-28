import os
import argparse
import joblib
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate all 5-model combinations using VotingClassifier.")
    parser.add_argument('--train_sentences', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--dev_sentences', type=str, required=True)
    parser.add_argument('--dev_labels', type=str, required=True)
    parser.add_argument('--test_sentences', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--classifier_dir', type=str, required=True)
    return parser.parse_args()

# Load data
def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return sentences, labels

def main():
    args = parse_arguments()

    # Load data
    train_sentences, train_labels = load_data(args.train_sentences, args.train_labels)
    dev_sentences, dev_labels = load_data(args.dev_sentences, args.dev_labels)
    test_sentences, test_labels = load_data(args.test_sentences, args.test_labels)

    # Create combined TF-IDF vectorizer for char and word n-grams
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    tfidf_combined = FeatureUnion([("char", char_vectorizer), ("word", word_vectorizer)])

    # Fit and transform
    tfidf_combined.fit(train_sentences)
    X_dev = tfidf_combined.transform(dev_sentences)
    X_test = tfidf_combined.transform(test_sentences)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)

    # Classifier file mapping (assumes all files are in classifier_dir)
    classifier_files = {
        "adaboost": os.path.join(args.classifier_dir, "adaboost.pkl"),
        "decision_tree": os.path.join(args.classifier_dir, "decision_tree.pkl"),
        "knn": os.path.join(args.classifier_dir, "knn.pkl"),
        "logistic": os.path.join(args.classifier_dir, "logistic.pkl"),
        "naive_bayes": os.path.join(args.classifier_dir, "naive_bayes.pkl"),
        "random_forest": os.path.join(args.classifier_dir, "random_forest.pkl"),
        "sgd": os.path.join(args.classifier_dir, "sgd.pkl"),
        "svc": os.path.join(args.classifier_dir, "svc.pkl")
    }

    # Load classifiers
    classifiers = {}
    for name, file in classifier_files.items():
        try:
            model = joblib.load(file)
            classifiers[name] = model
        except Exception as e:
            print(f"Could not load {name}: {e}")

    if len(classifiers) < 5:
        raise ValueError("Not enough models loaded to create combinations of 5.")

    combos = list(itertools.combinations(classifiers.items(), 5))

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each 5-model combination
    for combo in combos:
        names = [name for name, _ in combo]
        models = [(name, model) for name, model in combo]
        combo_name = "_".join(names)

        voting_type = 'soft' if all(hasattr(model, "predict_proba") for _, model in models) else 'hard'
        print(f"\nEvaluating combination: {combo_name} | Voting: {voting_type}")

        try:
            voting_clf = VotingClassifier(estimators=models, voting=voting_type)
            voting_clf.fit(X_dev, y_dev)

            pred_dev = voting_clf.predict(X_dev)
            report_dev = classification_report(y_dev, pred_dev, zero_division=0)

            pred_test = voting_clf.predict(X_test)
            report_test = classification_report(y_test, pred_test, zero_division=0)

            # Save reports
            with open(os.path.join(args.output_dir, f"dev_report_{combo_name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Classification Report - Dev Set ({voting_type} voting)\n\n")
                f.write(report_dev)

            with open(os.path.join(args.output_dir, f"test_report_{combo_name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Classification Report - Test Set ({voting_type} voting)\n\n")
                f.write(report_test)

            # Save predictions
            pd.DataFrame({
                "sentence": dev_sentences,
                "true_label": dev_labels,
                "predicted_label": pred_dev
            }).to_csv(os.path.join(args.output_dir, f"dev_preds_{combo_name}.csv"), index=False)

            pd.DataFrame({
                "sentence": test_sentences,
                "true_label": test_labels,
                "predicted_label": pred_test
            }).to_csv(os.path.join(args.output_dir, f"test_preds_{combo_name}.csv"), index=False)

        except Exception as e:
            print(f"Skipping combo {combo_name} due to error: {e}")

    print("\nAll 5-model combinations evaluated and saved.")

if __name__ == "__main__":
    main()
