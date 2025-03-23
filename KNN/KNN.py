"""-Nearest Neighbors (KNN) classifier for text classification using TF-IDF (Term Frequency-Inverse Document Frequency) features.
It uses GridSearchCV to find the best K value (n_neighbors) and evaluates the model using precision, recall, F1-score, and classification reports on test and development datasets."""

import os
import argparse
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support


#  Argument Parser Setup

def get_arguments():
    """Parse command-line arguments for dataset file paths."""
    parser = argparse.ArgumentParser(description="KNN Text Classification using TF-IDF and GridSearchCV.")
    
    parser.add_argument("--train_sent", type=str, required=True, help="Path to training sentence file")
    parser.add_argument("--train_label", type=str, required=True, help="Path to training label file")
    parser.add_argument("--test_sent", type=str, required=True, help="Path to test sentence file")
    parser.add_argument("--test_label", type=str, required=True, help="Path to test label file")
    parser.add_argument("--dev_sent", type=str, required=True, help="Path to development sentence file")
    parser.add_argument("--dev_label", type=str, required=True, help="Path to development label file")

    return parser.parse_args()



#  Function to Load Data

def load_data(sentence_file, label_file):
    """Load sentences and labels from text files."""
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    return sentences, labels



#  Main Function

def main():
    args = get_arguments()

    # Load dataset
    train_sentences, train_labels = load_data(args.train_sent, args.train_label)
    test_sentences, test_labels = load_data(args.test_sent, args.test_label)
    dev_sentences, dev_labels = load_data(args.dev_sent, args.dev_label)

    # Initialize TF-IDF Vectorizer and Label Encoder
    label_encoder = LabelEncoder()
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))

    # Transform text data into numerical vectors
    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    y_train = label_encoder.fit_transform(train_labels)
    
    X_test = tfidf_vectorizer.transform(test_sentences)
    y_test = label_encoder.transform(test_labels)

    X_dev = tfidf_vectorizer.transform(dev_sentences)
    y_dev = label_encoder.transform(dev_labels)

    
    #  KNN Classifier with Grid Search
    
    knn_classifier = KNeighborsClassifier()
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]}

    # Perform GridSearchCV to find the best 'k'
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\n Best Parameters Found:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    
    #  Evaluate Model on Test and Dev Data
    
    print("\n Test Set Classification Report:")
    y_pred_test = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    print("\n Development Set Classification Report:")
    y_pred_dev = best_model.predict(X_dev)
    print(classification_report(y_dev, y_pred_dev, target_names=label_encoder.classes_))

    
    #  Compute Precision, Recall, F1-score
    
    def compute_metrics(y_true, y_pred, dataset_name):
        """Compute and print precision, recall, and F1-score for a given dataset."""
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        print(f"\n {dataset_name} Set Metrics (Macro Averaging):")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
        print(f"\n {dataset_name} Set Metrics (Micro Averaging):")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        print(f"\n {dataset_name} Set Metrics (Weighted Averaging):")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")

    # Compute metrics for Test and Dev datasets
    compute_metrics(y_test, y_pred_test, "Test")
    compute_metrics(y_dev, y_pred_dev, "Dev")

    print("\n Model evaluation complete!")


#  Entry Point

if __name__ == "__main__":
    main()

#example of use 
#python knn_text_classification.py --train_sent path_to_train_sentences --train_label path_to_train_labels --test_sent path_to_test_sentences --test_label path_to_test_labels --dev_sent path_to_dev_sentences --dev_label path_to_dev_labels
#python knn_text_classification.py --train_sent "C:/Users/yashi/OneDrive/Desktop/ML project/text_suffling/shuffle_train_sent_all.txt" --train_label "C:/Users/yashi/OneDrive/Desktop/ML project/text_suffling/shuffle_train_label_all.txt" --test_sent "C:/Users/yashi/OneDrive/Desktop/ML project/text_suffling/shuffle_test_sent_all.txt" --test_label "C:/Users/yashi/OneDrive/Desktop/ML project/text_suffling/shuffle_test_label_all.txt" --dev_sent "C:/Users/yashi/OneDrive/Desktop/ML project/text_suffling/shuffle_dev_sent_all.txt" --dev_label "C:/Users/yashi/OneDrive/Desktop/ML project/text_suffling/shuffle_dev_label_all.txt"
