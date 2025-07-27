#here all the models can be download from hugging face 


import os
import joblib
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
import traceback

# Load data
def load_data(sentence_file, label_file):
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return sentences, labels

# Normalize mni labels
def normalize_mni(label):
    return 'mni' if label in ['mni_Mtei', 'mni_Beng', 'mni'] else label

# File paths
test_sentence_file  = r"D:\\Test-Data\\Test-Data\\sentences.txt"
test_label_file     = r"D:\\Test-Data\\Test-Data\\labels.txt"

# Load test data
test_sentences, test_labels = load_data(test_sentence_file, test_label_file)
test_labels = [normalize_mni(lab) for lab in test_labels]
y_test = np.array(test_labels)

# Define classifier and vectorizer paths
classifier_vectorizer_pairs = {
    "decision_tree": (
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\decision_tree\\new_report_and_other\\dt-depthNone-crit-gini---combined-word(1,2)+char(2,6).pkl",
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\decision_tree\\new_report_and_other\\vectorizer---dt-depthNone-crit-gini---combined-word(1,2)+char(2,6).pkl"
    ),
    "knn": (
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\knn\\new_report_and_other\\knn-classifier---char-2-6-word-1-2.pkl",
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\knn\\new_report_and_other\\vectorizer---char-2-6-word-1-2.pkl"
    ),
    "logistic": (
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\logisitc regression\\new_report_and_other\\logreg-C1-penaltyl2---word(1-2)+char(2-6).pkl",
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\logisitc regression\\new_report_and_other\\logreg-C1-penaltyl2---vectorizer---word(1-2)+char(2-6).pkl"
    ),
    "naive_bayes": (
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\mulinomail naive bayes\\new_report_and_other\\mnb-alpha-1.0---combined-char2-6-word1-2.pkl",
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\mulinomail naive bayes\\new_report_and_other\\word_char_combined_vectorizer---combined-char2-6-word1-2.pkl"
    ),
    "random_forest": (
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\random forest\\new_report_and_other\\rf-n100-depthNone-crit-gini---word(1-2)+char(2-6).pkl",
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\random forest\\new_report_and_other\\vectorizer---word(1-2)+char(2-6).pkl"
    ),
    "sgd": (
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\SGD\\new_report_and_other\\sgd_model.pkl",
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\SGD\\new_report_and_other\\tfidf_vectorizer.pkl"
    ),
    "svc": (
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\SVM\\new_report_and_other\\svm-C1-kernel-linear---word(1,2)+char(2,6).pkl",
        "C:\\Users\\Lenovo\\OneDrive\\Desktop\\models and reports\\SVM\\new_report_and_other\\word_char_combined_tfidf_vectorizer.pkl"
    )
}

# Load classifiers and vectorizers
pipelines = {}
for name, (clf_path, vec_path) in classifier_vectorizer_pairs.items():
    try:
        clf = joblib.load(clf_path)
        vec = joblib.load(vec_path)
        pipelines[name] = (clf, vec)
    except Exception as e:
        print(f"Error loading {name}:\n{traceback.format_exc()}")

# Ensure at least 5 models are loaded
if len(pipelines) < 5:
    raise ValueError("Not enough valid models to create 5-classifier combinations.")

# Create model combinations
combos = list(itertools.combinations(pipelines.items(), 5))

# Output directory
output_dir = "D:\\Voting_Results"
os.makedirs(output_dir, exist_ok=True)

# Evaluate each 5-model combination on test data
for combo in combos:
    names = [name for name, _ in combo]
    combo_name = "_".join(names)

    try:
        estimators = []
        for name, (model, vectorizer) in combo:
            cloned_model = clone(model)
            pipeline = make_pipeline(clone(vectorizer), cloned_model)
            estimators.append((name, pipeline))

        voting_type = 'soft' if all(hasattr(model, "predict_proba") for (_, (model, _)) in combo) else 'hard'
        print(f"\nEvaluating combo: {combo_name} | Voting: {voting_type}")

        voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
        pred_test = [normalize_mni(p) for p in voting_clf.fit(test_sentences, y_test).predict(test_sentences)]

        report_test = classification_report(y_test, pred_test, zero_division=0)

        # Save classification report
        with open(os.path.join(output_dir, f"test_report_{combo_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{combo_name}\nClassification Report -sir_new_data_set Test Set ({voting_type} voting)\n\n")
            f.write(report_test)

        # Save prediction results
        pd.DataFrame({
            "sentence": test_sentences,
            "true_label": y_test,
            "predicted_label": pred_test
        }).to_csv(os.path.join(output_dir, f"new-dataset_test_preds_{combo_name}.csv"), index=False)

    except Exception as e:
        print(f"Skipping {combo_name} due to error: {e}")

print("\nAll test combinations evaluated and reports saved.")
