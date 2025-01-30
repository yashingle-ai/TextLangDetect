# Import required libraries
import itertools
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Function to load data
def load_data(sentence_file, label_file):
    """Load sentences and labels from files."""
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    sentences = [line.strip() for line in sentences]
    labels = [line.strip() for line in labels]

    return sentences, labels

# File paths (keeping your original paths)
train_sentence_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_train_sent_all.txt'
train_label_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_train_label_all.txt'
dev_sentence_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_dev_sent_all.txt'
dev_label_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_dev_label_all.txt'
test_sentence_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_test_sent_all.txt'
test_label_file = 'C:\\Users\\yashi\\OneDrive\\Desktop\\ML project\\text_suffling\\shuffle_test_label_all.txt'

# Load data
train_sentences, train_labels = load_data(train_sentence_file, train_label_file)
dev_sentences, dev_labels = load_data(dev_sentence_file, dev_label_file)
test_sentences, test_labels = load_data(test_sentence_file, test_label_file)

# Initialize Label Encoder and TF-IDF Vectorizer
label_encoder = LabelEncoder()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using bigrams for better feature extraction

# Transform data
X_train = tfidf_vectorizer.fit_transform(train_sentences)
y_train = label_encoder.fit_transform(train_labels)

X_test = tfidf_vectorizer.transform(test_sentences)
y_test = label_encoder.transform(test_labels)

X_dev = tfidf_vectorizer.transform(dev_sentences)
y_dev = label_encoder.transform(dev_labels)

# Define classifiers with your specified hyperparameters
classifiers = {
    "SVM": SVC(probability=True, C=1, kernel='rbf'),
    "KNN": KNeighborsClassifier(n_neighbors=1),  # Choosing smallest value from [1,3,5,7]
    "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=2, max_depth=None, criterion='gini'),
    "LogisticRegression": LogisticRegression(C=1),  # Default value from [0.1,1,10]
    "SGD": SGDClassifier(penalty='l2', learning_rate='optimal', l1_ratio=np.float64(1.0), eta0=np.float64(0.03162277660168379), alpha=np.float64(0.0001)),
    "AdaBoost": AdaBoostClassifier(n_estimators=50),  # Choosing first value from [50,100]
    "DecisionTree": DecisionTreeClassifier(max_depth=10),  # Selecting a value from [5,10,20,None]
    "NaiveBayes": MultinomialNB(alpha=0.001)
}

# Generate all possible 3-classifier combinations
combinations = list(itertools.combinations(classifiers.keys(), 3))

# Evaluate each ensemble combination
for combo in combinations:
    print(f"\nEvaluating combination: {combo}")
    
    selected_models = [(name, classifiers[name]) for name in combo]
    ensemble = VotingClassifier(estimators=selected_models, voting='soft')
    ensemble.fit(X_train, y_train)

    # Predict on test set
    y_pred_test = ensemble.predict(X_test)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    # Predict on dev set
    y_pred_dev = ensemble.predict(X_dev)
    print("\nDev Set Classification Report:")
    print(classification_report(y_dev, y_pred_dev, target_names=label_encoder.classes_))

print("Evaluation completed for all classifier combinations!")
