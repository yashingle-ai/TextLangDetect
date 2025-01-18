import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from collections import defaultdict

class NaiveBayesTFIDF:
    def __init__(self):
        self.class_priors = {}  # Log prior probabilities for each class
        self.class_word_probs = {}  # Log conditional probabilities for each word and class
        self.classes = []  # List of unique class labels
        self.vectorizer = None  # TF-IDF vectorizer instance

    def preprocess(self, sentence):
        """ Basic text preprocessing """
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
        sentence = sentence.lower()  # Convert to lowercase
        return sentence

    def fit(self, X, y):
        """ Train the Naive Bayes model with TF-IDF vectors """
        n_samples = len(X)  #number of samples (sentences)
        self.classes = list(set(y))  # Get unique class labels

        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(preprocessor=self.preprocess)
        X_tfidf = self.vectorizer.fit_transform(X).toarray()  # Convert to dense array

        # Initialize dictionaries for class priors and word probabilities
        class_counts = defaultdict(int)       #counts the number of sentences of perticular language intially with default values
        word_counts = defaultdict(lambda: np.zeros(X_tfidf.shape[1]))  
        #word_count store language with thier vocabulary like word_count={hindi=[hindi_vocabulary vector ],english=[eng_vocabulaary vector]and same for other language} 
        

        # Calculate class priors and word frequencies
        for i in range(n_samples):    #iterate through each sentnces 
            class_counts[y[i]] += 1    #y[i] returns the language and after that class_count of element(language) of that language increase with one 
            word_counts[y[i]] += X_tfidf[i]     #y[i] return the language an after that word_count of elment(language) of that add array here 
            

        # Calculate class priors (log of the probability for numerical stability)
        for cls in self.classes:    #iterate through each unique language 
            self.class_priors[cls] = np.log(class_counts[cls] / float(n_samples))         #calculating language probability 

        # Calculate word probabilities for each class
        self.class_word_probs = {}      
        
        for cls in self.classes:        
            total_tfidf_in_class = word_counts[cls].sum()  # Total TF-IDF weight in this class 
            self.class_word_probs[cls] = np.log((word_counts[cls] + 1) / (total_tfidf_in_class + len(word_counts[cls])))     #here  i use lapalace smothing formula 

    
    #defining function predict
    def predict(self, X):
        """ Predict the class for each sentence """
        X_tfidf = self.vectorizer.transform(X).toarray()  # Transform input to TF-IDF vectors
        y_pred = [self._predict(x) for x in X_tfidf]
        return np.array(y_pred)

    def _predict(self, x):
        """ Predict for a single TF-IDF vector """
        posteriors = {}

        # Calculate the posterior probability for each class
        for cls in self.classes:
            posterior = self.class_priors[cls]  # Start with the prior
            posterior += np.sum(x * self.class_word_probs[cls])  # Weighted sum of log probabilities
            posteriors[cls] = posterior

        # Return the class with the highest posterior probability
        return max(posteriors, key=posteriors.get)


def load_data(sentence_file, label_file):
    """ Load sentences and labels from files """
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    # Remove extra whitespace/newlines
    sentences = [line.strip() for line in sentences]
    labels = [line.strip() for line in labels]

    return sentences, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Classifier with TF-IDF")
    parser.add_argument('--train_sentences', required=True, help="Path to training sentences file")
    parser.add_argument('--train_labels', required=True, help="Path to training labels file")
    parser.add_argument('--dev_sentences', required=True, help="Path to development sentences file")
    parser.add_argument('--dev_labels', required=True, help="Path to development labels file")
    parser.add_argument('--test_sentences', required=True, help="Path to test sentences file")
    parser.add_argument('--test_labels', required=True, help="Path to test labels file")

    args = parser.parse_args()

    # Load datasets
    train_sentences, train_labels = load_data(args.train_sentences, args.train_labels)
    dev_sentences, dev_labels = load_data(args.dev_sentences, args.dev_labels)
    test_sentences, test_labels = load_data(args.test_sentences, args.test_labels)

    # Initialize and train the classifier on the training set
    nb_tfidf = NaiveBayesTFIDF()
    nb_tfidf.fit(train_sentences, train_labels)

    # Evaluate on the development set
    dev_predictions = nb_tfidf.predict(dev_sentences)
    dev_accuracy = accuracy_score(dev_labels, dev_predictions)
    print("Development Set Accuracy:", dev_accuracy)

    # Evaluate on the test set
    test_predictions = nb_tfidf.predict(test_sentences)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print("Test Set Accuracy:", test_accuracy)

    # Example predictions for new sentences
    new_sentences = [
        "यह एक नया उदाहरण है",  # Hindi
        "I am learning machine learning naive bayes algorithm.",  # English
    ]
    new_predictions = nb_tfidf.predict(new_sentences)
    print("Predictions for new sentences:", new_predictions)
