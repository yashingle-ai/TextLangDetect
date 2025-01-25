# for identification of sentence with the help of logistic regrssion algorithms 


#import required libraries
from sklearn.linear_model import LogisticRegression            #for our algorithms
from sklearn.model_selection import GridSearchCV               #for hypertuning
from sklearn.metrics import classification_report              #for summery of precision recall fscore 
from sklearn.feature_extraction.text import TfidfVectorizer    #for converting sentence into numeric form
from sklearn.preprocessing import LabelEncoder      #label encoder for converting labeles means label of sentence which is language to convert into laebls 

# Function to load data
def load_data(sentence_file, label_file):
    """Load sentences and labels from files."""
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    # Remove extra whitespace/newlines
    sentences = [line.strip() for line in sentences]
    labels = [line.strip() for line in labels]
      
    return sentences, labels

# File paths
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
tfidf_vectorizer = TfidfVectorizer()

# Transform data
X_train = tfidf_vectorizer.fit_transform(train_sentences)                   #here we first use fit and transform both method for creating vocabulary as well as converting sentence acording to the library 
y_labels = label_encoder.fit_transform(train_labels)                         

X_test = tfidf_vectorizer.transform(test_sentences)                         #here we just use transform method for converting sentence and labels into numeric for with the help of previous library 
y_test = label_encoder.transform(test_labels)

X_dev = tfidf_vectorizer.transform(dev_sentences)                           #here we just use transform method for converting sentence and labels into numeric for with the help of previous library 
y_dev = label_encoder.transform(dev_labels)

# Logistic Regression
log_reg = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000)

# Parameter Grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # ElasticNet removed due to complexity for now
}

# GridSearchCV
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_labels)

# Best Parameters and Evaluation
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluation on Test Data
y_pred_test = best_model.predict(X_test)
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Evaluation on Dev Data
y_pred_dev = best_model.predict(X_dev)
print("\nDev Set Classification Report:")
print(classification_report(y_dev, y_pred_dev, target_names=label_encoder.classes_))
