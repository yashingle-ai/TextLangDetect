### **Language Identification Research Project**  
*A comprehensive study on text representation, machine learning, and fastText-based language classification.*

## **Overview**  
This research focuses on developing an accurate **language identifier** by implementing text representation methods and various machine learning models from scratch. The project evaluates performance using both **handcrafted algorithms** and **library-based approaches** to compare efficiency and accuracy.

## **Project Workflow**  

### ** 1️. Data Collection & Preprocessing**  
- Collected and preprocessed data for **22 languages**.  
- Extracted key linguistic features from sentences.  

### ** 2️. Text Representation Techniques**  
Implemented various **feature extraction** methods from scratch and compared them with library-based implementations:  
- **Bag of Words (BoW)**  
- **K Most Similar Sentences**  
- **Local TF-IDF**  
- **Bigram TF-IDF**  
- **Sentence Similarity**  

### ** 3️. Information Extraction Across Languages**  
- Used **`re` library** to extract mobile numbers, addresses, names, and other essential information from multilingual text.  

### ** 4️. Rule-Based Language Identifier**  
- Developed an initial **rule-based** approach, but it had poor accuracy.  

### ** 5️. Machine Learning for Language Identification**  
Implemented various **machine learning classifiers** from scratch and compared them with **library-based** versions:  
- **AdaBoost**  
- **K-Nearest Neighbors (KNN)**  
- **Logistic Regression**  
- **Multinomial Naïve Bayes**  
- **Support Vector Classifier (SVC)**  
- **Decision Trees**  
- **Voting Classifier (Ensemble Learning with multiple algorithms)**  

### ** 6️. Model Enhancements & Hyperparameter Tuning**  
- Used **n-grams** for better feature extraction.  
- Applied **Laplace Correction (0.01, 0.001)** for Naïve Bayes.  
- Evaluated models using **precision, recall, and F1-score** for each language.  
- Created detailed reports in text files.  

### ** 7️. Scaling Up: FastText Classifier**  
- Prepared a dataset of **10K+ sentences per language** for **22 languages**.  
- Trained a **FastText text classifier** for large-scale language identification.  

## **How to Use**  

### **Setup Environment**  
Ensure Python is installed, then install required dependencies:  
```bash
pip install -r requirements.txt
```

### **Train & Evaluate Models**  
- Run text representation methods:  
  ```bash
  python text_representation.py
  ```
- Train and test ML classifiers:  
  ```bash
  python train_ml_models.py
  ```
- Train FastText on large dataset:  
  ```bash
  python train_fasttext.py
  ```

### **Results & Reports**  
- **Evaluation metrics** for each model are saved in text files.  
- FastText accuracy is reported in CSV format.  

## **Contributors**  
- **Yash Ingle** – B.Tech AI, SVNIT Surat
- **Dr. Pruthwik Mishra - Assistant Professor SVNIT Surat**

---
