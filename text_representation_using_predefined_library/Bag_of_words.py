from sklearn.feature_extraction.text import CountVectorizer
import re 
from nltk.tokenize import word_tokenize
import argparse
# instialise
vector=CountVectorizer()

# preprocess the text 
def pre_process(text):
    text = re.sub(r'[\n\t]', ' ', text)             # Remove newlines and tabs
    text = re.sub(r'[^\w\s]', '', text.lower())     # Remove punctuation and convert to lowercase
    return word_tokenize(text)   # Tokenize text into words

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input_file",type=str ,help="give the input file  that contain sentences in each lines ")
    args=parser.parse_args()
    
    #if file contain diffrent language sentence then change the encoding part of input file (here i am considering that the file in in eng language)
    
    with open(args.input_file,'r',encoding='utf-8')as f:
        lines = f.readlines()
        
        #preprocess :removing punctuation and removing extra whitespace and lowercase the words 
        cleaned_sentences = [pre_process(line.strip()) for line in lines if line.strip()]
        cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]
        
        
        
        #initailse the vector using countvecorizer
        vectoriser=CountVectorizer()
        
        #converting the sentence into bag of words 
        #here bag_of_words_matrices contain all sentence representation into the vector form
        Bag_of_words_matrices = vectoriser.fit_transform(cleaned_sentences)
        
        # vocabulary contain features name or here we have feature as words
        volcabulary=vectoriser.get_feature_names_out()
        
        print(volcabulary)
        
        # Convert the sparse matrix to a dense array for easier printing
        dense_matrix = Bag_of_words_matrices.toarray()

        # Print the matrix with column headers
        print("Feature Names (Vocabulary):", volcabulary)
        print("\nBag of Words Representation (Matrix):\n")

        # Print each row (sentence) and its vector representation
        for i, row in enumerate(dense_matrix):
            print(f"Sentence {i + 1}: {row}")
                
        
        

