import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# English Text
english_text = '''
Education plays a vital role in our lives. It provides knowledge, skills, and the ability to make decisions. 
Every parent should ensure their children receive education. Education is the backbone of a nation's progress!

Email: john.doe@example.com, support@mywebsite.co.uk  
Mobile: +1 12345 67890  9876543210  
In ancient times, centers like Nalanda and Takshashila were renowned for education. 
Education without action is like a tree without fruit.
'''

# 1. Word Tokenization 
english_words = word_tokenize(english_text)
print("English Words:", english_words)

# 2. Sentence Tokenization 
english_sentences = sent_tokenize(english_text)
print("\nEnglish Sentences:")
for sentence in english_sentences:
    print(sentence)

# 3. Paragraph Tokenization 
english_paragraphs = english_text.strip().split("\n\n")
print("\nEnglish Paragraphs:")
for para in english_paragraphs:
    print(para)

# 4. Email Extraction
email_pattern_english = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
english_emails = re.findall(email_pattern_english, english_text)
print("\nEnglish Emails:", english_emails)

# 5. Mobile Number Extraction
english_mobile_pattern = r'(?:\+1\s?)?\d{5}\s?\d{5}|\d{10,12}' 
english_mobile_numbers = re.findall(english_mobile_pattern, english_text)
print("\nEnglish Mobile Numbers:", english_mobile_numbers)
