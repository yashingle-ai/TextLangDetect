import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Assamese Text
assamese_text = '''
শিক্ষা জীৱনৰ এটা গুৰুত্বপূৰ্ণ অংশ। ই মানুহক জ্ঞান আৰু কৌশল প্ৰদান কৰে। 
প্ৰতিটো অভিভাৱকৰ উচিত তেওঁলোকৰ সন্তানক শিক্ষিত কৰা। 
Email: rajesh.kumar@gmail.com  ৰাজেশ.কুমাৰ@গমেইল.কম  
Mobile: +৯১ ৯৮৫৬৭ ১২৩৪৫  9876543210  
প্ৰাচীন ভাৰতত নালন্দা আৰু তক্ষশিলাৰ দৰে শিক্ষাকেন্দ্ৰ আছিল। +91 9876543210
'''

# Assamese Unicode Block and Digit Block
assamese_unicode_block = r'[\u0980-\u09FF]' 
assamese_digit_block = r'[\u09E6-\u09EF]'   

# 1. Word Tokenization
assamese_word_pattern = r'[\u0980-\u09FF]+' 
assamese_words = re.findall(assamese_word_pattern, assamese_text)
print("Assamese Words:", assamese_words)

# 2. Sentence Tokenization (Sentences ending with । or ! or ?)
assamese_sentence_pattern = r'[\u0980-\u09FF\s]+[।!?/n]'
assamese_sentences = re.findall(assamese_sentence_pattern, assamese_text)
print("\nAssamese Sentences:", assamese_sentences)

#2.1 sentence tokenisation
assamese_sentences=sent_tokenize(assamese_text)   #here sent tokensation is not working for assames language()
print("\nAssamese Sentences:",assamese_sentences)

# 3. Paragraph Tokenization
assamese_paragraphs = assamese_text.split("\n\n")
print("\nAssamese Paragraphs:")
for para in assamese_paragraphs:
    print(para)

# 4. Email Extraction (Assamese + English Emails)
email_pattern_assamese = r'[\u0980-\u09FFa-zA-Z0-9._%+-]+@[\u0980-\u09FFa-zA-Z0-9.-]+\.[\u0980-\u09FFa-zA-Z]{2,}'
assamese_emails = re.findall(email_pattern_assamese, assamese_text)
print("\nAssamese Emails:", assamese_emails)

# 5. Mobile Number Extraction (Assamese and English Digits)
assamese_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u09E6-\u09EF]{2}\s?)?[\u09E6-\u09EF0-9]{5}\s?[\u09E6-\u09EF0-9]{5}'
assamese_mobile_numbers = re.findall(assamese_mobile_pattern, assamese_text)
print("\nAssamese Mobile Numbers:", assamese_mobile_numbers)
