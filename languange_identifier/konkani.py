import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Konkani Text (Devanagari Script with Email and Mobile Numbers)
konkani_text = '''
शिक्षण ही जीवनाची महत्वपूर्ण गोष्ट आहे। हे माणसाला ज्ञान आणि कौशल्य प्राप्त करून देते। 
प्रत्येक पालकांनी त्यांच्या मुलांना शिक्षण देणे गरजेचे आहे। 
Email: rajesh.kumar@gmail.com  राजेश.कुमार@गमैल.कॉम  
Mobile: +९१ ९८५६७ १२३४५  9876543210  
भारतामध्ये नालंदा आणि तक्षशिला यांसारखे महान शिक्षण केंद्रे होती। +91 9876543210
'''

# Konkani Unicode Block and Digits
konkani_unicode_block = r'[\u0900-\u097F]'  
konkani_digit_block = r'[\u0966-\u096F]'    

# 1. Word Tokenization
konkani_word_pattern = r'[\u0900-\u097F]+'  
konkani_words = re.findall(konkani_word_pattern, konkani_text)
print("Konkani Words:", konkani_words)

# 2. Sentence Tokenization (Sentences ending with । or ! or ?)
konkani_sentence_pattern = r'[\u0900-\u097F\s]+[।!?]'
konkani_sentences = re.findall(konkani_sentence_pattern, konkani_text)
print("\nKonkani Sentences:", konkani_sentences)

# 3. Paragraph Tokenization
konkani_paragraphs = konkani_text.split("\n\n")
print("\nKonkani Paragraphs:")
for para in konkani_paragraphs:
    print(para)

# 4. Email Extraction (Konkani + English Emails)
email_pattern_konkani = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[\u0900-\u097Fa-zA-Z]{2,}'
konkani_emails = re.findall(email_pattern_konkani, konkani_text)
print("\nKonkani Emails:", konkani_emails)

# 5. Mobile Number Extraction (Konkani and English Digits)
konkani_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'
konkani_mobile_numbers = re.findall(konkani_mobile_pattern, konkani_text)
print("\nKonkani Mobile Numbers:", konkani_mobile_numbers)
