import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Hindi Text
hindi_text = '''
शिक्षा जीवन का एक महत्वपूर्ण हिस्सा है। यह व्यक्ति को ज्ञान और कौशल प्रदान करती है। 
हर माता-पिता को अपने बच्चों को शिक्षित करना चाहिए। 
Email: rajesh.kumar@gmail.com  राजेश.कुमार@जीमेल.कॉम  
Mobile: +९१ ९८५६७ १२३४५  9876543210  
प्राचीन भारत में नालंदा और तक्षशिला जैसे शिक्षा केंद्र थे। +91 9876543210
'''

# Hindi Unicode Block and Digit Block
hindi_unicode_block = r'[\u0900-\u097F]' 
hindi_digit_block = r'[\u0966-\u096F]'   

# 1. Word Tokenization
hindi_word_pattern = r'[\u0900-\u097F]+' 
hindi_words = re.findall(hindi_word_pattern, hindi_text)
print("Hindi Words:", hindi_words)

# 2. Sentence Tokenization (Sentences ending with ।, !, or ?)
hindi_sentence_pattern = r'[\u0900-\u097F\s]+[।!?]'
hindi_sentences = re.findall(hindi_sentence_pattern, hindi_text)
print("\nHindi Sentences (Regex):", hindi_sentences)

# 2.1 Sentence Tokenization using NLTK (not suitable for Hindi)
hindi_sentences_nltk = sent_tokenize(hindi_text)   # Doesn't work effectively for Hindi
print("\nHindi Sentences (NLTK):", hindi_sentences_nltk)

# 3. Paragraph Tokenization
hindi_paragraphs = hindi_text.split("\n\n")
print("\nHindi Paragraphs:")
for para in hindi_paragraphs:
    print(para)

# 4. Email Extraction (Hindi + English Emails)
email_pattern_hindi = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[\u0900-\u097Fa-zA-Z]{2,}'
hindi_emails = re.findall(email_pattern_hindi, hindi_text)
print("\nHindi Emails:", hindi_emails)

# 5. Mobile Number Extraction (Hindi and English Digits)
hindi_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'
hindi_mobile_numbers = re.findall(hindi_mobile_pattern, hindi_text)
print("\nHindi Mobile Numbers:", hindi_mobile_numbers)
