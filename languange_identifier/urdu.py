import re
import nltk
from re import search
from nltk.tokenize import sent_tokenize, RegexpTokenizer

# urdu text
urdu_text = '''
تعلیم ایک قوم کی ترقی کا بنیادی ستون ہے۔ یہ نہ صرف افراد کو علم فراہم کرتی ہے بلکہ انہیں معاشرے کا ایک مفید شہری بھی بناتی ہے۔ 
کیا آپ سمجھتے ہیں کہ تعلیم ضروری ہے؟ یقیناً! تعلیم کے بغیر ترقی ممکن نہیں۔  
ای میل: rajesh.kumar@gmail.com  
موبائل: +۹۲ ۳۴۵۶۷۸۹۰۱۲ 03456789012  
'''

# Urdu Unicode Range and Sentence Enders
urdu_unicode_block = r'[\u0600-\u06FF]'  
urdu_sentence_enders = r'[۔؟!]'          

# 1. Word Tokenization (extracting Urdu words)
urdu_word_pattern = r'[\u0600-\u06FF]+' 
urdu_words = re.findall(urdu_word_pattern, urdu_text)
print("Urdu Words:", urdu_words)

# 2. Sentence Tokenization 
urdu_sentence_pattern = r'[^۔؟!]+[۔؟!]'  
urdu_sentences = re.findall(urdu_sentence_pattern, urdu_text)
print("\nUrdu Sentences:")
for sentence in urdu_sentences:
    print(sentence.strip())

# 3. Paragraph Tokenization 
urdu_paragraphs = urdu_text.split("\n\n")
print("\nUrdu Paragraphs:")
for para in urdu_paragraphs:
    print(para.strip())

# 4. Email Extraction 
email_pattern_urdu = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
urdu_emails = re.findall(email_pattern_urdu, urdu_text)
print("\nUrdu Emails:", urdu_emails)

# 5. Mobile Number Extraction 
urdu_mobile_pattern = r'(?:\+۹۲\s?)?(?:\d{5}\s?\d{6}|\d{10,12})'
urdu_mobile_numbers = re.findall(urdu_mobile_pattern, urdu_text)
print("\nUrdu Mobile Numbers:", urdu_mobile_numbers)
