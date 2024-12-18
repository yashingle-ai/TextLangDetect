import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from re import findall, search

# Punjabi Text (contains email and mobile numbers)
punjabi_text = '''
ਸਿੱਖਿਆ ਜ਼ਿੰਦਗੀ ਵਿੱਚ ਬਹੁਤ ਮਹੱਤਵਪੂਰਨ ਭੂਮਿਕਾ ਨਿਭਾਉਂਦੀ ਹੈ। ਇਹ ਮਨੁੱਖ ਦੇ ਗਿਆਨ ਨੂੰ ਵਧਾਉਂਦੀ ਹੈ ਅਤੇ ਬੁੱਧੀਮਤਾ ਅਤੇ ਫੈਸਲੇ ਦੀ ਸਮਰੱਥਾ ਨੂੰ ਵਿਕਸਿਤ ਕਰਦੀ ਹੈ। ਸਿੱਖਿਆ ਕਿਸੇ ਵੀ ਸਮਾਜ ਦੀ ਤਰੱਕੀ ਲਈ ਜ਼ਰੂਰੀ ਹੈ।  
ਅਸੀਂ ਇੱਕ ਅਜਿਹੇ ਜਗਤ ਦੀ ਕਲਪਨਾ ਕਰ ਸਕਦੇ ਹਾਂ ਜਿੱਥੇ ਹਰ ਵਿਅਕਤੀ ਪੜ੍ਹਿਆ-ਲਿਖਿਆ ਹੋਵੇ। ਹਰ ਮਾਪੇ ਨੂੰ ਆਪਣੇ ਬੱਚਿਆਂ ਨੂੰ ਸਿੱਖਿਆ ਦੇਣੀ ਚਾਹੀਦੀ ਹੈ।  
Email: rajesh.kumar@gmail.com  
Mobile: +੯੧ ੯੮੫੬੭ ੧੨੩੪੫  9876543210  
ਨਲੰਦਾ ਅਤੇ ਤਕਸ਼ਸ਼ਿਲਾ ਦੀ ਤਰ੍ਹਾਂ, ਸਿੱਖਿਆ ਦਾ ਅਤੀਤ ਭਾਰਤ ਵਿੱਚ ਮਹਾਨ ਰਿਹਾ ਹੈ। +91 9876543210
'''

# Punjabi Unicode Block and Digit Block
punjabi_unicode_block = r'[\u0A00-\u0A7F]'  
punjabi_digit_block = r'[\u0A66-\u0A6F]'    

# 1. Word Tokenization
punjabi_word_pattern = r'[\u0A00-\u0A7F]+'  
punjabi_words = re.findall(punjabi_word_pattern, punjabi_text)
print("Punjabi Words:", punjabi_words)

# 2. Sentence Tokenization (Sentences ending with . or ! or ?)
# punjabi_sentence_pattern = r'[\u0A00-\u0A7F\s]+[.!?|]'
# punjabi_sentences = re.findall(punjabi_sentence_pattern, punjabi_text)
# print("\nPunjabi Sentences:", punjabi_sentences)

#2.1 sentence tokensation
punjabi_sentence=sent_tokenize(punjabi_text)
print("punjabi sentences are :-",punjabi_sentence)

# 3. Paragraph Tokenization
punjabi_paragraphs = punjabi_text.split("\n\n")
print("\nPunjabi Paragraphs:")
for para in punjabi_paragraphs:
    print(para)

# 4. Email Extraction (Punjabi + English Emails)
email_pattern_punjabi = r'[\u0A00-\u0A7Fa-zA-Z0-9._%+-]+@[\u0A00-\u0A7Fa-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
punjabi_emails = re.findall(email_pattern_punjabi, punjabi_text)
print("\nPunjabi Emails:", punjabi_emails)

# 5. Mobile Number Extraction (Punjabi and English Digits)
punjabi_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0A66-\u0A6F]{2}\s?)?[\u0A66-\u0A6F0-9]{5}\s?[\u0A66-\u0A6F0-9]{5}'
punjabi_mobile_numbers = re.findall(punjabi_mobile_pattern, punjabi_text)
print("\nPunjabi Mobile Numbers:", punjabi_mobile_numbers)
