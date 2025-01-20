import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Nepali Text
nepali_text = '''
नेपाल एक सुंदर देश हो। यहाँका मानिसहरु शान्त र मिलनसार छन्। 
नेपालमा विभिन्न जातजातिहरु बसोबास गर्छन्। 
Email: rajesh.kumar@gmail.com  राजेश.कुमार@गमेल.कम  
Mobile: +९१ ९८५६७ १२३४५  9876543210  
काठमाडौं शहर नेपालको राजधानी हो। +91 9876543210
'''

# Nepali Unicode Block and Digit Block
nepali_unicode_block = r'[\u0900-\u097F]'  # Unicode block for Nepali
nepali_digit_block = r'[\u0966-\u096F]'  # Unicode block for Nepali digits

# 1. Word Tokenization
nepali_word_pattern = r'[\u0900-\u097F]+'  # Adjusting for Nepali characters
nepali_words = re.findall(nepali_word_pattern, nepali_text)
print("Nepali Words:", nepali_words)

# 2. Sentence Tokenization
# Update the regex to catch sentence-ending punctuation more accurately.
# We'll look for sentences that end with '।' (Nepali punctuation), '!' or '?'
nepali_sentence_pattern = r'[\u0900-\u097F\s]+[।!?]'
nepali_sentences = re.findall(nepali_sentence_pattern, nepali_text)
print("\nNepali Sentences:", nepali_sentences)

# 3. Paragraph Tokenization
nepali_paragraphs = nepali_text.split("\n\n")
print("\nNepali Paragraphs:")
for para in nepali_paragraphs:
    print(para)

# 4. Email Extraction (Nepali + English Emails)
email_pattern_nepali = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
nepali_emails = re.findall(email_pattern_nepali, nepali_text)
print("\nNepali Emails:", nepali_emails)

# 5. Mobile Number Extraction (Nepali and English Digits)
mobile_pattern_nepali = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'
nepali_mobile_numbers = re.findall(mobile_pattern_nepali, nepali_text)
print("\nNepali Mobile Numbers:", nepali_mobile_numbers)
