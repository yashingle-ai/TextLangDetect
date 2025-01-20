import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Meitei (Manipuri) Text
manipuri_text = '''
ꯃꯇꯩꯂꯣꯟ ꯍꯥꯛ ꯀꯇꯨꯂꯥ ꯆꯥꯛꯕꯤ ꯍꯨꯛꯤꯡ ꯄꯨꯝꯂꯦꯟ ꯇꯝꯁꯤ ꯇꯡ ꯇꯝꯁꯤ ꯇꯡ ꯇꯩꯂꯥꯅꯥ ꯆꯥꯛꯕꯤ ꯀꯩꯄꯨꯂ ꯄꯨꯝꯂꯦꯟ ꯅꯧꯕꯥ ꯇ꯭ꯔꯤꯇꯦ ꯇꯝꯁꯤꯇꯦ ꯅꯧꯡꯛꯁꯤ ꯍꯛꯀꯤꯡ ꯍꯥꯛ ꯁꯦꯝꯥꯔꯦꯟ ꯆꯨꯋ ꯄꯨꯝꯂꯦꯟ ꯇꯝꯁꯤ ꯇꯡ ꯁꯧꯝꯅꯝꯀꯤ ꯈꯨꯝꯂꯦꯟ ꯇꯝꯁꯤ.
'''

# Meitei (Manipuri) Unicode Block and Digit Block
manipuri_unicode_block = r'[\uABC0-\uABFF]'  # Unicode block for Meitei
manipuri_digit_block = r'[\u09E6-\u09EF]'  # Unicode block for Bengali digits (same for Manipuri)

# 1. Word Tokenization
manipuri_word_pattern = r'[\uABC0-\uABFF]+'  # Adjusting for Meitei characters
manipuri_words = re.findall(manipuri_word_pattern, manipuri_text)
print("Meitei (Manipuri) Words:", manipuri_words)

# 2. Sentence Tokenization
# Update the regex to catch sentence-ending punctuation more accurately.
# We'll look for sentences that end with ꯁ (Meitei punctuation) or with standard punctuation marks like '!' or '।'
manipuri_sentence_pattern = r'[\uABC0-\uABFF\s]+[।!?ꯁ]'
manipuri_sentences = re.findall(manipuri_sentence_pattern, manipuri_text)
print("\nMeitei (Manipuri) Sentences:", manipuri_sentences)

# 3. Paragraph Tokenization
manipuri_paragraphs = manipuri_text.split("\n\n")
print("\nMeitei (Manipuri) Paragraphs:")
for para in manipuri_paragraphs:
    print(para)

# 4. Email Extraction (Meitei + English Emails)
email_pattern_manipuri = r'[\uABC0-\uABFFa-zA-Z0-9._%+-]+@[\uABC0-\uABFFa-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
manipuri_emails = re.findall(email_pattern_manipuri, manipuri_text)
print("\nMeitei (Manipuri) Emails:", manipuri_emails)

# 5. Mobile Number Extraction (Manipuri and English Digits)
mobile_pattern_manipuri = r'(?:\+91\s?)?(?:[\+\u09E6-\u09EF]{2}\s?)?[\u09E6-\u09EF0-9]{5}\s?[\u09E6-\u09EF0-9]{5}'
manipuri_mobile_numbers = re.findall(mobile_pattern_manipuri, manipuri_text)
print("\nMeitei (Manipuri) Mobile Numbers:", manipuri_mobile_numbers)
