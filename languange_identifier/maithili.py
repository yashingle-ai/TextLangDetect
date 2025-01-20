import re

# Maithili Text
maithili_text = '''
हमर नाम राजेश कुमार ह। हम दरभंगा से छी। 
ईमेल: rajesh.kumar@gmail.com  राजेश.कुमार@गमेल.कॉम  
मोबाइल: +91 98567 12345  9876543210  
भारत के प्राचीन विश्वविद्यालय, नालंदा और तक्षशिला के बारे में बहुत कुछ कहा जाता है। 
'''

# Maithili Unicode Block and Digit Block
maithili_unicode_block = r'[\u0900-\u097F]'  # Devanagari block used for Maithili
maithili_digit_block = r'[\u0966-\u096F]'    # Devanagari digits

# 1. Word Tokenization (Maithili words in Devanagari script)
maithili_word_pattern = r'[\u0900-\u097F]+' 
maithili_words = re.findall(maithili_word_pattern, maithili_text)
print("Maithili Words:", maithili_words)

# 2. Sentence Tokenization (Sentences ending with । or ! or ?)
maithili_sentence_pattern = r'[\u0900-\u097F\s]+[।!?]'
maithili_sentences = re.findall(maithili_sentence_pattern, maithili_text)
print("\nMaithili Sentences:", maithili_sentences)

# 3. Paragraph Tokenization
maithili_paragraphs = maithili_text.split("\n\n")
print("\nMaithili Paragraphs:")
for para in maithili_paragraphs:
    print(para)

# 4. Email Extraction (Maithili + English Emails)
email_pattern_maithili = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[\u0900-\u097Fa-zA-Z]{2,}'
maithili_emails = re.findall(email_pattern_maithili, maithili_text)
print("\nMaithili Emails:", maithili_emails)

# 5. Mobile Number Extraction (Maithili and English Digits)
maithili_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'
maithili_mobile_numbers = re.findall(maithili_mobile_pattern, maithili_text)
print("\nMaithili Mobile Numbers:", maithili_mobile_numbers)
