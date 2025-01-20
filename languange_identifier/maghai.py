import re

# Magahi Text
magahi_text = '''
हमर नाम राजेश कुमार ह। हम पटना से बानी। 
ईमेल: rajesh.kumar@gmail.com  राजेश.कुमार@गमेल.कॉम  
मोबाइल: +91 98567 12345  9876543210  
भारत के प्राचीन विश्वविद्यालय, नालंदा और तक्षशिला के बारे में बहुत कुछ कहा जाता है। 
'''

# Magahi Unicode Block and Digit Block
magahi_unicode_block = r'[\u0900-\u097F]'  # Devanagari block used for Magahi
magahi_digit_block = r'[\u0966-\u096F]'    # Devanagari digits

# 1. Word Tokenization (Magahi words in Devanagari script)
magahi_word_pattern = r'[\u0900-\u097F]+' 
magahi_words = re.findall(magahi_word_pattern, magahi_text)
print("Magahi Words:", magahi_words)

# 2. Sentence Tokenization (Sentences ending with । or ! or ?)
magahi_sentence_pattern = r'[\u0900-\u097F\s]+[।!?]'
magahi_sentences = re.findall(magahi_sentence_pattern, magahi_text)
print("\nMagahi Sentences:", magahi_sentences)

# 3. Paragraph Tokenization
magahi_paragraphs = magahi_text.split("\n\n")
print("\nMagahi Paragraphs:")
for para in magahi_paragraphs:
    print(para)

# 4. Email Extraction (Magahi + English Emails)
email_pattern_magahi = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[\u0900-\u097Fa-zA-Z]{2,}'
magahi_emails = re.findall(email_pattern_magahi, magahi_text)
print("\nMagahi Emails:", magahi_emails)

# 5. Mobile Number Extraction (Magahi and English Digits)
magahi_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'
magahi_mobile_numbers = re.findall(magahi_mobile_pattern, magahi_text)
print("\nMagahi Mobile Numbers:", magahi_mobile_numbers)
