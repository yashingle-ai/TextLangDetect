import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Bengali Text
bengali_text = '''
শিক্ষা জীবনের একটি গুরুত্বপূর্ণ অংশ। এটি মানুষকে জ্ঞান এবং দক্ষতা প্রদান করে। 
প্রতিটি পিতামাতার উচিত তাদের সন্তানদের শিক্ষিত করা। 
Email: rajesh.kumar@gmail.com  রাজেশ.কুমার@গমেইল.কম  
Mobile: +৯১ ৯৮৫৬৭ ১২৩৪৫  9876543210  
প্রাচীন ভারতে নালন্দা এবং তক্ষশিলার মতো শিক্ষাকেন্দ্র ছিল। +91 9876543210
'''

# Bengali Unicode Block and Digit Block
bengali_unicode_block = r'[\u0980-\u09FF]' 
bengali_digit_block = r'[\u09E6-\u09EF]'    

# 1. Word Tokenization
bengali_word_pattern = r'[\u0980-\u09FF]+' 
bengali_words = re.findall(bengali_word_pattern, bengali_text)
print("Bengali Words:", bengali_words)

# 2. Sentence Tokenization (Sentences ending with । or ! or ?)
bengali_sentence_pattern = r'[\u0980-\u09FF\s]+[।!?]'
bengali_sentences = re.findall(bengali_sentence_pattern, bengali_text)
print("\nBengali Sentences:", bengali_sentences)

# 3. Paragraph Tokenization
bengali_paragraphs = bengali_text.split("\n\n")
print("\nBengali Paragraphs:")
for para in bengali_paragraphs:
    print(para)

# 4. Email Extraction (Bengali + English Emails)
email_pattern_bengali = r'[\u0980-\u09FFa-zA-Z0-9._%+-]+@[\u0980-\u09FFa-zA-Z0-9.-]+\.[\u0980-\u09FFa-zA-Z]{2,}'
bengali_emails = re.findall(email_pattern_bengali, bengali_text)
print("\nBengali Emails:", bengali_emails)

# 5. Mobile Number Extraction (Bengali and English Digits)
bengali_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u09E6-\u09EF]{2}\s?)?[\u09E6-\u09EF0-9]{5}\s?[\u09E6-\u09EF0-9]{5}'
bengali_mobile_numbers = re.findall(bengali_mobile_pattern, bengali_text)
print("\nBengali Mobile Numbers:", bengali_mobile_numbers)
