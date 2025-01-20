import re
import nltk
from nltk.tokenize import word_tokenize

# Santali Text (Ol Chiki script)
santali_text = '''
ᱡᱤᱧᱮᱞᱟ ᱵᱷᱟᱯᱮᱛ ᱥᱮᱡᱤ ᱠᱩᱛᱷᱟᱭᱟᱝ ᱢᱮᱝᱝᱟᱨᱟᱝ।ᱥᱮᱡᱤ ᱢᱮᱝᱝᱟᱨᱟᱝ ᱟᱞᱮᱛᱮᱜᱤ ᱟᱛᱟᱜᱟᱛ ᱥᱮᱡᱤ ᱡᱤᱧᱮᱞᱟ ᱥᱮᱡᱤ ᱨᱮᱵᱮᱛᱮᱜᱤ ᱧᱮᱝᱛᱮᱛᱟᱝ ᱜᱷᱟᱝᱷᱮᱞᱤ ?
ᱥᱮᱡᱤ ᱢᱮᱝᱝᱟᱨᱟᱝ ᱟᱞᱮᱛᱮᱜᱤ ᱟᱛᱟᱜᱟᱛ ᱥᱮᱡᱤ ᱡᱤᱧᱮᱞᱟ ᱥᱮᱡᱤ ᱨᱮᱵᱮᱛᱮᱜᱤ ᱧᱮᱝᱛᱮᱛᱟᱝ ᱜᱷᱟᱝᱷᱮᱞᱤ।
Email: rajesh.kumar@gmail.com  ᱨᱟᱯᱤᱥ.ᱠᱩᱛᱤᱯ@gmail.com  
Mobile: +91 98567 12345  9876543210  
ᱵᱷᱟᱯᱮᱛ ᱥᱮᱡᱤ ᱢᱮᱝᱝᱟᱨᱟᱝ ᱡᱤᱧᱮᱞᱟ ᱥᱮᱡᱤ ᡞᱤᱧᱮᱞᱟᱝᱮᱠᱟᱝ ᱧᱮᱝᱛᱮᱛᱟᱝ।
'''

# Santali Unicode Block and Digit Block (Ol Chiki script and Santali digits)
santali_unicode_block = r'[\u1C00-\u1C4F]'  # Ol Chiki script range
santali_digit_block = r'[\u1C50-\u1C59]'  # Santali digits range

# 1. Custom Sentence Tokenization based on full stop (।) symbol, newlines, and extra spaces
santali_sentences = re.split(r'(?<=।)|\n', santali_text.strip())  # split at full stop (।) and newlines
santali_sentences = [sentence.strip() for sentence in santali_sentences if sentence.strip()]  # Remove empty strings

print("\nSantali Sentences:", santali_sentences)

# Checking the sentence count
print("Santali Sentences Length:", len(santali_sentences))


# 3. Paragraph Tokenization
# Splitting the text by blank lines or paragraph markers
santali_paragraphs = santali_text.split("\n\n")
print("\nSantali Paragraphs:")
for para in santali_paragraphs:
    print(para)

# 4. Email Extraction (Santali + English Emails)
email_pattern_santali = r'[\u1C00-\u1C4Fa-zA-Z0-9._%+-]+@[\u1C00-\u1C4Fa-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
santali_emails = re.findall(email_pattern_santali, santali_text)
print("\nSantali Emails:", santali_emails)

# 5. Mobile Number Extraction (Santali and English Digits)
mobile_pattern_santali = r'(?:\+91\s?)?(?:[\+\u1C50-\u1C59]{2}\s?)?[\u1C50-\u1C590-9]{5}\s?[\u1C50-\u1C590-9]{5}'
santali_mobile_numbers = re.findall(mobile_pattern_santali, santali_text)
print("\nSantali Mobile Numbers:", santali_mobile_numbers)
