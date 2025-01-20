import re
from nltk.tokenize import sent_tokenize, RegexpTokenizer

# Braj Text Example
braj_text = """
राधे राधे! ब्रजभाषा के रसिक लोगन के जय हो। 
मोरे ईमेल: braj.rasik@gmail.com 
फोन नंबर: +91 12345 67890
कहे मोहन, "बृज भूमि के रज माटी चंदन समान।" 
श्रीकृष्ण गोपाल! ब्रज भूमि के सुंदरता जग में अद्भुत मानी गई। 
यमुना किनारे के कुंजवनन में ग्वाल-बाल संग खेलत नंदलाल के रूप के दर्शन करब, तो मन आनंदित हो जाइ। 
गोपियां मोहन के मुरली के मधुर स्वर सुनि, सब कुछ भुला देत रहीं। 

ब्रज के रज माटी, जेहके स्पर्श मात्र से आत्मा पवित्र हो जाइ। 
गौशाला के पास, ग्वाल-बाल दूध-दही लेके बाजार चलत रहे। 
कहे सुदामा, "हे गोविंद! तेरो प्रेम अछोर ह।"
"""

# Define Braj Unicode block
braj_unicode_block = r'[\u0900-\u097F\u200C\u200D\w]+'  # For Devanagari script (used in Braj) and alphanumeric characters

# Sentence Splitting
print("Sentences in Braj Text:")
sentences = sent_tokenize(braj_text)  # Sentence tokenization
for sentence in sentences:
    print(sentence)

# Word Tokenization
print("\nWord Tokenization for Each Sentence:")
regexp_tokenizer = RegexpTokenizer(braj_unicode_block)  # Tokenize words based on Braj and Devanagari characters
for sentence in sentences:
    words = regexp_tokenizer.tokenize(sentence)
    print(f"Words in sentence: {sentence}")
    print(words)

# Email Detection
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
emails = re.findall(email_pattern, braj_text)
print("\nEmails Found in the Text:")
print(emails)

# Mobile Number Detection
mobile_pattern = r'\+?[0-9]{1,3}[\s-]?[0-9]{5}[\s-]?[0-9]{5}'
mobile_numbers = re.findall(mobile_pattern, braj_text)
print("\nMobile Numbers Found in the Text:")
print(mobile_numbers)
