import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Braj Text
braj_text = '''
श्रीकृष्ण सदा ब्रजवासियों के प्रिय रहें। ब्रजभूमि सदा से ही पवित्र और सुंदर रही है। 
हर भक्त को ब्रजभूमि की यात्रा करनी चाहिए। 
श्रीकृष्ण गोपाल! ब्रज भूमि के सुंदरता जग में अद्भुत मानी गई। 
यमुना किनारे के कुंजवनन में ग्वाल-बाल संग खेलत नंदलाल के रूप के दर्शन करब, तो मन आनंदित हो जाइ। 
गोपियां मोहन के मुरली के मधुर स्वर सुनि, सब कुछ भुला देत रहीं। 

ब्रज के रज माटी, जेहके स्पर्श मात्र से आत्मा पवित्र हो जाइ। 
गौशाला के पास, ग्वाल-बाल दूध-दही लेके बाजार चलत रहे। 
कहे सुदामा, "हे गोविंद! तेरो प्रेम अछोर ह।"
Email: kanha.braj@gmail.com  राधा.श्री@ब्रज.कॉम  
Mobile: +९१ ९८५६७ १२३४५  9876543210  
ब्रजभूमि में गोवर्धन पर्वत और यमुना नदी का विशेष महत्व है। +91 9876543210
'''

# Braj Unicode Block and Digit Block
braj_unicode_block = r'[\u0900-\u097F]' 
braj_digit_block = r'[\u0966-\u096F]'   

# 1. Word Tokenization
braj_word_pattern = r'[\u0900-\u097F]+' 
braj_words = re.findall(braj_word_pattern, braj_text)
print("Braj Words:", braj_words)

# 2. Sentence Tokenization (Sentences ending with ।, !, or ?)
braj_sentence_pattern = r'[\u0900-\u097F\s]+[।!?]'
braj_sentences = re.findall(braj_sentence_pattern, braj_text)
print("\nBraj Sentences (Regex):", braj_sentences)


print(len(braj_sentences))

# 3. Paragraph Tokenization
braj_paragraphs = braj_text.split("\n\n")
print("\nBraj Paragraphs:")
for para in braj_paragraphs:
    print(para)

# 4. Email Extraction (Braj + English Emails)
email_pattern_braj = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[\u0900-\u097Fa-zA-Z]{2,}'
braj_emails = re.findall(email_pattern_braj, braj_text)
print("\nBraj Emails:", braj_emails)

# 5. Mobile Number Extraction (Braj and English Digits)
braj_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'
braj_mobile_numbers = re.findall(braj_mobile_pattern, braj_text)
print("\nBraj Mobile Numbers:", braj_mobile_numbers)
