#here we give input as string of paragraph of containing some text about dogri language 
#and this code retrun some information like email,mobile number, sentences ,words with the help of re and nltk module 

import re
from nltk.tokenize import word_tokenize, sent_tokenize

# Dogri text
dogri_text = '''
𑠩𑠬𑠤𑠳 𑠢𑠝𑠯𑠊𑠹𑠋 𑠢𑠴𑠪𑠹𑠢𑠬 𑠙𑠳 𑠀𑠜𑠭𑠊𑠬𑠤𑠳𑠷 𑠛𑠳 𑠠𑠭𑠧𑠳 𑠏 𑠑𑠝𑠢𑠴 𑠚𑠢𑠬𑠷 𑠩𑠯𑠙𑠴𑠷𑠙𑠤 𑠙𑠳 𑠠𑠤𑠵𑠠𑠤 𑠝। 𑠄'𑠝𑠳𑠷𑠌𑠮 𑠠𑠯𑠛𑠹𑠜𑠮 𑠙𑠳 𑠑𑠢𑠮𑠤𑠴 𑠛𑠮 𑠛𑠳𑠝 𑠚𑠹𑠪𑠵𑠃 𑠇 𑠙𑠳 𑠄'𑠝𑠳𑠷𑠌𑠮 𑠁𑠞𑠰𑠷-𑠠𑠭𑠏𑠹𑠏𑠳𑠷 𑠡𑠬𑠃𑠏𑠬𑠤𑠳 𑠛𑠳 𑠡𑠬𑠦𑠴 𑠊𑠝𑠹𑠝𑠴 𑠠𑠹𑠣𑠪𑠬𑠤 𑠊𑠤𑠝𑠬 𑠥𑠵𑠫𑠛𑠬 𑠇।
𑠩𑠬𑠤𑠳 𑠢𑠝𑠯𑠊𑠹𑠋 𑠢𑠴𑠪𑠹𑠢𑠬 𑠙𑠳 𑠀𑠜𑠭𑠊𑠬𑠤𑠳𑠷 𑠛𑠳 𑠠𑠭𑠧𑠳 𑠏 𑠑𑠝𑠢𑠴 𑠚𑠢𑠬𑠷 𑠩𑠯𑠙𑠴𑠷𑠙𑠤 𑠙𑠳 𑠠𑠤𑠵𑠠𑠤 𑠝। 

𑠄𑠞𑠩𑠬𑠏𑠮𑠌𑠥𑠢𑠳𑠢𑠤 𑠠𑠯𑠛𑠹𑠜𑠮 𑠑𑠢𑠮𑠤𑠴 dogri.example@डोग्रीmail.com 𑠛𑠮 𑠛𑠳𑠝 𑠚𑠹𑠪𑠵𑠃। 

𑠄𑠥𑠳𑠝𑠰𑠬 𑠞𑠝𑠤𑠜𑠬 𑠚𑠤𑠳 +91 12345 67890 𑠞𑠭𑠚𑠤𑠳𑠝। 
dogri_text@example.com is also a valid email for testing purposes.

'''

# Unicode range for Dogri language
dogri_unicode_block = r'[\u11680-\u116CF]'
dogri_digit_block = r'[\u0966-\u096F]'

# 1. Word Tokenization
#when we use re moudule this is not correclty capture the all dogri words 
# dogri_word_pattern = rf'{dogri_unicode_block}+'
# dogri_words = re.findall(dogri_word_pattern, dogri_text)
# print("Dogri Words:", dogri_words)

dogri_words=wordtokenzie(dogri_text)
print("Dogri Words:", dogri_words)

# 2. Sentence Tokenization (Sentences ending with । or ! or ?)
dogri_sentence_pattern = rf'{dogri_unicode_block}[\s\u11680-\u116CF]*[।!?]'
dogri_sentences = re.findall(dogri_sentence_pattern, dogri_text)
print("\nDogri Sentences:", dogri_sentences)

# 2.1 NLTK-based Sentence Tokenization (fallback if specific sentence pattern fails)
dogri_sentences_nltk = sent_tokenize(dogri_text)  # Adjust NLTK tokenizers if necessary
print("\nDogri Sentences (NLTK):", dogri_sentences_nltk)

# 3. Paragraph Tokenization
dogri_paragraphs = dogri_text.split("\n\n")
print("\nDogri Paragraphs:")
for para in dogri_paragraphs:
    print(para)

# 4. Email Extraction (Dogri + English Emails)
email_pattern_dogri = rf'[{dogri_unicode_block}a-zA-Z0-9._%+-]+@[{dogri_unicode_block}a-zA-Z0-9.-]+\.[{dogri_unicode_block}a-zA-Z]{{2,}}'
dogri_emails = re.findall(email_pattern_dogri, dogri_text)
print("\nDogri Emails:", dogri_emails)

# 5. Mobile Number Extraction (Dogri and English Digits)
dogri_mobile_pattern = rf'(?:\+91\s?)?(?:[{dogri_digit_block}0-9]{{2}}\s?)?[{dogri_digit_block}0-9]{{5}}\s?[{dogri_digit_block}0-9]{{5}}'
dogri_mobile_numbers = re.findall(dogri_mobile_pattern, dogri_text)
print("\nDogri Mobile Numbers:", dogri_mobile_numbers)

