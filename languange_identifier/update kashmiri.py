import re
from nltk.tokenize import sent_tokenize, word_tokenize

# Kashmiri text
kashmiri_text = '''
کٲشُر چھُ گژھنک ژٲتھ سہٚتھ متھے زٲن ٲسہ۔
کٔنمٲٹھُ کرنکُ چھس منزِ ٲسُن ؤچے منز مژھٕرتھ۔
email@example.com یہ چھُ ای میل پٲٹھ۔
+91 98765 43210 یے چھُ فون نمبر۔ 

𑆐𑆲𑆢𑆳𑆫𑆤𑆁 𑆦𑆳𑆚𑆫𑆤𑆳𑆁 𑆯𑆢𑆳 𑆨𑆫𑆳𑆩 𑆯𑆷𑆫𑆁 𑆡𑆲𑆬𑆁۔ 
𑆓𑆾𑆯𑆶 𑆡𑆲𑆐𑆁 𑆄𑆬𑆁 𑆩𑆤𑆱𑆭𑆳𑆫𑆴 𑆏𑆱𑆴𑆯𑆳𑆩۔
'''

# Unicode range for Kashmiri language (Perso-Arabic and Devanagari scripts)
perso_arabic_range = r'\u0600-\u06FF'
devanagari_range = r'\u0900-\u097F\u1C50-\u1C7F'
kashmiri_unicode_range = f'{perso_arabic_range}{devanagari_range}'
digit_range = r'\u0660-\u0669\u0966-\u096F'

# 1. Word Tokenization
word_pattern = rf'[{kashmiri_unicode_range}]+'
kashmiri_words = re.findall(word_pattern, kashmiri_text)
print("Kashmiri Words:", kashmiri_words)

# 2. Sentence Tokenization (Sentences ending with ۔ or ۔ or ! or ? or ।)
sentence_pattern = rf'[{kashmiri_unicode_range}][^{kashmiri_unicode_range}]*[۔!?۔]'
kashmiri_sentences = re.findall(sentence_pattern, kashmiri_text)
print("\nKashmiri Sentences:", kashmiri_sentences)
print("number of kashmiri sentences :-",len(kashmiri_sentences))

# 2.1 NLTK-based Sentence Tokenization (as a fallback)
kashmiri_sentences_nltk = sent_tokenize(kashmiri_text)  # Adjust NLTK tokenizers if needed
print("\nKashmiri Sentences (NLTK):", kashmiri_sentences_nltk)

# 3. Paragraph Tokenization
kashmiri_paragraphs = kashmiri_text.strip().split("\n\n")
print("\nKashmiri Paragraphs:")
for para in kashmiri_paragraphs:
    print(para)

# 4. Email Extraction (Supports English and Kashmiri characters)
email_pattern = rf'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}}'
kashmiri_emails = re.findall(email_pattern, kashmiri_text)
print("\nKashmiri Emails:", kashmiri_emails)

# 5. Mobile Number Extraction (Supports English and Kashmiri digits)
mobile_pattern = rf'(?:\+91\s?)?(?:[{digit_range}0-9]{{2}}\s?)?[{digit_range}0-9]{{5}}\s?[{digit_range}0-9]{{5}}'
kashmiri_mobile_numbers = re.findall(mobile_pattern, kashmiri_text)
print("\nKashmiri Mobile Numbers:", kashmiri_mobile_numbers)

# 6. NLTK Word Tokenization
words_nltk = word_tokenize(kashmiri_text)
print("\nNLTK Tokenized Words:", words_nltk)
