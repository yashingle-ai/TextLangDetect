import re

# Example Bodo text (both Devanagari and Roman script versions)
bodo_text = '''
बोड़ो भाषा असम राज्य के प्रमुख भाषा है। बोड़ो भाषायें भारत के अन्य हिस्सों में भी बोली जाती है। 
हमार गाम में बहुते लोग बोड़ो भाषा बोलय। लोग सब मिलजुल के काम करत हैं। 
मोबाइल: +९१ ९८५६७ १२३४५ 9876543210  
ईमेल: rajesh.kumar@gmail.com  राजेश.कुमार@जीमेल.कॉम  
बोड़ो में होली और दीवाली के समय खूब धूमधाम से त्योहार मनाय जाए।
'''

# Sentence Tokenization Pattern (Ends with '।', '!', or '?')
bodo_sentence_pattern = r'[\u0900-\u097F\s]+[।!?]'  # For Devanagari
# For Roman Bodo sentences, we might use a simpler punctuation-based pattern.
roman_sentence_pattern = r'[^.!?]*[.!?]'  # For Roman Bodo

# Word Tokenization Pattern (Capture words in Devanagari script or Roman script)
bodo_word_pattern = r'[\u0900-\u097F]+'  # For Devanagari words
# For Roman Bodo text
roman_word_pattern = r'\b\w+\b'  # Words in Roman script

# Email Pattern (For both Devanagari and Roman)
bodo_email_pattern = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Mobile Number Pattern (Supports Hindi/Devanagari and English digits)
bodo_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'

# Extracting sentences
bodo_sentences_devanagari = re.findall(bodo_sentence_pattern, bodo_text)
print("Bodo Sentences (Devanagari):", bodo_sentences_devanagari)

# Extracting sentences for Roman Bodo text (you can use similar regex patterns)
bodo_sentences_roman = re.findall(roman_sentence_pattern, bodo_text)
print("\nBodo Sentences (Roman):", bodo_sentences_roman)

# Extracting words from Devanagari Bodo text
bodo_words_devanagari = re.findall(bodo_word_pattern, bodo_text)
print("\nBodo Words (Devanagari):", bodo_words_devanagari)

# Extracting words from Roman Bodo text
bodo_words_roman = re.findall(roman_word_pattern, bodo_text)
print("\nBodo Words (Roman):", bodo_words_roman)

# Extracting emails
bodo_emails = re.findall(bodo_email_pattern, bodo_text)
print("\nBodo Emails:", bodo_emails)

# Extracting mobile numbers
bodo_mobile_numbers = re.findall(bodo_mobile_pattern, bodo_text)
print("\nBodo Mobile Numbers:", bodo_mobile_numbers)
