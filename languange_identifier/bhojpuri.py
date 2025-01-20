import re
from nltk.tokenize import sent_tokenize, RegexpTokenizer

# Bhojpuri Text
bhojpuri_text = """
सबहिं के ओकर उचित सम्मान आओर मानव परिवार के सभे आदिमी के बराबरी के हक ही विश्व समुदाय के अजादी, न्याय आओर शांति के बुनियाद हवे।
मानवाधिकार के उल्लंघन हरदम अमानवीय काज के कारणो होखेला जा के चलते मानवता के अंत:करण दु:खी होखेला।
हमरा email ramkumar.bhojpuri@gmail.com पर contact करीं।
मोबाइल नंबर +91 9876543210 भी उपलब्ध बा। 
संयुक्त राष्ट्र के लोगिन आपन चार्टर में मौलिक मानवाधिकार के स्वीकार कइलन।
"""

# Define Bhojpuri Unicode block
bhojpuri_unicode_block = r'[\u0900-\u097F\u200C\u200D\w]+'  # For Devanagari script and standard alphanumeric characters

# Sentence Splitting
print("Sentences in Bhojpuri Text:")
sentences = sent_tokenize(bhojpuri_text)  # Sentence tokenization
for sentence in sentences:
    print(sentence)

# Word Tokenization
print("\nWord Tokenization for Each Sentence:")
regexp_tokenizer = RegexpTokenizer(bhojpuri_unicode_block)  # Tokenize based on Bhojpuri and words
for sentence in sentences:
    words = regexp_tokenizer.tokenize(sentence)
    print(f"Words in sentence: {sentence}")
    print(words)

# Email Detection
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
emails = re.findall(email_pattern, bhojpuri_text)
print("\nEmails Found in the Text:")
print(emails)

# Mobile Number Detection
mobile_pattern = r'\+?[0-9]{1,3}[\s-]?[0-9]{5}[\s-]?[0-9]{5}'
mobile_numbers = re.findall(mobile_pattern, bhojpuri_text)
print("\nMobile Numbers Found in the Text:")
print(mobile_numbers)
