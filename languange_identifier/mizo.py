import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Sample Mizo text input
mizo_text = """
Hming: Rajesh Kumar
Email: rajesh.kumar@domain.com
Mobile: +91 12345 67890

Kan chhiar chu hming pawimawh a ni. Inthlahna thil a hmingchhiat a, a hming pawimawh a ni. 
A hnuaiah, chhanna chhuak tawh, a chhanna lo tih chu a hming pawimawh a lo hman chhuak a.
"""

# Regular expression to match Mizo mobile numbers (assuming the format is similar to typical formats)
mizo_mobile_regex = r'\+?[0-9]{1,3}[\s]?[0-9]{5}[\s]?[0-9]{5}'  # Pattern for mobile numbers

# Regular expression to match Mizo email addresses
mizo_email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Standard email regex

# Function to extract mobile numbers
def extract_mobile_numbers(text):
    return re.findall(mizo_mobile_regex, text)

# Function to extract email addresses
def extract_email_addresses(text):
    return re.findall(mizo_email_regex, text)

# Function to tokenize the text into sentences and words
def process_text(text):
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    words = word_tokenize(text)  # Tokenize text into words
    return sentences, words

# Extract sentences and words
sentences, words = process_text(mizo_text)

# Print sentences and words
print("Sentences:",sentences)
print(len(sentences))


# Extract and print mobile numbers
mobile_numbers = extract_mobile_numbers(mizo_text)
print("\nMobile Numbers:")
for mobile in mobile_numbers:
    print(mobile)

# Extract and print email addresses
email_addresses = extract_email_addresses(mizo_text)
print("\nEmail Addresses:")
for email in email_addresses:
    print(email)
