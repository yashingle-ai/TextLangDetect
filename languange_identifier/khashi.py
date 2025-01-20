import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Sample Khasi text input
khasi_text = """
Ka Hming: Rajesh Kumar
Email: rajesh.kumar@domain.com
Mobile: +91 12345 67890

Ka Hming: John L. Syiem
Email: john.syiem@khasi.com
Mobile: +91 98765 43210

Nga iakren bad ka pyrthei. Ka jingiakren ka long kaba lyngkot, hynrei ka pynpaw ia ka jingsngewthuh kaba khraw. U jngohsan u la pynlong ia ka jingpynkhreh jong u. Ha ka por, u la ioh ka bynta ka jong ka por ba u kwah ban pyndep.

Ka jingim ka dei ka mawphlang ka ba sngewtynnad, hynrei ka dei ka kaba la pynkhreh bha. Ka jingshisha ba la kyntait, ka pynlong ia ka jingpyrkhat ba kynmaw ha ka por jong u. Ka rynjat ka long kaba kyntait ia ka jingiakren.

Mobile: +91 87654 32100
Email: john.syiem2@khasi.com

Ngi don ka jingsngewthuh ba ka pyrthei ka long ka bynta ka ba shisha, hynrei ka jingiakren ba la kynmaw ha ka por jong u ka pyrkhat ba pynlong ia ka jingshisha kaba pynkhreh.

Ka bynta jong ka jingiakren ka dei ka bynta kaba kyntait.

Nga thoh ka jingthoh ha ka por. Ka rynjat ka pynpaw ia ka jingsngewthuh kaba donkam. Ka rynjat ka jingïaid beit. 
Ka phareng ka kyntait ia ka jingiakren. Kaei ka jingiakren ka kynjoh aiu ia ka. 
"""

# Regular expression to match Khasi mobile numbers (assuming the format is similar to typical formats)
khasi_mobile_regex = r'\+?[0-9]{1,3}[\s]?[0-9]{5}[\s]?[0-9]{5}'  # Pattern for mobile numbers

# Regular expression to match Khasi email addresses
khasi_email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Standard email regex

# Function to extract mobile numbers
def extract_mobile_numbers(text):
    return re.findall(khasi_mobile_regex, text)

# Function to extract email addresses
def extract_email_addresses(text):
    return re.findall(khasi_email_regex, text)

# Function to tokenize the text into sentences and words
def process_text(text):
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    words = word_tokenize(text)  # Tokenize text into words
    return sentences, words

# Extract sentences and words
sentences, words = process_text(khasi_text)

# Print sentences and words
print("Sentences:",sentences)
print(len(sentences))

print("\nWords:",words)

# Extract and print mobile numbers
mobile_numbers = extract_mobile_numbers(khasi_text)
print("\nMobile Numbers:")
for mobile in mobile_numbers:
    print(mobile)

# Extract and print email addresses
email_addresses = extract_email_addresses(khasi_text)
print("\nEmail Addresses:")
for email in email_addresses:
    print(email)
