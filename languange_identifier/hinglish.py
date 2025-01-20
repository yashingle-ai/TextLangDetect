import re
from nltk.tokenize import sent_tokenize, word_tokenize

# Sample Hinglish text
hinglish_text = """
Mera naam Rajesh Kumar hai. 
Email: rajesh123@gmail.com aur mera phone number hai +91 9876543210.
Main engineer hoon aur mujhe cricket khelna pasand hai. 
Aap mujhse kabhi bhi sampark kar sakte hain.
Mera naam Rohan hai, aur main Mumbai mein rehta hoon. Tum kahan rehte ho?
Main kal movie dekhne jaa raha hoon. Tum chaloge?

"""

# Define regex patterns
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
phone_pattern = r'\+?\d{1,4}[\s-]?\d{10}'

# Tokenize sentences
sentences = sent_tokenize(hinglish_text)
print("Hinglish Sentences:",sentences)
print(len(sentences))


# Extract words
print("\nHinglish Words:")
words=[]
for sentence in sentences:
    word= word_tokenize(sentence)
    words.extend(word)
print(words)
    

# Find emails
emails = re.findall(email_pattern, hinglish_text)
print("\nEmails Found:")
print(emails)

# Find phone numbers
phones = re.findall(phone_pattern, hinglish_text)
print("\nPhone Numbers Found:")
print(phones)
