import re

# Sindhi (snd) Text Example
sindhi_text = '''
آپ ڪيئن آهيو؟ مان ٺيڪ آهيان. اسان گڏجي ڪم ڪري سگهون ٿا؟ .مونکي خوشي ٿيندي جيڪڏهن اوه توهان سان ڳالهايان! مونکي خوشي ٿيندي جيڪڏهن اوه توهان سان ڳالهايان?
توهان جو اي ميل: rajesh.kumar@gmail.com ۽ موبائل: +91 98765 43210 آهي.
مونکي خوشي ٿيندي جيڪڏهن اوه توهان سان ڳالهايان! 
موبائل نمبر: 91234 56789.
'''

# 1. Sentence Detection using punctuation marks (full stop, question mark, exclamation mark)
sindhi_sentences = re.split(r'(?<=۔|\?|!)\s*', sindhi_text.strip())  # split at full stop (۔), question mark (?) and exclamation mark (!)
sindhi_sentences = [sentence.strip() for sentence in sindhi_sentences if sentence.strip()]  # Remove empty strings

# 2. Word Detection
sindhi_words = re.findall(r'\b\w+\b', sindhi_text)

# 3. Email Detection
sindhi_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', sindhi_text)

# 4. Mobile Number Detection
sindhi_mobile_numbers = re.findall(r'\+?\d{1,2}\s?\(?\d{2,4}\)?\s?\d{7,10}', sindhi_text)

# Output the results
print("\nSindhi Sentences:", sindhi_sentences)
print("\nSindhi Words:", sindhi_words)
print("\nSindhi Emails:", sindhi_emails)
print("\nSindhi Mobile Numbers:", sindhi_mobile_numbers)


