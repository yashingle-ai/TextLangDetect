import re

# Example Awadhi text
awadhi_text = '''
हमार गाँव के लोग बहुत अच्छा हैं । सब लोग आपस में मिलजुल के रहते हैं ।
हमरे यहाँ के लोग धार्मिक हवन, पूजा पाठ आओर संस्कार के बारे में बहुत जागरूक हैं। 
मोबाइल: +९१ ९८५६७ १२३४५ 9876543210  
ईमेल: rajesh.kumar@gmail.com  राजेश.कुमार@जीमेल.कॉम  
हमरे यहाँ के प्रमुख त्योहार होली आओर दीपावली हैं।
हमरे गाँव में बहुत अच्छा वातावरण है। लोग आपस में मिलजुल कर रहते हैं। हर आदमी के पास अपनी छोटी सी ज़मीन है, जहाँ वह खेती करता है। सब लोग मेहनत से काम करते हैं।
उनका प्रिय त्योहार होली और दीवाली है। इन दोनों त्योहारों में गाँव के लोग बहुत धूमधाम से मिलकर उत्सव मनाते हैं।
ईमेल: rajesh.kumar@जीमेल.कॉम
मोबाइल: +९१ ९८५६७ १२३४५
'''

# 1. Awadhi Sentence Pattern (Sentence ending with '।', '!', '?', or a newline followed by a punctuation)
awadhi_sentence_pattern = r'[^।!?]*[।!?]'

# 2. Word Tokenization (Using regex to capture words in Devanagari script)
awadhi_word_pattern = r'[\u0900-\u097F]+'

# 3. Email Pattern (Supports both Hindi and English email addresses)
awadhi_email_pattern = r'[\u0900-\u097Fa-zA-Z0-9._%+-]+@[\u0900-\u097Fa-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# 4. Mobile Number Pattern (Supports Hindi and English digits)
awadhi_mobile_pattern = r'(?:\+91\s?)?(?:[\+\u0966-\u096F]{2}\s?)?[\u0966-\u096F0-9]{5}\s?[\u0966-\u096F0-9]{5}'

# Extracting sentences
awadhi_sentences = re.findall(awadhi_sentence_pattern, awadhi_text)
print("Awadhi Sentences:", awadhi_sentences)
print(len(awadhi_sentences))

# Extracting words
awadhi_words = re.findall(awadhi_word_pattern, awadhi_text)
print("\nAwadhi Words:", awadhi_words)

# Extracting emails
awadhi_emails = re.findall(awadhi_email_pattern, awadhi_text)
print("\nAwadhi Emails:", awadhi_emails)

# Extracting mobile numbers
awadhi_mobile_numbers = re.findall(awadhi_mobile_pattern, awadhi_text)
print("\nAwadhi Mobile Numbers:", awadhi_mobile_numbers)
