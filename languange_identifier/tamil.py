# for tamil language
import re,nltk 
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from re import findall,U,search,findall

tamil_text="""தமிழ் மொழி உலகின் மிகப்பெரிய பண்பாட்டுச் சின்னங்களில் ஒன்றாக விளங்குகிறது. பழமையான இலக்கியங்கள், வரலாறுகள் மற்றும் கலாச்சாரங்களின் தொகுப்பாக தமிழ் மொழி இன்று உலகம் முழுவதும் பேசப்படுகிறது. தமிழ் எழுத்துமுறை அதன் தனித்துவத்தால் உலகின் பல்வேறு நாடுகளில் புகழ் பெற்றது.

தமிழகத்தின் இயற்கை வளங்கள் அதன் கலாச்சார வளர்ச்சிக்கு மிக்க ஒதுக்கினை வழங்கின. பண்டைய தமிழர்கள் கலை, அறிவியல், வேளாண்மை மற்றும் சமூகம் ஆகியவற்றில் மேம்பட்டது. சங்க காலத்தில் எழுதப்பட்ட இலக்கியங்கள், தமிழர்களின் அறிவு மற்றும் பண்பாட்டின் உயர்தரத்தை காட்டுகின்றன.

திருவள்ளுவர் இயற்றிய திருக்குறள் தமிழர்களின் வாழ்வியல் நெறிகளை சுட்டிக்காட்டுகிறது. இது மனித வாழ்வின் ஒவ்வொரு பகுதிக்கும் சரியான வழிகாட்டி ஆகும். அதுவே தற்போது பல மொழிகளில் மொழிபெயர்க்கப்பட்டு உலகளவில் மதிப்புமிக்க நூலாக மாறியுள்ளது. திருக்குறளின் அர்த்தம் மற்றும் அறிவுரை இன்று மனித வாழ்வின் அடிப்படையாக பயன்படுத்தப்படுகிறது ௯௮௫௬௭ ௨௩௪௫௬.

மிகவும் வரலாற்று முக்கியத்துவம் வாய்ந்தது தமிழ் மொழி. அதனுடன் தொடர்புடைய வணிகர்களின் தகவல்: நாமம்: ராஜேஷ் குமார், மின்னஞ்சல்: ராஜேஷ்.குமார்@மெயில்.காம், கைபேசி: +௯௧ ௯௯௬௮௭ ௭௫௪௩௨. இது நமது தமிழ் பண்பாட்டின் மகத்துவத்தை வெளிப்படுத்துகிறது. கல்வி, கலாச்சாரம் மற்றும் தொழில் முன்னேற்றத்தில் தமிழர்களின் சாதனைகள் இன்றும் தொடர்கின்றன. அடுத்த தலைமுறையினருக்கு இந்த பண்பாட்டின் முக்கியத்துவத்தை புரியவைத்தல் அவசியமாகிறது.
"""#last paragraph contain email and mobile number in tamil language 

tamil_text_pattern=r'[\u0B80-\u0BFF]+'
#for word tokenisation 
tamil_words=findall(tamil_text_pattern,tamil_text)
print("the words from the tamil text is :- ",tamil_words)

#for paragraph split in the given input 
tamil_paragraph=tamil_text.split("\n\n")
for para in tamil_paragraph:
    tamil_sentence_pattern=r'[\u0B80-\u0BFF\s]+[.!?]'
    tamil_sentence=findall(tamil_sentence_pattern,tamil_text)

#tamil_email_pattern
email_pattern_tamil = r'[\u0B80-\u0BFFa-zA-Z0-9._%+-]+@[\u0B80-\u0BFFa-zA-Z0-9._-]+\.[\u0B80-\u0BFFa-zA-Z0-9._]{2,}'
tamil_email=findall(email_pattern_tamil,tamil_text)
print("extracted tamil email is :-",tamil_email)

#for tamil_mobile_number 
tamil_mobile_number_pattern=r'(?:\+[\u0BE6-\u0BEF]{2})?\s?[\u0BE6-\u0BEF]{5}\s?[\u0BE6-\u0BEF]{5}' 
tamil_mobile_number=findall(tamil_mobile_number_pattern,tamil_text)
print("tamil mobile number is :- ",tamil_mobile_number)