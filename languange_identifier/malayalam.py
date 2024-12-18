#for malyalam language 
import re,nltk 
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from re import findall,U,search,findall
malyalam_text="""കേരളം, ഇന്ത്യയുടെ തെക്കന്‍ അറ്റത്ത് സ്ഥിതിചെയ്യുന്ന ഒരു മനോഹരമായ സംസ്ഥാനമാണ്. സമുദ്രത്തിന്റെ നടുവില്‍ ഒരു ഹരിത ഭൂമിയാണ് കേരളം. കേരളം 'ഗോദ്സ് ഓണ്‍ കണ്‍ട്രി' എന്ന പേരിലാണ് ലോകമെമ്പാടും പ്രസിദ്ധം. 

കേരളത്തിലെ പ്രധാന ആകര്‍ഷണങ്ങളില്‍ ഒന്നാണ് ആലപ്പുഴയുടെ ഇടുക്കി കായലുകള്‍. പ്രകൃതിയുടെ അതുല്യസൗന്ദര്യം കാണാനാകുന്ന ഈ സ്ഥലം രാജ്യത്തിന്റെ പ്രധാന ടൂറിസ്റ്റ് കേന്ദ്രങ്ങളിലൊന്നാണ്. പച്ചപ്പും നദികളും ഒരുമിച്ചു ചേര്‍ന്നൊരു അതിസുന്ദരമായ ഭൂമിയാണ് ആലപ്പുഴ.

കേരളത്തിന്റെ പാചകശൈലിയും അതിന്റെ സ്വാദിഷ്ഠമായ ഭക്ഷണങ്ങളും ലോകപ്രശസ്തമാണ്. സാധാരണയായി, നെയ്ച്ചോറ്, കറി, സാദ്യം എന്നിവയാണ് പ്രധാന വിഭവങ്ങള്‍. വാഴയിലയില്‍ വിളമ്പുന്ന സദ്യ കേരളത്തിന്റെ സവിശേഷതയാണ്. 

വിദ്യാഭ്യാസത്തില്‍ മുന്നേറുന്ന സംസ്ഥാനമായ കേരളം അക്ഷരമാലയുടെ മണ്ണായാണ് അറിയപ്പെടുന്നത്. കേരളത്തിലെ ജനസംഖ്യയിലെ ഭൂരിപക്ഷവും സാക്ഷരരാണ്. വനിതകളുടെ വിദ്യാഭ്യാസ നിരക്കിലും കേരളം രാജ്യത്തേതില്‍ മുന്‍പിലാണ്.

കേരളത്തിലെ ഉത്സവങ്ങളും സംസ്കാരവും അതിവിശേഷങ്ങളാണ്. ഓണം, വിഷു, തിരുവാതിര തുടങ്ങിയ ഉത്സവങ്ങള്‍ കുടുംബങ്ങളില്‍ ഒരുമയും സന്തോഷവും പകരുന്നു. കേരളം പ്രകൃതിയും പാരമ്പര്യവും ഒരുമിച്ചു ചേര്‍ന്ന ഒരു സമൃദ്ധമായ ഭൂമിയാണ്.

മഞ്ജു.വാരിയർ123@ഗ്മെയിൽ.com +൯൧ ൯൮൫൬൭ ൧൨൩൪൫"""
malyalam_text_pattern=r'[\u0D00 - \u0D7F]+'
#for word tokenisation 
# malyalam_words=word_tokenize(malyalam_text)
# print(malyalam_words)


# Pattern to match Malayalam text
malayalam_pattern = r'[\u0D00-\u0D7F]+'

# Find all Malayalam words
words = re.findall(malayalam_pattern,malyalam_text)

# print("Malayalam words:", words)

#for malayam sentences 
malylam_paragraph=malyalam_text.split("\n\n")
malayalam_sentence_list=[]
for para in malylam_paragraph:
   malayalam_sentence_pattern=r'[\u0D00-\u0D7F\s]+[.]'
   malayalam_sentence=re.findall(malayalam_sentence_pattern,para)
   malayalam_sentence_list.append(malayalam_sentence)
print(malayalam_sentence_list)

#for malyalam email
email_pattern_malyalam = r'[\u0D00-\u0D7Fa-zA-Z0-9._%+-]+@[\u0D00-\u0D7Fa-zA-Z0-9._-]+\.[\u0D00-\u0D7Fa-zA-Z0-9._]{2,}'
malyalam_email=findall(email_pattern_malyalam,malyalam_text)
print(malyalam_email)

#for malyalam mobile number 
mobile_pattern_malyalam=r'(?:\+[\u0D66-\u0D6F]{2})?\s?[\u0D66-\u0D6F]{5}\s?[\u0D66-\u0D6F]{5}|(?:\+91\s?)?[\u0D66-\u0D6F]{10}'
malayalam_mobile=findall(mobile_pattern_malyalam,malyalam_text)
print(malayalam_mobile)
