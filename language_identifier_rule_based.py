import re

class LanguageIdentifier:
    '''
    Language identification for texts.
    '''
    def __init__(self):
        self.language_patterns = {
            "Hindi": r'[\u0900-\u097F]+',                  
            "Bengali": r'[\u0980-\u09FF]+',    
            "Punjabi": r'[\u0A00-\u0A7F]+',    
            "Urdu": r'[\u0600-\u06FF]+',  
            "odia":r'[\u0B00-\u0B7F]+' ,
            "marathi":r'[\u0900-\u097F]+',
            "malayalam":r'[\u0D00-\u0D7F]+',
            "tamil":r'[\u0B80-\u0BFF]+',
            "telgu":r'[\u0C00-\u0C7F]+',
            "gujrati":r'[\u0A80-\u0AFF]+',
            "kannada":r'[\u0C80-\u0CFF]+',
            "assamese":r'[\u0980-\u09FF]+',
            "konkani":r'[\u0900-\u097F]+',
            "english":r'[a-zA-Z0-9]+',
        }
    def language_identifier(self,text):
        print("length of given text is ",len(text))
        identify_count={}
        identify_percent={}
        text_language=0        # for storing length of perticular 
        if len(text)==0 :
            print("you are not entering anything for identify ")
            
        for language ,pattern in self.language_patterns.items():
            words =re.findall(pattern,text)
            i=0 
            l=0
            for i in words :
                l=l+len(i)
            identify_count[language]=l
            
        for language, acc in identify_count.items():
            text_language=acc/len(text)
            identify_percent[language]=text_language
        # print("identify perentage ",identify_percent.items())
        
        # for key , value in identify_percent.items():
        #     if value>0.8:
        #         print(f"given text is highly releted from a single language {value*100}% with language ")
        #         return key
        #     if value>0.6:
        #         print(f"given text is highly releted from a single language {value*100}% with language ")
        #         return key
        #     if value>0.5:
        #         print(f"given text is highly releted from a single language {value*100}% with language ")
            
        # print("given text is not properly releted from a single language , in the gien text user uses multiple languages ")
        return identify_count
    
            
            
            
       
text1 = LanguageIdentifier()

m=text1.language_identifier("my name is yash ingle and i dont want to do anthing from your side  ಮಾನವಜಾತಿ ಎದುರಿಸುತ್ತಿರುವ ಪ್ರಮುಖ ಸಮಸ್ಯೆಗಳಿಗೆ ಪರಿಹಾರವನ್ನು ಶಿಕ್ಷಣದ ಮೂಲಕವೇ ಕಂಡುಹಿಡಿಯಬಹುದಾಗಿದೆ. तो राजा हमारी गल‍ियों में नहीं आते")
print(m)

    
            
            
            
        

    