import re
import string

class TweetProcessor:
    """Process tweets text"""
    
    def __init__(self):
        pass
    
    def process_text(self,tweets):
        """processes tweet text"""
  
        #remove html
        tweets = tweets.apply(
            lambda x: re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+","",x))
    
        #remove punctuation
        tweets = tweets.apply(
            lambda x: x.translate(str.maketrans('', '', string.digits + string.punctuation)).lower().strip())
        
        return tweets