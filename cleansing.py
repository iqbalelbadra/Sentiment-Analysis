import pandas as pd
import re
 
def processing_text(input_text):
    text = input_text.lower()
    text = re.sub(r'[^\w\s]', '', text) # hapus semua punctuation (tanda baca)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','', text) #buang link
    text = text.replace("rt","")
    text = text.replace("user","")
    text = text.replace("\n","")
    text = text.replace("url","")
    text = text.strip()
    return text