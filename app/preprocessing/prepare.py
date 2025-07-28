from app.preprocessing.process_text import tokenize_text
import pandas as pd

def prepare_headline(headlines: pd.Series) -> pd.Series:
    tokenized_headlines = headlines.apply(tokenize_text)
    
    return headlines