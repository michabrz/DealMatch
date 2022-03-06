import pandas as pd
import numpy as np
import string 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

def get_targets_data():
    
    targets = pd.read_excel('../raw_data/deals_raw.xlsx')
    
    return targets

def get_investors_data():

    investors = pd.read_excel('../raw_data/invest_profile_keywords.xlsx')
    
    return investors

def get_matching_keys():

    key_match = pd.read_excel('../raw_data/new_keywords.xlsx')
    
    return key_match

def get_matching_table():

    matching_table = pd.read_excel('../raw_data/matching_table.xlsx')
    
    return matching_table

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text    

def clean_targets(targets):

    g = targets.groupby('deal_id')['sector_name', 'subsector_name',
                                'keyword'].apply(lambda x: list(np.unique(x)))
    g = pd.DataFrame(g, columns=['keywords_all'])
    g['strs'] = [', '.join(map(str, l)) for l in g['keywords_all']]
    g.drop(columns="keywords_all", inplace=True)
    g.reset_index(inplace=True)

    targets = pd.merge(targets, g, on="deal_id", how="left")
    targets.drop(columns=['subsector_name', 'keyword'], inplace=True)

    targets.dropna(subset='target_name',inplace=True)
    targets['strs'] = targets['strs'].str.replace(',',' ')
    targets.reset_index(inplace=True)

    targets['strs'] = targets['strs'].apply(remove_punctuations()) 
    targets['strs'] = targets['strs'].apply(lambda x: x.lower())

    stop_words = set(stopwords.words('german')) 

    for name_de in targets['strs']:    
        word_tokens = word_tokenize(name_de) 
        name_de = [w for w in word_tokens if not w in stop_words] 
        

    return targets

def clean_investors(investors,key_match):
    
    investors['name_de'] = investors['name_de'].replace(dict(zip(key_match.name_de,key_match.new_keyword)))
    investors_small = investors[['name','name_de']]
    investors_concat = investors_small.astype(str).groupby('name').agg({'name_de':', '.join})
        
    investors_concat1 = investors_concat['name_de'].str.replace('nan','').reset_index()
    investors_concat1['name_de'] = investors_concat1['name_de'].apply(remove_punctuations())        
    investors_concat1['name_de'].replace(r'^\s*$',np.nan,regex=True,inplace=True)
    investors_concat1 = investors_concat1.dropna().reset_index(inplace=True)
    investors_concat1['name_de'] = investors_concat1['name_de'].apply(lambda x: x.lower())

    stop_words = set(stopwords.words('german')) 

    for name_de in investors_concat1['name_de']:    
        word_tokens = word_tokenize(name_de) 
        name_de = [w for w in word_tokens if not w in stop_words] 
        
    return investors_concat1