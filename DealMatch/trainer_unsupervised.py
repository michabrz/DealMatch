import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from DealMatch.data_unsupervised import get_targets_data, get_investors_data, get_matching_keys, clean_targets, clean_investors

class Trainer():
    
    def __init__(self, X, Y):
    
        self.pipeline_targets = None
        self.pipeline_investors = None
        self.X = X
        self.Y = Y
   
    
    def set_pipeline_targets(self):
        
        num_transformer = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
                            ('scaler', RobustScaler())])
        
        cat_transformer = Pipeline([('imputer',
                             SimpleImputer(missing_values=np.nan,
                                           strategy='constant',
                                           fill_value='no_region'))
        ])
        
        
        
        preprocessor = ColumnTransformer([
            ('num_tr', num_transformer, ['target_ebit','target_ebitda','target_revenue']),
            ('cat_tr', cat_transformer, ['deal_type_name', 'country_name', 'region_name', 'sector_name']),
            ('tfidf',TfidfVectorizer(),['strs'])
        ], remainder='drop')
        

        self.pipeline_targets = Pipeline([
                            ('preproc',preprocessor),
                            ('pca',PCA(n_components=0.95)),
                            ('NN',NearestNeighbors(n_neighbors=10))
        ])
        
        
    def run_targets(self):
        
        self.set_pipeline_targets()
        self.pipeline_targets.fit(self.X) 
    
    def save_model_targets(self):
        
        joblib.dump(self.pipeline_targets,'model_targets.joblib')
        
        
        
    def set_pipeline_investors(self):
        
        preprocessor = ColumnTransformer([
                ('tfidf',TfidfVectorizer(),['name_de'])
        ], remainder='drop')
        
        self.pipeline_investors = Pipeline([
                            ('preproc',preprocessor),
                            ('pca',PCA(n_components=0.95)),
                            ('NN',NearestNeighbors(n_neighbors=10))
        ])        
    
    def run_investors(self):
        
        self.set_pipeline_investors()
        self.pipeline_investors.fit(self.Y)     
    
        
    def save_model_investors(self):
        
        joblib.dump(self.pipeline_investors,'model_investors.joblib')        

    
        
if __name__ == "__main__":
    
    df_targets = get_targets_data()
    df_investors = get_investors_data()
    df_investor_keys = get_matching_keys()
    df_targets_clean = clean_targets(df_targets)         
    df_investors_clean = clean_investors(df_investors,df_investor_keys)
    X = df_targets_clean
    Y = df_investors_clean
    trainer = Trainer(X,Y)
    trainer.run_targets()
    trainer.run_investors()
    trainer.save_model_targets()
    trainer.save_model_investors()
    
    
    