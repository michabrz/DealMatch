import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
import joblib
from sklearn.base import TransformerMixin
from DealMatch.custom_transformer import DenseTransformer
from DealMatch.data_unsupervised import get_targets_data, get_investors_data, get_matching_keys, clean_targets, clean_investors, get_matching_table

class Trainer():

    def __init__(self, X, Y):

        self.pipeline_targets = None
        self.pipeline_investors = None
        self.pipeline_data = None
        self.X = X
        self.Y = Y


    def preproc_targets_pipe(self):

        tfidf_features = 'strs'
        tfidf_transformer = Pipeline([('tfidf', TfidfVectorizer()), ('dense', DenseTransformer())])

        num_transformer = Pipeline([('imputer',
                                     SimpleImputer(missing_values=np.nan,
                                                   strategy='constant',
                                                   fill_value=0)),
                                    ('scaler', RobustScaler())])

        preproc = ColumnTransformer(transformers=[
            ('num_tr', num_transformer,
             ['target_ebit', 'target_ebitda', 'target_revenue']),
                  ('tfidf', tfidf_transformer, tfidf_features)
        ],
                                    remainder='drop')

        self.pipeline_targets = Pipeline([('preproc', preproc), ('pca', PCA(0.95))])

        self.pipeline_targets.fit(self.X)

        joblib.dump(self.pipeline_targets, 'pipeline.pkl')


    def clean_target_data(self):

        num_transformer = Pipeline([('imputer',
                                     SimpleImputer(missing_values=np.nan,
                                                   strategy='constant',
                                                   fill_value=0)),
                                    ('scaler', RobustScaler())])

        cat_transformer = Pipeline([('imputer',
                                     SimpleImputer(missing_values=np.nan,
                                                   strategy='constant',
                                                   fill_value='no_region'))])

        preprocessor = ColumnTransformer([
            ('num_tr', num_transformer,
             ['target_ebit', 'target_ebitda', 'target_revenue']),
            ('cat_tr', cat_transformer,
             ['deal_type_name', 'country_name', 'region_name', 'sector_name'])
        ],
                                         remainder='passthrough')

        self.pipeline_data = Pipeline([('preproc', preprocessor)])

        self.pipeline_data.fit(self.X)

        joblib.dump(self.pipeline_data, 'preproc_input_target.pkl')


    def nn_trainer(self):

        X_transformed = self.pipeline_targets.transform(self.X)
        nn = NearestNeighbors(n_neighbors=10).fit(X_transformed)
        joblib.dump(nn, 'nn.pkl')


    def run_targets(self):

        self.preproc_pca_pipe()
        self.pipeline_targets.fit_transform(self.X)
    # self.pipeline_targets.transform(self.X)

    # def save_model_targets(self):

    #     joblib.dump(self.pipeline_targets,'model_targets.joblib')


    def set_pipeline_investors(self):
        tfidf_features = 'name_de'
        tfidf_pipe = Pipeline([('tfidf', TfidfVectorizer()), ('dense', DenseTransformer())])

        preprocessor = ColumnTransformer([
                ('name_de',tfidf_pipe,tfidf_features)
        ], remainder='drop')

        self.pipeline_investors = Pipeline([
                            ('preproc',preprocessor),
                            ('pca',PCA(0.95))
        ])
        

        self.pipeline_investors.fit(self.Y)

        joblib.dump(self.pipeline_investors, 'pipeline_investors.pkl')

    def nn_investors(self):
        Y_transformed = self.pipeline_investors.transform(self.Y)
        nn_investors = NearestNeighbors(n_neighbors=10).fit(Y_transformed)
        joblib.dump(nn_investors, 'nn_investors.pkl')


    # def run_investors(self):

    #     self.set_pipeline_investors()
    #     self.pipeline_investors.fit(self.Y)


    # def save_model_investors(self):

    #     joblib.dump(self.pipeline_investors,'model_investors.joblib')



if __name__ == "__main__":

    df_targets = get_targets_data()
    df_investors = get_investors_data()
    df_investor_keys = get_matching_keys()
    df_targets_clean = clean_targets(df_targets)
    df_investors_clean = clean_investors(df_investors,df_investor_keys)
    df_targets_clean = pd.read_csv('targets.csv', index_col=0)
    df_investors_clean = pd.read_csv('investors.csv', index_col=0)
    get_matching_table()
    X = df_targets_clean
    Y = df_investors_clean
    trainer = Trainer(X,Y)
    trainer.preproc_targets_pipe()
    trainer.nn_trainer()
    trainer.clean_target_data()
    trainer.set_pipeline_investors()
    trainer.nn_investors()
    #trainer.run_investors()
    #trainer.save_model_targets()
    #trainer.save_model_investors()
