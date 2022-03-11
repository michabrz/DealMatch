import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from DealMatch.data_supervised import get_targets_clean_data
from sklearn.preprocessing import OneHotEncoder

class Trainer():

    def __init__(self, data):

        self.pipeline_targets = None
        self.data = data
        #self.Y = Y

    def clean_targets_pipeline(self):

        num_transformer = Pipeline([('imputer',
                                     SimpleImputer(missing_values=np.nan,
                                                   strategy='constant',
                                                   fill_value=0)),
                                    ('scaler', RobustScaler())])

        cat_transformer = Pipeline([('imputer',
                                    SimpleImputer(missing_values=np.nan,
                                                strategy='constant',
                                                fill_value='no_region')),
                                    ('ohe',
                                    OneHotEncoder(handle_unknown='ignore',
                                                sparse=False))])

        preprocessor = ColumnTransformer([
            ('num_tr', num_transformer,
             ['target_ebit', 'target_ebitda', 'target_revenue']),
            ('cat_tr', cat_transformer,
             ['deal_type_name', 'country_name', 'region_name', 'sector_name'])
        ],
                                         remainder='drop')

        self.pipeline_targets = Pipeline([('preproc', preprocessor)])
        
    def run_target_cleaner_for_matching(self):

        self.clean_targets_pipeline()
        joblib.dump(self.pipeline_targets, 'pipeline_targets.pkl')
        self.pipeline_targets.fit(self.data)

    def save_model_clean_targets_for_matching(self):

        joblib.dump(self.pipeline_targets, 'model_target_cleaner_for_matching.joblib')


if __name__ == "__main__":
    df_targets = get_targets_clean_data()
    trainer = Trainer(df_targets)
    trainer.run_target_cleaner_for_matching()
    trainer.save_model_clean_targets_for_matching()
