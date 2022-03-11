import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

MODEL_SUPERVISED = 'model_supervised_MLP1.joblib'
MODEL_TARGETS_CLEANING = 'model_target_cleaner_for_matching.joblib'

def get_input_data():
    input_data = pd.read_excel('targets_clean_test.xlsx')
    return input_data

def get_model_targets_cleaning():
    pipe_targets_cleaner = joblib.load(MODEL_TARGETS_CLEANING)
    return pipe_targets_cleaner

def get_model_supervised():
    pipe_supervised = joblib.load(MODEL_SUPERVISED)
    return pipe_supervised

def transform_investors():
    df_investors = pd.read_csv('investors_output.csv', index_col=0)
    investor_profiles = pd.read_csv('investor_profiles_to_merge.csv', index_col=0)
    df_investors_supervised = pd.merge(df_investors['name'], investor_profiles, on="name")
    df_investors_supervised.drop_duplicates(inplace=True)
    return df_investors_supervised
     
def transform_targets():
    pipe_targets_cleaner = get_model_targets_cleaning()
    input_data = get_input_data()
    input_data_transformed = pipe_targets_cleaner.transform(input_data)
    SimpleImputer.get_feature_names_out = (lambda self, names=None: self.feature_names_in_)
    input_data_transformed = pd.DataFrame(input_data_transformed,
             columns=pipe_targets_cleaner.get_feature_names_out())
    return input_data_transformed

def get_pred_table():
    
    df_investors = transform_investors()
    df_target = transform_targets()
    pred_df = pd.concat([df_target,df_investors],axis=1).ffill().sum(level=0, axis=1)
    pred_df.drop(columns=['name','investor_id'],inplace=True)
    return pred_df

def custom_predict(X, custom_threshold):
    model = get_model_supervised()
    probs = model.predict_proba(
            X)  # Get likelihood of each sample being classified as 0 or 1
    expensive_probs = probs[:, 1]  # Only keep expensive likelihoods (1)
    return (expensive_probs > custom_threshold).astype(
        int)  # Boolean outcome converted to 0 or 1



if __name__ == "__main__":
    X = get_pred_table()
    updated_preds = custom_predict(X=X,
                                   custom_threshold=0.1273888936253839)
    print(updated_preds)
