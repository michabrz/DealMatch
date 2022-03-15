import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from DealMatch.predict_unsupervised import make_prediction_investors

MODEL_SUPERVISED = './DealMatch/model_supervised_MLP1.joblib'
MODEL_TARGETS_CLEANING = './DealMatch/model_target_cleaner_for_matching.joblib'

def get_input_data():
    #input_data = pd.read_excel('targets_clean_test.xlsx')
    input_data = pd.read_csv('input_data.csv', index_col=0)
    return input_data

def get_model_targets_cleaning():
    pipe_targets_cleaner = joblib.load(MODEL_TARGETS_CLEANING)
    return pipe_targets_cleaner

def get_model_supervised():
    pipe_supervised = joblib.load(MODEL_SUPERVISED)
    return pipe_supervised

def transform_investors():
    df_investors = pd.read_csv('./DealMatch/investors_output.csv', index_col=0)
    investor_profiles = pd.read_csv('./DealMatch/investor_profiles_to_merge.csv', index_col=0)
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
    pred_df.to_csv('pred_df_names.csv')
    pred_df.drop(columns=['name','investor_id'],inplace=True)
    pred_df.to_csv('pred_df')
    return pred_df

def custom_predict(X, custom_threshold):
    df_final = pd.read_csv('pred_df_names.csv', index_col=0)
    df_investors = pd.read_csv('investors_output.csv', index_col=0)
    df_investors = df_investors.drop_duplicates(subset='name', keep="first")
    model = get_model_supervised()
    probs = model.predict_proba(
            X)  # Get likelihood of each sample being classified as 0 or 1
    expensive_probs = probs[:, 1]  # Only keep expensive likelihoods (1)
    class_list = (expensive_probs > custom_threshold).astype(int)
    df_final['investor_classification'] = class_list
    df_final = df_final[['name','investor_classification']]
    missing_investors = []
    for name in df_investors['name']:
        if ~df_final['name'].str.contains(name).any():
            missing_investors.append(name)
    df_mi = pd.DataFrame({'name':missing_investors,'investor_classification':len(missing_investors)*['Manual Review Required']})
    df_final = pd.concat([df_final,df_mi])
    df_final.reset_index(inplace=True)
    df_final.drop(columns='index',inplace=True)
    df_final = df_final.merge(df_investors,on='name',how='left').drop(columns=['distance_investor<=>investor','distance_target<=>target'])
    df_final['Rationale'] = 'Fit gem. DealCircle Datenbank'
    df_final.to_csv('final_prediction.csv',encoding='utf-8-sig')
    return df_final


if __name__ == "__main__":
    X = get_pred_table()
    updated_preds = custom_predict(X=X,
                                   custom_threshold=0.1273888936253839)
    print(updated_preds)
