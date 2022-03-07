import pandas as pd
import numpy as np
from DealMatch.predict_unsupervised import matching_investors, best_investors, make_prediction_investors
import joblib

MODEL_SUPERVISED = '../models/model_supervised_MLP1.joblib'
MODEL_TARGETS_CLEANING = '../models/model_target_cleaner_for_matching.joblib'

def get_model_targets_cleaning():
    pipe_targets_cleaner = joblib.load(MODEL_TARGETS_CLEANING)
    return pipe_targets_cleaner

def get_matching_table():
    matching_investors = pd.read_excel('../raw_data/matching_table_raw.xlsx')
    return matching_investors

def get_model_supervised():
    pipe_supervised = joblib.load(MODEL_SUPERVISED)
    return pipe_supervised

def transform_targets():
    pipeline = get_model_targets_cleaning()
    input_data = pd.read_csv('input.csv') # how will we read in the input data that should be transformed using the pre-trained targets-preproc-pipeline?
    input_data_transformed = pipeline.transform(input_data)
    return input_data_transformed

def get_pred_table():
    df_investors_sorted = make_prediction_investors()
    df_investors = get_matching_table()
    df_target = transform_targets()
    df_investors_to_match_target = pd.merge(df_investors_sorted, df_investors, on="investor_id", how="left")
    df_target = df_target[df_target.columns.drop(list(df_target.filter(regex='remainder')))]
    pred_df = pd.concat([df_target, df_investors_to_match_target], axis=1)
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

    # match_investors = matching_investors(pred_df)
    # match_investors_list = best_investors(match_investors)
    # final_investors = make_prediction_investors(match_investors,
    #                                             match_investors_list)
