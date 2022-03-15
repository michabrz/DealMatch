from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.impute import SimpleImputer
import joblib
import pandas as pd
from DealMatch.predict_unsupervised import *
from DealMatch.predict_supervised import *

MODEL_PREPROC_1 = './DealMatch/pipeline.pkl'
MODEL_TARGETS_1 = './DealMatch/nn.pkl'
MODEL_SUPERVISED = './DealMatch/model_supervised_MLP1.joblib'
MODEL_TARGETS_CLEANING = './DealMatch/model_target_cleaner_for_matching.joblib'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"greeting": "Welcome to the Deal Match recommender"}

@app.get("/recommend")
def recommend(deal_id, deal_name, deal_type_name, target_company_id,
              target_name, target_description, target_revenue, target_ebitda,
              target_ebit, country_name, region_name, sector_name, strs):

    X = pd.DataFrame(dict(
        deal_id=[int(deal_id)],
        deal_name=[str(deal_name)],
        deal_type_name=[str(deal_type_name)],
        target_company_id=[int(target_company_id)],
        target_name=[str(target_name)],
        target_description=[target_description],
        target_revenue=[float(target_revenue)],
        target_ebitda=[float(target_ebitda)],
        target_ebit=[float(target_ebit)],
        country_name=[str(country_name)],
        region_name=[str(region_name)],
        sector_name=[str(sector_name)],
        strs=[str(strs)]),
                        index=[0])

    targets = pd.read_csv('./DealMatch/targets.csv')
    df_companies = make_prediction_targets(X, MODEL_PREPROC_1, MODEL_TARGETS_1, targets)

    match_investors = matching_investors(df_companies)
    match_investors_list = best_investors(match_investors)
    final_investors = make_prediction_investors(match_investors, match_investors_list)

    custom_threshold=0.1273888936253839

    investor_profiles = pd.read_csv('./DealMatch/investor_profiles_to_merge.csv', index_col=0)
    df_final = pd.merge(final_investors['name'], investor_profiles, on="name")
    df_final.drop_duplicates(inplace=True)

    pipe_targets_cleaner = joblib.load(MODEL_TARGETS_CLEANING)
    input_data_transformed = pipe_targets_cleaner.transform(X)
    SimpleImputer.get_feature_names_out = (lambda self, names=None: self.feature_names_in_)
    input_data_transformed = pd.DataFrame(input_data_transformed,
             columns=pipe_targets_cleaner.get_feature_names_out())

    pred_df = pd.concat([input_data_transformed,df_final],axis=1).ffill().sum(level=0, axis=1)
    pred_df.drop(columns=['name','investor_id'],inplace=True)

    df_final = df_final.drop_duplicates(subset='name', keep="first")
    model = joblib.load(MODEL_SUPERVISED)
    probs = model.predict_proba(pred_df)
    expensive_probs = probs[:, 1]
    class_list = (expensive_probs > custom_threshold).astype(int)
    df_final['match_probability'] = class_list
    df_final = df_final[['name','match_probability']]
    missing_investors = []
    for name in final_investors['name']:
        if ~df_final['name'].str.contains(name).any():
            missing_investors.append(name)
    df_mi = pd.DataFrame({'name':missing_investors,'match_probability':len(missing_investors)*['Manual Review Required']})
    df_final = pd.concat([df_final,df_mi])
    df_final.reset_index(inplace=True)
    df_final.drop(columns='index',inplace=True)
    df_final = df_final.merge(final_investors,on='name',how='left').drop(columns=['distance_investor<=>investor','distance_target<=>target'])
    df_final['Rationale'] = 'Fit gem. DealCircle Datenbank'
    df_final.drop_duplicates(subset="name",inplace=True)

    return df_final
