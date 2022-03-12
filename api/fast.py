from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

MODEL_PREPROC = './DealMatch/pipeline.pkl'
MODEL_TARGETS = './DealMatch/nn.pkl'

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
       target_revenue=[int(target_revenue)],
       target_ebitda=[int(target_ebitda)],
       target_ebit=[int(target_ebit)],
       country_name=[str(country_name)],
       region_name=[str(region_name)],
       sector_name=[str(sector_name)],
       strs=[str(strs)]),
                    index=[0])

   preproc = joblib.load(MODEL_PREPROC)
   X_transformed = preproc.transform(X)
   targets_pipe = joblib.load(MODEL_TARGETS)
   nearest_targets = targets_pipe.kneighbors(X_transformed)

   name = []
   description = []
   distance = []

   targets = pd.read_csv('./DealMatch/targets.csv')

   for x, y in zip(nearest_targets[1][0], nearest_targets[0][0]):
        name.append(targets['target_name'].iloc[x])
        description.append(targets['strs'].iloc[x])
        distance.append(y)

   df_companies = {'name': name,'description': description,'distance': distance}

   return df_companies
