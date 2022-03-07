import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

MODEL_TARGETS = 'model_targets.joblib'
MODEL_INVESTORS = 'model_investors.joblib'

def get_target_data():
    df_pred = pd.read_csv('target_data.csv')
    return df_pred

def get_model_target():
    pipe_targets = joblib.load(MODEL_TARGETS)
    return pipe_targets

def get_model_investors():
    pipe_investors = joblib.load(MODEL_INVESTORS)
    return pipe_investors

def make_prediction_targets():
    df = get_target_data()
    pipeline = get_model_target()
    nearest_targets = pipeline.kneighbors(df,10)

    targets = pd.read_csv('targets.csv')

    name = []
    description = []
    distance = []

    for x,y in zip(nearest_targets[1][0],nearest_targets[0][0]):
        name.append(targets['target_name'].iloc[x])
        description.append(targets['strs'].iloc[x])
        distance.append(y)


    df_companies = pd.DataFrame({'name':name,
                'description':description,
                'distance':distance})

    return df_companies

def matching_investors(df_companies):

    matching_table = pd.read_excel('matching_table_raw.xlsx')

    matching_investors = []
    matching_target = []
    matching_distance = []

    for company in df_companies['name']:
        next_investor = matching_table[(matching_table['target_name']==company) & (matching_table['deal_stage_id']>=3)]['comp_name'].tolist()
        matching_investors+=next_investor
        matching_target+=len(next_investor)*[company]
        next_distance = df_companies[df_companies['name']==company]['distance'].tolist()
        matching_distance+=len(next_investor)*next_distance
    df_match_investors = pd.DataFrame({'investors':matching_investors,'targets':matching_target,'distance':matching_distance})

    return df_match_investors

def best_investors(df_match_investors):

    if len(df_match_investors['investors'].unique())>=10:
        best_investors = df_match_investors['investors'].unique()[:10].tolist()
    else:
        best_investors = df_match_investors['investors'].unique().tolist()

    return best_investors

def make_prediction_investors(df_match_investors, best_investors):

    investors_clean = pd.read_csv('investors.csv')

    name_investor = []
    description_investor = []
    distance_investor_investor = []
    distance_target_target = []


    for investor in best_investors:
        name_investor.append(investor)
        if investors_clean['name'].str.contains(investor).any():
            description_investor.append(investors_clean[investors_clean['name']==investor]['name_de'].to_list()[0])
        else:
            description_investor.append('Investor not in the list')
        distance_investor_investor.append(0)
        distance_target_target.append(df_match_investors[df_match_investors['investors']==investor]['distance'].min())


    for investor in best_investors:
        if investors_clean['name'].str.contains(investor).any():
            first_distance = df_match_investors[df_match_investors['investors']==investor]['distance'].min()
            to_pred = investors_clean[investors_clean['name']==investor]
        pipeline = get_model_investors()
        nearest_investors = pipeline.kneighbors(investor,4)


        for x,y in zip(nearest_investors[1][0],nearest_investors[0][0]):
            name_investor.append(investors_clean['name'].iloc[x])
            description_investor.append(investors_clean['name_de'].iloc[x])
            distance_investor_investor.append(y)
            distance_target_target.append(first_distance)

    df_investors = pd.DataFrame({'name':name_investor,
                'description':description_investor,
                'distance_investor<=>investor':distance_investor_investor,
                    'distance_target<=>target':distance_target_target,
                    'distance_target<=>investor': [a+b for a,b in zip(distance_investor_investor,distance_target_target)]})

    df_investors_sorted = df_investors.sort_values('distance_target<=>investor')
    df_investors_sorted.reset_index(inplace=True)
    df_investors_sorted.drop('index',axis=1,inplace=True)

    return df_investors_sorted


if __name__ == "__main__":
    target_nn = make_prediction_targets()
    match_investors = matching_investors(target_nn)
    match_investors_list = best_investors(match_investors)
    final_investors = make_prediction_investors(match_investors, match_investors_list)
