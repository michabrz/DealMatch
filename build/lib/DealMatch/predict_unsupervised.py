import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from DealMatch.custom_transformer import DenseTransformer
from DealMatch.data_unsupervised import remove_punctuations
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

MODEL_TARGETS = 'nn.pkl'
MODEL_PREPROC = 'pipeline.pkl'
MODEL_INVESTORS = 'nn_investors.pkl'
MODEL_PREPROC_INVESTORS = 'pipeline_investors.pkl'

def get_target_data():
    df_pred = pd.read_excel('targets_clean_test.xlsx')
    return df_pred

def get_model_target():
    pipe_targets = joblib.load(MODEL_TARGETS)
    return pipe_targets

def get_model_preproc():
    pipe_preproc = joblib.load(MODEL_PREPROC)
    return pipe_preproc

def get_model_investors():
    pipe_investors = joblib.load(MODEL_INVESTORS)
    return pipe_investors

def get_investors_preproc():
    preproc_investors = joblib.load(MODEL_PREPROC_INVESTORS)
    return preproc_investors

def make_prediction_targets():
    df = get_target_data()
    preproc = get_model_preproc()
    
    nltk.download('stopwords')
    
    df['strs'] = df['strs'].str.replace(',',' ')
    df['strs'] = df['strs'].apply(lambda x: remove_punctuations(x))
    df['strs'] = df['strs'].apply(lambda x: x.lower())
    
    
    stop_words = set(stopwords.words('german'))

    for name_de in df['strs']:
        word_tokens = word_tokenize(name_de)
        name_de = [w for w in word_tokens if not w in stop_words]
        
    df.to_csv('input_data.csv')
    
    df_transformed = preproc.transform(df)
    targets_pipe = get_model_target()
    nearest_targets = targets_pipe.kneighbors(df_transformed)
    print(nearest_targets)

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

    matching_table = pd.read_csv('matching_table.csv')

    matching_investors = []
    matching_target = []
    matching_distance = []

    for company in df_companies['name']:
        next_investor = matching_table[(matching_table['target_name']==company) & (matching_table['deal_stage_id']>=3)]['investor_name'].tolist()
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
    
    preproc_investors = get_investors_preproc()
    investors_pipe = get_model_investors()
        
    # df = get_target_data()
    # preproc = get_model_preproc()
    # df_transformed = preproc.transform(df)
    # targets_pipe = get_model_target()
    # nearest_targets = targets_pipe.kneighbors(df_transformed)
    # print(nearest_targets)


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
            to_pred = investors_clean[investors_clean['name']==investor].drop(columns=['Unnamed: 0'])
            
            #preproc_investors = get_investors_preproc()
            #print(preproc_investors)
            to_pred_transformed = preproc_investors.transform(to_pred)
            #investors_pipe = get_model_investors()
            nearest_investors = investors_pipe.kneighbors(to_pred_transformed,4)

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
    df_investors_sorted.drop_duplicates(inplace=True)
    df_investors_sorted.reset_index(inplace=True)
    df_investors_sorted.drop('index',axis=1,inplace=True)

    df_investors_sorted.to_csv('investors_output.csv')

    return df_investors_sorted


if __name__ == "__main__":
    target_nn = make_prediction_targets()
    match_investors = matching_investors(target_nn)
    match_investors_list = best_investors(match_investors)
    final_investors = make_prediction_investors(match_investors, match_investors_list)
