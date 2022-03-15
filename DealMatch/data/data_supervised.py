import pandas as pd


def get_targets_clean_data():

    targets = pd.read_excel('../raw_data/targets_clean.xlsx')

    return targets
