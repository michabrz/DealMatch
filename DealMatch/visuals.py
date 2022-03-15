import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


def get_visual_data():
    df = pd.read_csv('sector_investors.csv')
    return df

def create_frame(df, investor_name):
    frame_1 = df[df['deal_stage_id']>=4]
    frame_2 = frame_1[['name', 'sector_id', 'name_de', 'target_revenue', 'target_ebitda', 'target_ebit', 'region']]
    frame_3 = frame_2[frame_2['name'] == investor_name]
    frame_4 = frame_3.groupby([frame_3['name_de']]).agg({
        'sector_id':
        'count',
        'target_revenue':
        'median',
        'target_ebitda':
        'median'})
    frame_4.rename(columns={'sector_id': 'sector_count'}, inplace=True)

    return frame_4

def visualize(investor_name):

    df = get_visual_data()

    df = create_frame(df, investor_name)

    sns.set_style('whitegrid')

    g = sns.relplot(x="target_ebitda", y="target_revenue", hue='name_de', size='sector_count',
            sizes=(10, 4000), alpha=.5, palette="muted",
            height=10, aspect=1.5, data=df)

    g._legend.remove()

    g.set(ylabel='Median Target Revenue', xlabel='Median Target EBITDA')
    g.set(title=f"{investor_name}")

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+0.05, point['y']-0.2, str(point['val']))

    return label_point(df.target_ebitda, df.target_revenue, df.name_de, plt.gca())
