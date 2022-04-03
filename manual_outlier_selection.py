import json

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

app = Dash(__name__)

stations_dict = pd.read_csv('./data/stations.csv').groupby(
    ['common_id']).first().to_dict('index')

common_id = '2736731000100-de'
# df = pd.read_parquet(f'data/raw/parquet/{common_id}.parquet')
# df['water_level'] = df['value']
# del df['value']
# df['timestamp'] = df['sourceDate']
# df['is_outlier'] = False
# del df['sourceDate']
# df.sort_values(by='timestamp', inplace=True)
# df.reset_index(drop=True, inplace=True)
# df.to_parquet(f'./data/{common_id}_outliers_classified.parquet')
df = pd.read_parquet(f'./data/{common_id}_outliers_classified.parquet')
df.info()
fig = px.scatter(df, x='timestamp', y='water_level', title=f'{common_id}',
                 color='is_outlier')
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     step="day",
                     stepmode="backward"),
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
# fig.update_layout(yaxis_range=[38950, 39200])
app.layout = html.Div([
    html.H1('Manual Outlier Selection'),
    dcc.Graph(
        id='water-level-graph',
        figure=fig
    ),
    html.Button('Refresh', id='refresh-btn', n_clicks=0),
    html.Div([
        dcc.Markdown("""
        **Click Data**

        Click on points in the graph.
    """),
        html.Pre(id='click-data'),
    ]),
])


def toggle_outlier(timestamp: str):
    # t = pd.to_datetime(timestamp)
    print(df.loc[(df['timestamp'] == timestamp), 'is_outlier'])
    print(np.invert(df.loc[(df['timestamp'] == timestamp), 'is_outlier']))
    df.loc[(df['timestamp'] == timestamp), 'is_outlier'] = np.invert(
        df.loc[(df['timestamp'] == timestamp), 'is_outlier'])


@app.callback(
    # Output('water-level-graph', 'figure'),
    Output('click-data', 'children'),
    Input('water-level-graph', 'clickData'))
def display_click_data(clickData):
    if clickData is not None:
        toggle_outlier(clickData['points'][0]['x'])
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('water-level-graph', 'figure'),
    Input('refresh-btn', 'n_clicks'))
def update_output(n_clicks):
    fig = px.scatter(df, x='timestamp', y='water_level',
                     title=f'{common_id}', color='is_outlier')
    df.to_parquet(f'./data/{common_id}_outliers_classified.parquet')
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # print(df.loc[df['is_outlier'] == False, 'water_level'].min())
    fig.update_layout(
        yaxis_range=[df.loc[df['is_outlier'] == False, 'water_level'].min() - 5,
                     df.loc[df['is_outlier'] == False, 'water_level'].max() + 5])
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
