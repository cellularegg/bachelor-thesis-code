import json

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output

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
app.layout = html.Div([
    html.H1('Manual Outlier Selection'),
    dcc.Graph(
        id='water-level-graph',
        figure=fig
    ),
    html.Button('Refresh', id='refresh-btn', n_clicks=0),
    html.Button('Prev <-', id='prev-btn', n_clicks=0),
    html.Button('Next ->', id='next-btn', n_clicks=0),

    html.Div([
        dcc.Markdown('Click Data:'),
        html.Pre(id='click-data'),
    ]),
])


def toggle_outlier(timestamp: str):
    print(df.loc[(df['timestamp'] == timestamp), 'is_outlier'])
    print(np.invert(df.loc[(df['timestamp'] == timestamp), 'is_outlier']))
    df.loc[(df['timestamp'] == timestamp), 'is_outlier'] = np.invert(
        df.loc[(df['timestamp'] == timestamp), 'is_outlier'])


outlier_idx = 0


def set_cur_outlier(relative_pos: int = 1):
    global outlier_idx
    global cur_outlier
    outlier_idx += relative_pos
    if outlier_idx < 0:
        outlier_idx = 0
    if df.loc[df['is_outlier'] == True, 'timestamp'].shape[0] <= outlier_idx:
        outlier_idx = 0
    if df.loc[df['is_outlier'] == True, 'timestamp'].shape[0] > outlier_idx:
        cur_outlier = df.loc[df['is_outlier'] == True, 'timestamp'].iloc[
            outlier_idx]
    else:
        cur_outlier = None


set_cur_outlier()


@app.callback(
    Output('click-data', 'children'),
    Input('water-level-graph', 'clickData'))
def display_click_data(clickData):
    if clickData is not None:
        toggle_outlier(clickData['points'][0]['x'])
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('water-level-graph', 'figure'),
    Input('refresh-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    Input('prev-btn', 'n_clicks'))
def update_output(refresh_btn, next_btn, prev_btn):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    fig = px.scatter(df, x='timestamp', y='water_level',
                     title=f'{common_id}', color='is_outlier')
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
    fig.update_layout(
        yaxis_range=[df.loc[df['is_outlier'] == False, 'water_level'].min() - 5,
                     df.loc[
                         df['is_outlier'] == False, 'water_level'].max() + 5])
    if 'refresh-btn' in changed_id:
        df.to_parquet(f'./data/{common_id}_outliers_classified.parquet')
    elif 'next-btn' in changed_id:
        set_cur_outlier(1)
    elif 'prev-btn' in changed_id:
        set_cur_outlier(-1)
    if 'next-btn' in changed_id or 'prev-btn' in changed_id:
        fig.update_layout(
            xaxis_range=[cur_outlier - pd.Timedelta(5, 'D'),
                         cur_outlier + pd.Timedelta(5, 'D')])

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
