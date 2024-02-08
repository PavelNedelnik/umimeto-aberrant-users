import json
import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import math
from src.load_scripts import load_item, load_log
from plotly.subplots import make_subplots
from pathlib import Path
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


def submission_to_string(idx, log, item):
    # TODO maybe print side by side
    submission = log.loc[idx]
    task = item.iloc[submission['item']]

    return                                                                                          \
        f"SUBMISSION: by user: {submission['user']} of task: {submission['item']}-{task['name']}" + \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"DISTANCE:\n {submission['distance']}" +                                                   \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"INSTRUCTIONS:\n {task['instructions']}" +                                                 \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"SOLUTION:\n {task['solution']}" +                                                         \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"ANSWER:\n {submission['answer']}"


data_path = Path('data/')

item = pd.read_csv(data_path / 'cached_item.csv', index_col=0)
log = pd.read_csv(data_path / 'cached_log.csv', index_col=0)
log['time'] = pd.to_datetime(log['time'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer(min_df=0.005)
vectors = vectorizer.fit_transform(log['linter_messages'])

log['linter_messages'] = list(map(np.array, vectors.toarray().tolist()))

with open(Path('data/edulint/results.txt'), 'r') as f:
    results = f.read().lower()

feature_descriptions = []
for feature_name in vectorizer.get_feature_names_out():
    begin = results.find(feature_name)
    feature_descriptions.append(results[begin:results.find('_"', begin)])




from src.linter_profile import MeanTaskProfiler, MeanNormTaskProfiler, NormSumTaskProfiler, NormForgetUserProfiler
from src.model import DistanceModel

dim = len(feature_descriptions)
log['task_profile'] = MeanNormTaskProfiler(dim).build_profiles(log)
log['user_profile'] = NormForgetUserProfiler(dim).build_profiles(log)

model = DistanceModel('euclidean', 'l2')

target = 'task_profile'

log['distance'] = model.calculate_distances(log[target], log['linter_messages'])

"""
25070815
33901490
17126622
"""

user_id = 30297714
subplot_dim = 3
user_history = log[log['user'] == user_id].copy()
user_history = user_history.sort_values('time')
user_history['final'] = np.append(user_history['item'].iloc[:-1].values != user_history['item'].iloc[1:].values, True)
session_breakpoints = np.nonzero((user_history['time'].iloc[1:] - user_history['time'].iloc[:-1]) > pd.Timedelta(1, 'h'))[0].tolist()

session_fig = make_subplots(rows=math.ceil((len(session_breakpoints) + 1) / subplot_dim), cols=subplot_dim)

start = 0
for i, end in enumerate(session_breakpoints + [len(user_history)]):
    session = user_history[start:end + 1]
    session_fig.add_trace(
        go.Scatter(
            x=session.index,
            y=session['distance'],
            text='task id ' + session['item'].astype(str),
            mode='lines+markers',
            marker=dict(
                color=session['correct'].apply(lambda x: 'green' if x else 'red'),
                symbol=session['final'].apply(lambda x: 'x' if x else 'circle'),
                size=10
            ),
        ),
        col=i % subplot_dim + 1, row=i // subplot_dim + 1
    )
    session_fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=40),
        showlegend=False,
        title=f'Sessions of user id {user_id}'
    )
    session_fig.update_xaxes(
        tickformat="%H:%M<br>%d-%m"
    )
    start = end + 1


cutoff = 0.15
counts = dict(zip(feature_descriptions, log[log['user'] == user_id]['linter_messages'].mean()))
counts = {label:value for label, value in counts.items() if value >= cutoff}
counts = dict(sorted(counts.items(), reverse=True, key=lambda x: x[1]))
profile_fig = px.bar(x=counts.keys(), y=counts.values(), title=f'User {user_id} profile')
profile_fig.update_layout(
    yaxis_title="Average occurance count", xaxis_title=""
)

# app layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='session_graph',
                        figure=session_fig
                    )
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='user_profile',
                        figure=profile_fig
                    )
                ])
            ])
        ], width=6),

        dbc.Col([
            html.Div(id='submission_description', style={'whiteSpace': 'pre-wrap'})
        ], width=6),
    ]),
])

# callbacks
@app.callback(
    Output('submission_description', 'children'),
    Input('session_graph', 'clickData'),
)
def describe_submission(clickData):
    if clickData is None:
        return 'Click on a point to display additional information.'
    
    point_info = clickData['points'][0]
    x_value = point_info['x']
    y_value = point_info['y']

    return submission_to_string(x_value, log, item)


# callbacks
@app.callback(
    Output('user_profile', 'figure'),
    Input('session_graph', 'clickData'),
)
def describe_submission(clickData):
    if clickData is None:
        return profile_fig
    
    point_info = clickData['points'][0]
    x_value = point_info['x']
    y_value = point_info['y']

    labels = feature_descriptions[:-2] + [feature_descriptions[-1]]
    vals = log['task_profile'].loc[x_value] * log['user_profile'].loc[x_value]

    return px.bar(x=labels, y=vals[:-2].tolist() + [vals[-1]], title=f'Recommendation')

# run the app
if __name__ == '__main__':
    app.run_server(debug=False)
