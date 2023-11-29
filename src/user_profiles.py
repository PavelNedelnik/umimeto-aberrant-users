import json
import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import math
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

item = pd.concat([
    item,
    pd.DataFrame({'name': 'unknown', 'solution': 'pass'}, index=[12]),
    pd.DataFrame({'name': 'unknown', 'solution': 'pass'}, index=[118]),

])

# load from cache
log = pd.read_csv(data_path / 'dataset.csv')
log['time'] = pd.to_datetime(log['time'])
log['linter_messages'] = log['linter_messages'].apply(lambda x: np.array(eval(x)))

feature_descriptions = json.load(open(Path(data_path / 'edulint' / 'features.json'), 'r'))
feature_descriptions[0] = 'c0103_snake_case_naming_style'
feature_descriptions[24] = 'r1705_unnecessary_elif_after_return'
feature_descriptions[20] = 'f841_unused_local_variable'


# naive model
log['distance'] = log['linter_messages'].apply(np.sum)


user_id = 14723775
subplot_dim = 2
user_history = log[log['user'] == user_id].copy()
user_history['final'] = np.append(user_history['item'][:-1].values != user_history['item'][1:].values, True)
session_breakpoints = np.nonzero((user_history['time'][1:] - user_history['time'][:-1]) > pd.Timedelta(1, 'h'))[0].tolist()

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
profile_fig = px.bar(x=counts.keys(), y=counts.values(), title=f'Naive detector - user {user_id} profile')
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

# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
