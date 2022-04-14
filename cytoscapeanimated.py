import dash
import dash_cytoscape as cyto
from dash import html, dcc, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
df = pd.read_csv('cleaned_dataset.csv')

df['submission_date'] = pd.to_datetime(df['submission_date'])

df = df.set_index('submission_date').groupby('state').resample('M').sum()
df = df.reset_index()
#All non continental states are excluded
df = df.sort_values(['state', 'submission_date'], ascending=True)
df = df.dropna()
#sets pivot for a proper correlation shape
dfp = df.pivot('submission_date', 'state')
df_values = df.sort_values(['state', 'submission_date'], ascending=True)
df_values['case_percentage'] = (df_values.groupby('state')['new_case'].apply(pd.Series.pct_change) * 100).round(2)
df_values['case_percentage'] = df_values['case_percentage'].replace([np.inf, -np.inf], np.NaN)
df_values = df_values.drop(columns=['new_case'], axis=1)
df_values = df_values.sort_values(['submission_date', 'state'], ascending=True)
df_values = list(df_values.itertuples())
#Makes a pairwise correlation matrix between the pivoted dates and states
df_corr = dfp['new_case'].corr(method='pearson')
dfp = dfp['new_case'].rolling(3).corr(method='pearson')
state_dictionary = { 
    "AL": ['FL', 'GA', 'MS', 'TN' ],
    "AZ": ['CA', 'CO', 'NV', 'NM', 'UT' ], 
    "AR": ['LA'],
    "CA": ['AZ', 'NV', 'OR'],
    "CO": [],
    "CT": ['MA', 'NY', 'RI'],
    "DE": ['MD', 'NJ', 'PA'],
    "FL": ['AL', 'GA'],
    "GA": ['AL', 'FL', 'NC', 'SC', 'TN'],
    "ID": ['MT', 'NV', 'OR', 'UT', 'WA', 'WY'],
    "IL": ['IN', 'IA', 'MI', 'KY', 'WI'],
    "IN": ['IL', 'KY', 'MI', 'OH'],
    "IA": ['IL', 'MN', 'MO', 'NE', 'SD', 'WI'],
    "KS": ['CO', 'MO', 'NE', 'OK'], 
    "KY": ['IL', 'IN', 'MO', 'OH', 'TN', 'VA', 'WV'],
    "LA": ['AR'], 
    "ME": ['NH'], 
    "MD": ['DE', 'PA', 'VI', 'WV'], 
    "MA": ['CT', 'NH', 'NY', 'RI', 'VT'],
    "MI": ['IL', 'IN', 'OH', 'WI'],
    "MN": ['IA', 'ND', 'SD', 'WI'], 
    "MS": ['AR', 'LA'], 
    "MO": ['AR', 'IL', 'IA', 'KS', 'KY', 'NE', 'OK', 'TN'],
    "MT": ['ID', 'ND', 'SD', 'WY'], 
    "NE": ['CO', 'IA', 'KS', 'MO', 'SD', 'WY'],
    "NV": ['CA', 'ID', 'OR', 'UT'],
    "NH": ['ME', 'MA', 'VT'],
    "NJ": ['DE', 'NY', 'PA'],
    "NM": ['AZ', 'CO', 'OK', 'TX', 'UT'],
    "NY": ['CT', 'MA', 'NJ', 'PA', 'VT'],
    "NC": ['GA', 'SC', 'TN', 'VA'],
    "ND": ['MN', 'MT', 'SD'],
    "OH": ['IN', 'KY', 'MI', 'PA', 'WV'],
    "OK": ['AR', 'CO', 'KS', 'MO', 'NM', 'TX'],
    "OR": ['CA', 'ID', 'NV', 'WA'],
    "PA": ['DE', 'MD', 'NJ', 'NY', 'OH', 'WV'],
    "RI": ['CT', 'MA'],
    "SC": ['GA', 'NC'],
    "SD": ['IA', 'MT', 'ND', 'WY'],
    "TN": ['AL', 'AR', 'KY', 'GA', 'MS', 'MO', 'NC', 'VA'],
    "TX": ['AR', 'LA', 'NM', 'OK'],
    "UT": ['AZ', 'CO', 'ID', 'NV', 'NM', 'WY'],
    "VT": ['MA', 'NH', 'NY'],
    "VA": ['KY', 'MD', 'NC', 'TN', 'WV'],
    "WA": ['ID', 'OR'],
    "WV": ['KY', 'MD', 'OH', 'PA', 'VA'],
    "WI": ['IL', 'IA', 'MI', 'MN'],
    "WY": ['CO', 'ID', 'MT', 'NE', 'SD', 'UT'],
}
state_locations = {
    "AL": [32.32, -86.90],
    "AZ": [34.05, -111.09], 
    "AR": [34.79, -92.19],
    "CA": [36.78, -119.42],
    "CO": [39.11, -105.36],
    "CT": [41.59, -72.69],
    "DE": [39.00, -75.50],
    "FL": [27.99, -81.76],
    "GA": [33.25, -83.44],
    "ID": [44.07, -114.72],
    "IL": [40.00, -89.00],
    "IN": [40.27, -86.13],
    "IA": [42.03, -93.58],
    "KS": [38.50, -98.00], 
    "KY": [37.84, -84.27],
    "LA": [30.39, -92.33], 
    "ME": [45.37, -68.97], 
    "MD": [39.04, -76.64], 
    "MA": [42.41, -71.39],
    "MI": [44.18, -84.50],
    "MN": [46.39, -94.63], 
    "MS": [33.00, -90.00], 
    "MO": [38.57, -92.60],
    "MT": [46.97, -109.53], 
    "NE": [41.50, -100.00],
    "NV": [39.88, -117.22],
    "NH": [44.00, -71.50],
    "NJ": [39.83, -74.87],
    "NM": [34.31, -106.02],
    "NY": [43.00, -75.00],
    "NC": [35.78, -80.79],
    "ND": [47.65, -100.44],
    "OH": [40.37, -82.99],
    "OK": [36.08, -96.92],
    "OR": [44.00, -120.50],
    "PA": [41.20, -77.19],
    "RI": [41.74, -71.74],
    "SC": [33.83, -81.16],
    "SD": [44.50, -100.00],
    "TN": [35.86, -86.66],
    "TX": [31.00, -100.00],
    "UT": [39.42, -111.95],
    "VT": [44.00, -72.69],
    "VA": [37.93, -78.02],
    "WA": [47.75, -120.74],
    "WV": [39.00, -80.50],
    "WI": [44.50, -89.50],
    "WY": [43.08, -107.29],
}
#edge and node lists to be appended to
state_longitude=[
    32.32, 34.79, 34.05, 36.78, 39.11, 41.59, 37.50, 
    27.99, 33.25, 42.03, 44.07, 40.00, 40.27, 38.50, 37.84, 
    30.39, 42.41, 39.04, 45.37, 44.18, 46.39, 38.57, 33.00, 
    46.97, 35.78, 47.65, 41.50, 44.00, 
    39.83, 34.31, 39.88, 43.00, 40.37, 36.08, 44.00,
    41.20, 39.5, 33.83, 44.50, 35.86, 31.00, 39.42,
    37.6, 44.00, 47.75, 44.50, 39.00, 
    43.08,
]
state_latitude =[
    -86.90,  -92.19, -111.09, -119.42, -105.36, -72.69, -74.6, -81.76, 
    -83.44, -93.58, -114.72, -89.00, -86.13, -98.00, -84.27, 
    -92.33, -69.6, -76.64, -68.97, -84.50, -94.63, -92.60, -90.00, -109.53, 
    -80.79, -100.44, -100.00, -71.0, -73.4, -106.02, -117.22, -76.00, -82.99, -96.92, 
    -120.50, -77.19, -71.4, -81.16, -100.00, -86.66, -100.00, -111.95, -78.6, -74, 
    -120.74, -89.50, -80.50, -107.29
]
sources = []
destinations = []
weights = []
date_int = 1
#global dfp_map 
dfp_map = list(dfp.itertuples())
global index_len
index_len = (len(dfp_map)/48)
#appends data to sources, destination and weights list
for row in df_corr.itertuples():
    #skips the starting which is just a label
    for k, v in list(enumerate(row[2:])):
        if df_corr.columns.values[k-1] != row.Index and df_corr.columns.values[k-1] in state_dictionary[row.Index]:
            sources.append(row.Index)
            destinations.append(df_corr.columns.values[k-1])
            weights.append(v)
MIN = min(weights)
#declaration of app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],)
#creates nodes objects
node_data = []
for num, row in (enumerate(df_values[(0*48):((0*48)+48)])):
    #skips the starting which is just a label
        node_data.append(row[3])
        print(row)

nodes = [{
        'data': {'id': state, 'label': state, 'case': case},
        'position': {'x': 25 *y, 'y': -25*x}
        } 
    for state, case, x, y in zip(df_corr.index, node_data, state_longitude, state_latitude)]
#creates edges objects
edges = [{'data': {'source': x, 'target': y, 'weight': z, 'absweight': abs(z)}, }
    for x,y,z in zip(sources, destinations, weights)]

elements = nodes + edges
date_frames = ['January', 'February']
app.layout = html.Div(
html.Div(children=[
	html.Div(children=[
		html.Div(
        		[html.H1(children='Neighboring State Analysis for COVID-19 Cases'), 
        		html.H3(children='By: Brandon Cossin, Troy Toth | Advisor: Dr. Xiang Lian')
                ],
            	style={'textAlign': 'center', }
            ),
    html.Hr(),
 	cyto.Cytoscape(
        	id='cytoscape-event-callbacks-2',
        	elements=elements,
        	style={'width': '100%', 'height': '75vh', 'display': 'inline-block', 'margin-top': '0px'},
        	layout={
            'name': 'preset',
        },
         stylesheet=[
             {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'background-color': '#DAA520',
                        'text-valign': 'center',
                        #'width': '200%',
                        #'height': '200%'
                    }
                },
                {
                    'selector': '[case == 0]',
                    'style': {
                        'background-color': '#DAA520',
                        'text-valign': 'center',
                    }
                },
                {
                    'selector': '[case == None]',
                    'style': {
                        'background-color': 'grey',
                        'text-valign': 'center',
                    }
                },
                {
                    'selector': '[case < 0]',
                    'style': {
                        'background-color': 'green',
                        'text-valign': 'center',
                    }
                },
                {
                    'selector': '[case > 0]',
                    'style': {
                        'background-color': '#FFA500',
                        'text-valign': 'center',
                    }
                },
                
                {
                    'selector': 'edges',
                    'style': {
                        #'line-color': 'mapData(weight, {}, 1, blue, red)'.format(MIN),
                        #'width': 'mapData(absweight, 0, 1, 1, 10)',
                        #'content' : 'ele.data("weight")',
                    }
                },
             {
                    'selector': '[weight < 0]',
                    'style': {
                        'line-color': 'blue',
                        'width': 'mapData(absweight, 0, 1, 1, 15)',
                    }
                },
                {
                    'selector': '[weight > 0]',
                    'style': {
                        'line-color': 'red',
                        'width': 'mapData(absweight, 0, 1, 1, 15)',
                    }
                },
               
                
         ]
    ),
], className='column', style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top': ' 0'}),
html.Div(children=[
html.Div(
    html.P(id='cytoscape-tapEdgeData-output'), 
    style={'display': 'inline-block', 'width': '100%'}
),
html.Div(
    html.P(id='cytoscape-tapNodeData-output'), 
    style={'display': 'inline-block', 'width': '100%'}
),
#dcc.RadioItems(id='timeframe_selector', options = ['Total', 'Weekly','Monthly', 'Yearly'], value = 'Monthly'),
html.P('Choose Timeframe:'),
html.Div(
dcc.Dropdown(['Total', 'Weekly','Monthly', 'Quarter-Yearly'], 'Monthly', id='timeframe_selector'), style={'display': 'inline-block', 'width': '90%'}
),
html.Br(),
html.P('Choose Weight Range:'),
#dcc.RangeSlider(min=-1, max=1, step=.1, value=[-1, 1], id='cor-range-slider'),
dbc.InputGroup(
    [
html.Div(children=[
dbc.Input(id='range-min', type='number', min=-1, max=1, step=.1, value=-1),
dbc.FormText("Minimum Weight"),
], style={'margin-right': '20px'}
),
html.Div(
    children=[
dbc.Input(id='range-max', type='number', max =1 , min = -1, step=.1, value=1),
dbc.FormText("Maximum Weight"),
    ]
),
    
    ]
),
html.Br(),
html.P('Correlation Strength'),
html.Img(src= '/assets/cytolegend.png'),
html.P('New Case Change'),
html.Img(src= '/assets/cytolegend2.png'),
html.Div([
    dbc.Button('Play', id='play-button', className="me-1", disabled=True),
    dbc.Button('Pause', id='pause-button', className="me-1"),
    dbc.Button('Restart', id='restart-button', className="me-1")
], style={'margin-top': '5px'}),
html.Div(
    children=[
dbc.Input(id='speed-slider', type='number', max = 2.75 , min = .25, step=.25, value=1),
dbc.FormText("Adjust Speed"),
    ], style={'width': '40%', 'margin-top': '5px'}
),
], className='column', style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top','margin-top': '0'}),
html.Div(
    children=[
    html.Div(
    dcc.Slider(
                step=None, 
                marks = dict([(k, str(v)) for k, v in enumerate(date_frames)]), 
                value=0, id='my-range-slider'), 
                style={'width': '90%', 'margin-left': 'auto', 'margin-right': 'auto'}
                ),
    html.Div(id="interval-div", children = dcc.Interval(
            id='interval-component',
            
            interval=1*2000, # in milliseconds
            disabled=False,
            n_intervals=0,
            max_intervals= (len(dfp_map)/48)
        ),
    )           
    ], className='row' ), 
], className='row'), style={"overflow-x": "hidden"}
)
#each time iterated by interval will advance the date slider
@app.callback( Output('my-range-slider', 'value'),
              Input('interval-component', 'n_intervals'))
def update_metrics(i):
#each "date frame" is 48 states long
    return i
#date slider will render that date frame
@app.callback(Output('cytoscape-event-callbacks-2', 'elements'),
              [Input('my-range-slider', 'value'),
              Input('range-min', 'value'),
              Input('range-max', 'value')
              ])
def render_frame(i, value1, value2):
    global date_int
    date_int = i
    global sources
    global destinations
    global weights
    
    sources=[]
    destinations=[] 
    weights=[]
    if(len(dfp_map) > 48):
        for num, row in (enumerate(dfp_map[(i*48):((i*48)+48)])):
    #skips the starting which is just a label
            #print(row)
            for k, v in list(enumerate(row[2:])):
                #print(k, v)
                if dfp.columns.values[k+1] != row.Index[1] and dfp.columns.values[k+1] in state_dictionary[row.Index[1]]:
                    sources.append(row.Index[1])
                    destinations.append(dfp.columns.values[k+1])
                    weights.append(v)
    
    else:
        for row in dfp.itertuples():
    #skips the starting which is just a label
            #print(row)
            for k, v in list(enumerate(row[2:])):
                #print(k, v)
                if dfp.columns.values[k+1] != row.Index and dfp.columns.values[k+1] in state_dictionary[row.Index]:
                    sources.append(row.Index)
                    destinations.append(dfp.columns.values[k+1])
                    weights.append(v)
    
    node_data = []
    if(len(dfp_map) > 48) :
        for num, row in (enumerate(df_values[(i*48):((i*48)+48)])):
        #skips the starting which is just a label
            node_data.append(row[3])
            #print(row)
    else :
        for i in range(48):
        #skips the starting which is just a label
            node_data.append(0)
    nodes = [{
        'data': {'id': state, 'label': state, 'case': case},
        'position': {'x': 25 *y, 'y': -25*x}
        } 
    for state, case, x, y in zip(df_corr.index, node_data, state_longitude, state_latitude)]

    edges = [{'data': {'source': x, 'target': y, 'weight': z, 'absweight': abs(z)}} 
    for x,y,z in zip(sources, destinations, weights) if z > value1 and z < value2]
    elements = nodes + edges
    return elements
#Timeframe selector changes the slider values
@app.callback(Output('my-range-slider', 'marks'),
                Output('interval-div', 'children'),
              Input('timeframe_selector', 'value'),
              )
def revalueSlider(d):
    global dfp_map
    global dfp
    global df_values
    df = pd.read_csv('cleaned_dataset.csv')
    df['submission_date'] = pd.to_datetime(df['submission_date'])
    if d == 'Monthly':
        df = df.set_index('submission_date').groupby('state').resample('M').sum()
        df = df.reset_index()
        df = df.sort_values(['state', 'submission_date'], ascending=True)
        df = df.dropna()
        dfp = df.pivot('submission_date', 'state')
        df_values = df.sort_values(['state', 'submission_date'], ascending=True)
        df_values['case_percentage'] = (df_values.groupby('state')['new_case'].apply(pd.Series.pct_change) * 100).round(2)
        df_values['case_percentage'] = df_values['case_percentage'].replace([np.inf, -np.inf], np.NaN)
        df_values = df_values.drop(columns=['new_case'], axis=1)
        df_values = df_values.sort_values(['submission_date', 'state'], ascending=True)
        #print(df_values.head(200))
        df_values = list(df_values.itertuples())
        dfp = dfp['new_case'].rolling(3).corr(method='pearson')
        dfp_map = list(dfp.itertuples())
        #this gets how many 'dates' there are
        index_len = int((len(dfp_map)/48))
        dates_list = []
        for i in range(index_len) :
            #every 5th element is printed
            if i % 2 == 0:
                dates_list.append((dfp_map[(i*48)].Index[0]).date())
            else:
                dates_list.append(' ')
        marks = dict([(k, str(v)) for k, v in enumerate(dates_list)])
        child = [dcc.Interval(
            id='interval-component',
            
            interval=1*1000, # in milliseconds
            disabled=False,
            n_intervals=0,
            max_intervals= (len(dfp_map)/48) -1 )]
        return marks, child
    if d == 'Weekly':
        df = df.set_index('submission_date').groupby('state').resample('W').sum()
        df = df.reset_index()
        df = df.sort_values(['state', 'submission_date'], ascending=True)
        df = df.dropna()
        dfp = df.pivot('submission_date', 'state')
        df_values = df.sort_values(['state', 'submission_date'], ascending=True)
        df_values['case_percentage'] = (df_values.groupby('state')['new_case'].apply(pd.Series.pct_change) * 100).round(2)
        df_values['case_percentage'] = df_values['case_percentage'].replace([np.inf, -np.inf], np.NaN)
        df_values = df_values.drop(columns=['new_case'], axis=1)
        df_values = df_values.sort_values(['submission_date', 'state'], ascending=True)
        df_values = list(df_values.itertuples())
        dfp = dfp['new_case'].rolling(3).corr(method='pearson')
        dfp_map = list(dfp.itertuples())
        index_len = int((len(dfp_map)/48))
        dates_list = []
        for i in range(index_len) :
            #every 5th element is printed
            if i % 6 == 0:
                dates_list.append((dfp_map[(i*48)].Index[0]).date())
            else:
                dates_list.append(' ')
        marks = dict([(k, str(v)) for k, v in enumerate(dates_list)])
        child = [dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            disabled=False,
            n_intervals=0,
            max_intervals= (len(dfp_map)/48) -1 )]
        return marks, child
    if d == 'Quarter-Yearly':
        df = df.set_index('submission_date').groupby('state').resample('BQ').sum()
        df = df.reset_index()
        df = df.sort_values(['state', 'submission_date'], ascending=True)
        df = df.dropna()
        dfp = df.pivot('submission_date', 'state')
        df_values = df.sort_values(['state', 'submission_date'], ascending=True)
        df_values['case_percentage'] = (df_values.groupby('state')['new_case'].apply(pd.Series.pct_change) * 100).round(2)
        df_values['case_percentage'] = df_values['case_percentage'].replace([np.inf, -np.inf], np.NaN)
        df_values = df_values.drop(columns=['new_case'], axis=1)
        df_values = df_values.sort_values(['submission_date', 'state'], ascending=True)
        df_values = list(df_values.itertuples())
        dfp = dfp['new_case'].rolling(3).corr(method='pearson')
        dfp_map = list(dfp.itertuples())
        index_len = int((len(dfp_map)/48))
        dates_list = []
        for i in range(index_len) :
            dates_list.append((dfp_map[(i*48)].Index[0]).date())
        marks = dict([(k, str(v)) for k, v in enumerate(dates_list)])
        child = [dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            disabled=False,
            n_intervals=0,
            max_intervals= (len(dfp_map)/48) -1 )]
        return marks, child
    if d == 'Total':
        df = df.set_index('submission_date').groupby('state').resample('M').sum()
        df = df.reset_index()
        df = df.sort_values(['state', 'submission_date'], ascending=True)
        df = df.dropna()
        dfp = df.pivot('submission_date', 'state')
        df_values = df.sort_values(['state', 'submission_date'], ascending=True)
        df_values['case_percentage'] = (df_values.groupby('state')['new_case'].apply(pd.Series.pct_change) * 100).round(2)
        df_values['case_percentage'] = df_values['case_percentage'].replace([np.inf, -np.inf], np.NaN)
        df_values = df_values.drop(columns=['new_case'], axis=1)
        df_values = df_values.sort_values(['submission_date', 'state'], ascending=True)
        df_values = list(df_values.itertuples())
        dfp = dfp['new_case'].corr(method='pearson')
        dfp_map = list(dfp.itertuples())
        index_len = int((len(dfp_map)/48))
        dates_list = [' ', 'total']
        marks = dict([(k, str(v)) for k, v in enumerate(dates_list)])
        child = [dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            disabled=False,
            n_intervals=0,
            max_intervals= 1)]
        return marks, child
#pause/play and restart buttons
@app.callback(
    Output('interval-component', 'interval'),
    Output('interval-component', 'disabled'),
    Output('play-button', 'disabled'),
    Output('pause-button', 'disabled'),
    Output('interval-component', 'n_intervals'),
    Input('play-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    Input('restart-button', 'n_clicks'),
    Input('speed-slider', 'value'))
def buttonClick(button1, button2, button3, speed):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'play-button' in changed_id:
        return (3-speed)*1000, False, True, False, dash.no_update
    elif 'pause-button' in changed_id:
        return (3-speed)*1000, True, False, True, dash.no_update
    elif 'restart-button' in changed_id:
        return (3-speed)*1000, False, True, False, 0
    else:
        return (3-speed)*1000, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(Output('cytoscape-tapEdgeData-output', 'children'),
              Input('cytoscape-event-callbacks-2', 'tapEdgeData'))
def displayTapEdgeData(data):
    if data:
        if(len(dfp_map) > 48):
             return "The correlation between  " + \
               data['source'] + " and " + data['target'] + " has a weight of " + str(round(data['weight'], 3)) + \
                   " on " + str((dfp_map[(date_int*48)].Index[0]).date())
        else:
            return "The correlation between  " + \
               data['source'] + " and " + data['target'] + " has a weight of " + str(round(data['weight'], 3)) + " for all time "
@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
              Input('cytoscape-event-callbacks-2', 'tapNodeData'))
def displayTapNodeData(data):
    if data:
        if (len(dfp_map) > 48):
            return data['label'] + " has a new case percentage of  " + \
                str(data['case']) + "% on " + str((dfp_map[(date_int*48)].Index[0]).date())
        else:
            #print(dfp_map[(0)].Index[0])
            return data['label'] + " has a new case percentage of  None for all time"
if __name__ == '__main__':
    app.run_server(debug=True)
