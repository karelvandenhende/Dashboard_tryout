# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:37:15 2024

@author: kavdnhen
"""

from function_coating_v3 import Coating
from Function_remove_concrete_not_full_cover_v3 import C_paper_vectorized2
from Function_repair_mortar_v3 import C_paper_vectorized
import numpy as np
import scipy

x_range = np.linspace(0,100,101)

# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app
app = Dash(__name__)

inputs = [
    ("Description 1:", 5, "my-input1"),
    ("Description 2:", 2, "my-input2"),
    ("Description 3:", 50, "my-input3"),
    ("Description 4:", 0.1, "my-input4"),
    ("Description 5:", 0.01, "my-input5"),
    ("Description 6:", 1, "my-input6"),
    ("Description 7:", 20, "my-input7")
]

input_components = [html.Div([
                        html.Label(description),
                        dcc.Input(value=value, id=id)
                    ]) for description, value, id in inputs]


# App layout
app.layout = html.Div([
    *input_components,
    html.Div(children='My First App with Data, Graph, and Controls'),
    html.Hr(),
    html.Button('Calculate', id='calculate-button', n_clicks=0),
    #dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls-and-radio-item'),
    #dash_table.DataTable(data=df.to_dict('records'), page_size=6),
    dcc.Graph(figure={}, id='controls-and-graph'),
    #html.Div(id='my-output1')
])

def y_results(D,Cs,t1,t2,hs,x_range):
    y_range = []
    for x in x_range:
        y_range.append(Coating(np.array([float(x)]),300,np.array([float(D)]),np.array([float(D)]),np.array([float(hs)]),float(t1),float(t2),np.array([float(Cs)]),50,50)[0])
    return y_range

def y_results_overlay(D,D2,Cs,t1,t2,thick,x_range):
    y_range = []
    for x in x_range:
        y_range.append(C_paper_vectorized(float(x)-thick,thick,300,np.array([float(D2)]),np.array([float(D)]),np.array([float(D)]),float(t1),float(t2),np.array([float(Cs)]),50,50)[0])
    return y_range

def y_results_replace(D,D2,Cs,t1,t2,thick,x_range):
    y_range = []
    for x in x_range:
        y_range.append(C_paper_vectorized2(float(x),thick,300,np.array([float(D2)]),np.array([float(D)]),np.array([float(D)]),float(t1),float(t2),np.array([float(Cs)]),50,50)[0])
    return y_range

def y_results_norep(D,Cs,t1,t2,x_range):
    return float(Cs)*(1-scipy.special.erf(x_range/(2*np.power(float(D)*(float(t1)+float(t2)),0.5))))

# Add controls to build the interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    #Output('my-output1','children'),
    #Input(component_id='controls-and-radio-item', component_property='value'),
    Input('calculate-button', 'n_clicks'),
    State('my-input1',component_property='value'),
    State('my-input2',component_property='value'),
    State('my-input3',component_property='value'),
    State('my-input4',component_property='value'),
    State('my-input5',component_property='value'),
    State('my-input6',component_property='value'),
    State('my-input7',component_property='value')
)
def update_graph(n_clicks,input_val1,input_val2,input_val3,input_val4,input_val5,input_val6,input_val7):
    if n_clicks > 0:
        
        try:
            n_clicks = 0
            #y = y_results(input_val1,input_val2,input_val3,input_val4,input_val5,x_range)
            #fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg',title=input_val)
            y_values = y_results(input_val1,input_val2,input_val3,input_val4,input_val5,x_range)
            y_values2 = y_results_norep(input_val1,input_val2,input_val3,input_val4,x_range)
            y_values3 = y_results_replace(input_val1,input_val6,input_val2,input_val3,input_val4,input_val7,x_range)
            y_values4 = y_results_overlay(input_val1,input_val6,input_val2,input_val3,input_val4,input_val7,x_range)
            fig = px.scatter()
            fig.add_scatter(x=x_range, y=y_values2, mode='lines', name='No repair')
            fig.add_scatter(x=x_range, y=y_values, mode='lines', name='Coating')
            fig.add_scatter(x=x_range, y=y_values3, mode='lines', name='Replace')
            x_overlay = np.linspace(0,120,121) - float(input_val7)
            fig.add_scatter(x=x_overlay, y=y_values4, mode='lines', name='Overlay')
            #fig = px.line(x=x_range, y=y_values, title='Custom Plot', name='Line 1')

            fig.update_layout(xaxis_title='Depth [mm]', yaxis_title='Chloride concentration [m%cem]')

            # Add horizontal dotted line
            fig.add_hline(y=0.4, line_dash="dot", annotation_text="Critical content", annotation_position="bottom right")

            # Customize axes limits
            fig.update_xaxes(range=[-float(input_val7), 80])
            fig.update_yaxes(range=[0, float(input_val2)+0.5])
            return fig
        except:
            return {}
    else:
        # Return an empty figure if button not clicked yet
        return {}
    #return fig#, input_val

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8051)
