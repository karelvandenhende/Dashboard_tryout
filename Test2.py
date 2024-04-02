
from function_coating_v3 import Coating
from Function_remove_concrete_not_full_cover_v3 import C_paper_vectorized2
from Function_repair_mortar_v3 import C_paper_vectorized
import numpy as np
import scipy

x_range = np.linspace(0,100,101)

# Import packages
from dash import Dash, html, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app
app = Dash(__name__)

# Define input parameters
inputs = [
    ("Diffusion coefficient concrete Dc [m^2/y]:", .5, "my-input1"),
    ("Surface concentration Cs [m%cem]:", 2, "my-input2"),
    ("Age at application [y]:", 50, "my-input3"),
    ("Time after application [y]:", 5, "my-input4"),
    ("Convection coeff. coating [m/y]:", 0.01, "my-input5"),
    ("Diffusion coefficient mortar Dm [m^2/y]:", .1, "my-input6"),
    ("Thickness mortar layer [mm]:", 20, "my-input7"),
    ("Cover [mm]:", 40, "my-input8")
]

# Create checkboxes for curve selection
checkboxes = [
    dcc.Checklist(
        id='curve-selection',
        options=[
            {'label': 'No repair', 'value': 'no-repair'},
            {'label': 'Coating', 'value': 'coating'},
            {'label': 'Replace', 'value': 'replace'},
            {'label': 'Overlay', 'value': 'overlay'}
        ],
        value=['no-repair', 'coating', 'replace', 'overlay'],
        inline=True
    )
]

input_components = [html.Div([
                        html.Label(description),
                        dcc.Input(value=value, id=id)
                    ]) for description, value, id in inputs]


# App layout
app.layout = html.Div([
    html.Div(checkboxes),
    html.Hr(),
    *input_components,
    #html.Div(children='My First App with Data, Graph, and Controls'),
    html.Button('Calculate', id='calculate-button', n_clicks=0),
    #dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls-and-radio-item'),
    #dash_table.DataTable(data=df.to_dict('records'), page_size=6),
    #dcc.Graph(figure={}, id='controls-and-graph'),
    #dcc.Graph(figure={}, id='controls-and-graph2')
    html.Div([
        dcc.Graph(figure={}, id='controls-and-graph'),
        dcc.Graph(figure={}, id='controls-and-graph2')
    ], style={'display': 'flex', 'flex-direction': 'row'})
    #html.Div(id='my-output1')
])

def y_results(D,Cs,t1,t2,hs,x_range):
    y_range = []
    for x in x_range:
        y_range.append(Coating(np.array([float(x)]),300,np.array([float(D)*31.54]),np.array([float(D)*31.54]),np.array([float(hs)]),float(t1),float(t2),np.array([float(Cs)]),50,50)[0])
    return y_range

def y_results_overlay(D,D2,Cs,t1,t2,thick,x_range):
    y_range = []
    for x in x_range:
        y_range.append(C_paper_vectorized(float(x)-thick,thick,300,np.array([float(D2)*31.54]),np.array([float(D)*31.54]),np.array([float(D)*31.54]),float(t1),float(t2),np.array([float(Cs)]),50,50)[0])
    return y_range

def y_results_replace(D,D2,Cs,t1,t2,thick,x_range):
    y_range = []
    for x in x_range:
        y_range.append(C_paper_vectorized2(float(x),thick,300,np.array([float(D2)*31.54]),np.array([float(D)*31.54]),np.array([float(D)*31.54]),float(t1),float(t2),np.array([float(Cs)]),50,50)[0])
    return y_range

def y_results_norep(D,Cs,t1,t2,x_range):
    return float(Cs)*(1-scipy.special.erf(x_range/(2*np.power(float(D)*31.54*(float(t1)+float(t2)),0.5))))

def coating_over_time(D,Cs,t1,t_range2,hs,cover):
    y_range = []
    for t in t_range2:
        y_range.append(Coating(np.array([float(cover)]),300,np.array([float(D)*31.54]),np.array([float(D)*31.54]),np.array([float(hs)]),float(t1),t,np.array([float(Cs)]),50,50)[0])
    return y_range

def overlay_over_time(D,D2,Cs,t1,t_range2,thick,cover):
    y_range = []
    for t in t_range2:
        y_range.append(C_paper_vectorized(float(cover),thick,300,np.array([float(D2)*31.54]),np.array([float(D)*31.54]),np.array([float(D)*31.54]),float(t1),t,np.array([float(Cs)]),50,50)[0])
    return y_range

def replace_over_time(D,D2,Cs,t1,t_range2,thick,cover):
    y_range = []
    for t in t_range2:
        y_range.append(C_paper_vectorized2(float(cover),thick,300,np.array([float(D2)*31.54]),np.array([float(D)*31.54]),np.array([float(D)*31.54]),float(t1),t,np.array([float(Cs)]),50,50)[0])
    return y_range

def nothing_over_time(D,Cs,t_range,cover):
    y_range = []
    for t in t_range:
        y_range.append(float(Cs)*(1-scipy.special.erf(float(cover)/(2*np.power(float(D)*31.54*(float(t)),0.5)))))
    return y_range

# Add controls to build the interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Output(component_id='controls-and-graph2', component_property='figure'),
    #Output('my-output1','children'),
    #Input(component_id='controls-and-radio-item', component_property='value'),
    Input('calculate-button', 'n_clicks'),
    State('my-input1',component_property='value'),
    State('my-input2',component_property='value'),
    State('my-input3',component_property='value'),
    State('my-input4',component_property='value'),
    State('my-input5',component_property='value'),
    State('my-input6',component_property='value'),
    State('my-input7',component_property='value'),
    State('my-input8',component_property='value'),
    Input(component_id='curve-selection', component_property='value')
)
def update_graph(n_clicks,input_val1,input_val2,input_val3,input_val4,input_val5,input_val6,input_val7,input_val8,curves):
    if n_clicks > 0:
        
        try:
            n_clicks = 0

            #GRAPH1
            fig = px.scatter(title='Concentration over depth at age of '+str(float(input_val3)+float(input_val4))+' y')
            
            if 'no-repair' in curves:
                y_values2 = y_results_norep(input_val1,input_val2,input_val3,input_val4,x_range)
                fig.add_scatter(x=x_range, y=y_values2, mode='lines', name='No repair')
            
            if 'coating' in curves:
                y_values = y_results(input_val1,input_val2,input_val3,input_val4,input_val5,x_range)
                fig.add_scatter(x=x_range, y=y_values, mode='lines', name='Coating')
            
            if 'replace' in curves:
                y_values3 = y_results_replace(input_val1,input_val6,input_val2,input_val3,input_val4,input_val7,x_range)
                fig.add_scatter(x=x_range, y=y_values3, mode='lines', name='Replace')
                
            if 'overlay' in curves:
                y_values4 = y_results_overlay(input_val1,input_val6,input_val2,input_val3,input_val4,input_val7,x_range)
                x_overlay = np.linspace(0,120,121) - float(input_val7)
                fig.add_scatter(x=x_overlay, y=y_values4, mode='lines', name='Overlay')
            
            fig.update_layout(xaxis_title='Depth [mm]', yaxis_title='Chloride concentration [m%cem]')

            # Add horizontal dotted line
            fig.add_hline(y=0.4, line_dash="dot", annotation_text="Critical content", annotation_position="bottom left")
            fig.add_vline(x=float(input_val8), line_dash="dot", annotation_text="Cover", annotation_position="top right")

            # Customize axes limits
            fig.update_xaxes(range=[-float(input_val7), 80])
            fig.update_yaxes(range=[0, float(input_val2)+0.5])


            #GRAPH2
            t_range1 = np.linspace(0.01,float(input_val3),int(input_val3)+1)
            t_range2_prox = np.linspace(1,float(input_val4),int(input_val4))
            t_range2 = np.linspace(float(input_val3)+1,float(input_val3)+float(input_val4),int(input_val4))
            t_range_tot = t_range1.tolist() + t_range2.tolist()

            y_no_rep1 = nothing_over_time(input_val1,input_val2,t_range1,input_val8)
            
            fig2 = px.scatter(title='Concentration over time at cover')
            
            if 'no-repair' in curves:
                y_no_rep2 = nothing_over_time(input_val1,input_val2,t_range2,input_val8)
                y_no_rep = y_no_rep1 + y_no_rep2
                fig2.add_scatter(x=t_range_tot, y=y_no_rep, mode='lines', name='No repair')
                
            if 'coating' in curves:
                y_coating2 = coating_over_time(input_val1,input_val2,input_val3,t_range2_prox,input_val5,input_val8)
                y_coating = y_no_rep1 + y_coating2
                fig2.add_scatter(x=t_range_tot, y=y_coating, mode='lines', name='Coating')

            if 'replace' in curves:
                y_replace2 = replace_over_time(input_val1,input_val6,input_val2,input_val3,t_range2_prox,input_val7,input_val8)
                y_replace = y_no_rep1 + y_replace2
                fig2.add_scatter(x=t_range_tot, y=y_replace, mode='lines', name='Replace')
            
            if 'overlay' in curves:
                y_overlay2 = overlay_over_time(input_val1,input_val6,input_val2,input_val3,t_range2_prox,input_val7,input_val8)
                y_overlay = y_no_rep1 + y_overlay2
                fig2.add_scatter(x=t_range_tot, y=y_overlay, mode='lines', name='Overlay')
            
            fig2.update_layout(xaxis_title='Time [y]', yaxis_title='Chloride concentration [m%cem]')

            # Add horizontal dotted line
            fig2.add_hline(y=0.4, line_dash="dot", annotation_text="Critical content", annotation_position="bottom left")
            fig2.add_vline(x=float(input_val3), line_dash="dot", annotation_text="Intervention", annotation_position="top right")
            
            return fig, fig2
        except:
            return {}, {}
    else:
        # Return an empty figure if button not clicked yet
        return {}, {}
    #return fig#, input_val

# Run the app
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port="8050")
    #app.run(debug=True, port=8050)
