import pandas as pd
import dash
from dash import html
import flask
from dash import dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px



url = 'Pokemon.csv'
df = pd.read_csv(url)
df.head()

app = dash.Dash(__name__)

app.layout =html.Div([
    html.H1('Pokemon Data Visualization', style={'textAlign': 'center'}),
    dcc.Dropdown(id='select_group',
                 options=
                     [{'label':"All Region", 'value': 'All'}]+
                     [{'label':f'{Region.capitalize()}','value': Region} for Region in df['Region'].unique()],
                          multi=False,
                          value='All'
                          ),
    html.Div(id='Pokemon_Census', className='chart-grid'),
    html.Br(),


    dcc.Dropdown(id='select_pokemon',
                 options=[{'label':Name.capitalize(),'value': Name} for Name in df['Name']],
                          multi=False,
                          value='charmander'
                          ),
    html.Div(id='output_container', children=[]),
    html.Br(),
    dcc.Graph(id='Pokemon_stats', figure={})])

#============================================

@app.callback(
    Output(component_id='Pokemon_stats', component_property='figure'),
    Input(component_id='select_pokemon', component_property='value'))

def update_graph(option):
    filtered_df = df[df['Name'] == option]
    fig = go.Figure(
        data=[
            go.Bar(
                x=['HP','Attack','Defense','Special Attack','Special Defense','Speed'],
                y=[filtered_df['HP'].iloc[0],filtered_df['Attack'].iloc[0],filtered_df['Defense'].iloc[0],filtered_df['Special Attack'].iloc[0],filtered_df['Special Defense'].iloc[0],filtered_df['Speed'].iloc[0]],
                marker=dict(color=['green', 'red', 'blue', 'red', 'blue', 'yellow']) 
            )
        ]
    )
    fig.update_layout(
        title=f'{option.capitalize()}\'s Stats', xaxis_title=f'{option.capitalize()}\'s Attribute', yaxis_title='Value')
    return fig
@app.callback(
    Output(component_id='Pokemon_Census', component_property='children'),
    Input(component_id='select_group', component_property='value'))

def update_census(value):
    if value == 'All':
        Filtered_data = df[~df['Region'].isin(['No Region'])]
        Filtered_data = Filtered_data.groupby(['Primary Type'])['Name'].size().reset_index()
        first_graph = dcc.Graph(figure=px.pie(Filtered_data,
                     values='Name',
                     names ='Primary Type',
                     title = f'Census of All {1025} Pokemon based on Primary Type'
                     ))
        Filtered_data = df.groupby(['Secondary Type'])['Name'].size().reset_index()
        second_graph = dcc.Graph(figure=px.pie(Filtered_data,
                     values='Name',
                     names ='Secondary Type',
                     title = f'Census of All {1025} Pokemon based on Secondary Type'
                     ))
    else:
        Filtered_data = df[df['Region'] == value]
        Prim_Filtered_data = Filtered_data.groupby(['Primary Type'])['Name'].size().reset_index()
        first_graph = dcc.Graph(figure=px.pie(Prim_Filtered_data,
                     values='Name',
                     names ='Primary Type',
                     title = f'Census of All {Filtered_data["Name"].count()} Pokemon in {value} based on Primary Type'
                     ))
        Secondary_Filtered_data = Filtered_data.groupby(['Secondary Type'])['Name'].size().reset_index()
        second_graph = dcc.Graph(figure=px.pie(Secondary_Filtered_data,
                     values='Name',
                     names ='Secondary Type',
                     title = f'Census of All {Filtered_data["Name"].count()} Pokemon in {value} based on Secondary Type'
                     ))
    return [html.Div(className='chart-grid', children=[html.Div(children=first_graph), html.Div(children=second_graph)], style={'display': 'flex', 'justify-content': 'center', 'text-align':'center'})]


if __name__ == '__main__':
    app.run_server(debug=True)