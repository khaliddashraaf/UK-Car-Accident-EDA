import pandas as pd
import plotly.express as px

from sqlalchemy import create_engine
import dash
import dash_core_components as dcc
import dash_html_components as html
    
def fig1(df):
    df['date_time'] = df['date']+' '+df['time']
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True, infer_datetime_format=True)
    fig = px.scatter_mapbox(df,lon='longitude', lat='latitude',
                        zoom=5, color='accident_severity',
                        hover_name='police_force', 
                        hover_data=['date_time','first_road_class','first_road_number',
                                    'speed_limit','number_of_vehicles','number_of_casualties'],
                        width=1000,height=800,opacity=0.5,size_max=0.0001)
    fig.update_layout(mapbox_style='open-street-map', showlegend=True, title='Location of Accidents in the UK 2010')
    fig.update_layout(mapbox=dict(center=dict(lat=54,lon=-1)))
    return fig
    
def fig2(df):
    fig = px.histogram(df, x='police_force', title="Accidents per police force")
    fig.show()
    return fig

def fig3(df):
    road = df[df['first_road_class']  == df['first_road_class'].value_counts().idxmax()]
    road_number = road['first_road_number'].value_counts()[:30].reset_index()

    fig = px.bar(road_number, x='index', y='first_road_number', title='Accidents per Road number on road class "A"')
    fig.show()
    return fig

def create_dashboard(filename, df_filename):
    df = pd.read_csv(filename)
    df_feature = pd.read_csv(df_filename)
    app = dash.Dash()
    app.layout = html.Div(
    children=[
        html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),
        html.Br(),
        html.H1("UK Accidents dataset", style={'text-align': 'center'}),
        html.Br(),
        html.Div(),
        html.H1("Figure 1", style={'text-align': 'center'}),
        html.H4("Basic Info about accidents that took place in UK in 2010", style={'text-align': 'center'}),
        dcc.Graph(figure=fig1(df)),
        html.Br(),
        html.Div(),
        html.H1("Figure 2", style={'text-align': 'center'}),
        html.H4("count of accidents in each police force in the uk", style={'text-align': 'center'}),
        dcc.Graph(figure=fig2(df)),
        html.Br(),
        html.Div(),
        html.H1("Figure 3", style={'text-align': 'center'}),
        html.H4("Number of accidents that took place on the top 30 roads of type \"A\"", style={'text-align': 'center'}),
        dcc.Graph(figure=fig3(df)),
    ]
)
    app.run_server(host='0.0.0.0')
    print('dashboard is successful and running on port 8000')