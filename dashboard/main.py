import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go


df = pd.read_csv('test.csv')


app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Stock Data Dashboard"),
    
    
    dcc.Dropdown(
        id='stock-selector',
        options=[{'label': ticker, 'value': ticker} for ticker in df['ticker'].unique()],
        value=df['ticker'].unique()
    ),
    
    dcc.Graph(id='stock-graph'),
])

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('stock-selector', 'value')]
)
def update_graph(selected_stocks):
    filtered_df = df[df['ticker'].isin(selected_stocks)]
    
    yearly_data = filtered_df.groupby(['year', 'ticker'])['close'].mean().reset_index()
    
    traces = []
    for stock in selected_stocks:
        trace = go.Scatter(
            x=yearly_data[yearly_data['ticker'] == stock]['year'],
            y=yearly_data[yearly_data['ticker'] == stock]['close'],
            mode='lines',
            name=stock
        )
        traces.append(trace)
    
    layout = go.Layout(title='Yearly Stock Prices', xaxis={'title': 'Year'}, yaxis={'title': 'Average Close Price'})
    
    return {'data': traces, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
