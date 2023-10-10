import plotly.express as px
import pandas as pd

def plot_graphs(X_test, y_pred, y_test, DATEOFFSET):
    X_test['date'] = pd.to_datetime(X_test[['year', 'month', 'day']])
    X_test['y_test'] = y_test
    X_test['y_pred'] = y_pred
    Xdf = pd.melt(X_test, id_vars=['date'], value_vars=['close','y_test', 'y_pred'])
    fig = px.line(Xdf, x='date', y='value', color='variable',  template="plotly_dark", labels={'date': 'Date', 'close': 'Close'})
    # fig.add_(x=X_test['date'], y=y_pred, mode='lines', name='Predictions')
    # fig.add_(x=X_test['date'], y=y_test, mode='lines', name='Actual')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value')
    fig.update_layout(title='Comparison of Predictions and Actual Values', showlegend=True)
    fig.show()
