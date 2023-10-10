from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.io import curdoc
import pandas as pd

def plot_graphs(X_test, y_pred, y_test, DATEOFFSET):
    X_test['date'] = pd.to_datetime(X_test[['year', 'month', 'day']])
    curdoc().theme = 'dark_minimal'
    p1 = figure(title='Close', x_axis_label='Date', y_axis_label='Close')
    p1.line(X_test['date'], X_test['close'], line_color='purple', legend_label='Close')
    
    p2 = figure(title='Predictions', x_axis_label='Date', y_axis_label='y_pred')
    p2.line(X_test['date'], y_pred, line_color='blue', legend_label='y_pred')
    
    p3 = figure(title='Actual', x_axis_label='Date', y_axis_label='y_test')
    p3.line(X_test['date'], y_test, line_color='red', legend_label='y_test')
    
    p4 = figure(title='y_pred vs y_test', x_axis_label='y_pred', y_axis_label='y_test')
    p4.scatter(y_pred, y_test, color='green', legend_label='y_pred vs y_test')
    
    p5 = figure(title='Comparison of Predictions and Actual Values', x_axis_label='Date', y_axis_label='Value')
    p5.line(X_test['date'], X_test['close'], line_color='purple', legend_label='Close (X_test)')
    p5.line(X_test['date'] + pd.DateOffset(days=DATEOFFSET), y_pred, line_color='blue', legend_label='Predictions (y_pred)')
    p5.line(X_test['date'] + pd.DateOffset(days=DATEOFFSET), y_test, line_color='red', legend_label='Actual (y_test)')
    
    grid = gridplot([[p1, p2, p3, p4], [p5]])
    
    show(grid)
