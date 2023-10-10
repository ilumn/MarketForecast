# visualization.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplcyberpunk


def plot_graphs(X_test, y_pred, y_test, DATEOFFSET):
    # Create a figure with subplots
    X_test['date'] = pd.to_datetime(X_test[['year', 'month', 'day']])

    plt.style.use("cyberpunk")
    fig = plt.figure(figsize=(20, 10))
    
    # Create a 2x4 grid of subplots
    ax1 = plt.subplot2grid((2, 4), (0, 0))
    ax2 = plt.subplot2grid((2, 4), (0, 1))
    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax5 = plt.subplot2grid((2,4), (1,0), colspan=3)

    # Plot 'close' from X_test by date
    ax1.plot(X_test['date'], X_test['close'], color='purple', label='close')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('close')
    ax1.legend()

    # Plot y_pred by date
    ax2.plot(X_test['date'], y_pred, color='blue', label='y_pred')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('y_pred')
    ax2.legend()

    # Plot y_test by date
    ax3.plot(X_test['date'], y_test, color='red', label='y_test')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('y_test')
    ax3.legend()
        
    # Plot y_pred vs y_test
    ax4.scatter(y_pred, y_test, color='green')
    ax4.set_xlabel('y_pred')
    ax4.set_ylabel('y_test')



    # Plot 'close' from X_test by date (larger plot)
    ax5.plot(X_test['date'], X_test['close'], color='purple', label='Close (X_test)')
    ax5.plot((X_test['date'] + pd.DateOffset(days=DATEOFFSET)), y_pred, color='blue', label='Predictions (y_pred)')
    ax5.plot((X_test['date'] + pd.DateOffset(days=DATEOFFSET)), y_test, color='red', label='Actual (y_test)')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Value')
    ax5.legend()
    ax5.set_title('Comparison of Predictions and Actual Values')
    ax5.grid(True)
    ax5.set_xlabel('Date')
    ax5.set_ylabel('close')
    ax5.legend()

    mplcyberpunk.add_glow_effects(gradient_fill=False)
    plt.tight_layout()
    
    plt.show()
