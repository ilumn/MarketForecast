import mplfinance as mpf
import pandas as pd
import mplcyberpunk


def plot_graphs(X_test, y_pred, y_test, DATEOFFSET):
    X_test['date'] = pd.to_datetime(X_test[['year', 'month', 'day']])
    # mpf.style.use("cyberpunk")
    df = X_test.set_index('date')
    
    fig, axes = mpf.plot(df, type='candle', style='sas', mav=(3,6,9), title='Comparison of Predictions and Actual Values', figsize=(20, 10))
    mpf.plot(df, type='line', panel=1, color='blue', secondary_y=True)
    mpf.plot(df, type='line', panel=1, color='red', secondary_y=True)
    
    for ax in axes:
        ax.set_xlabel('Date')
    
    # mplcyberpunk.add_glow_effects(gradient_fill=False)
    # plt.tight_layout()
    plt.show()
