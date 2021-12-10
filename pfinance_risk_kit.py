import pandas as pd 
def drawdown(return_series: pd.Series): 
    """Takes a time series of asset returns.
       returns a DataFrame with columns for 
       wealth index, previous peaks and 
       percentile drawdown
    """

    wealth_index = 1000 * (1 + return_series).cumprod() 
    previous_peaks = wealth_index.cummax() 
    drawdowns = (wealth_index - previous_peaks) / previous_peaks 
    return pd.DataFrame({
        'wealth': wealth_index,
        'previous_peaks': previous_peaks,
        'drawdown': drawdowns
    })

def get_ffme_returns(): 
    """
    Load the fama-french dataset for the returns of the top and bottom deciles by market_cap
    """
    me_m = pd.read_csv(r"D:\python-finance\coursera_finance\data\Portfolios_Formed_on_ME_monthly_EW.csv", 
    header = 0,
    index_col = 0,
    na_values = -99.99
    )  
    rets = me_m[['Lo 10','Hi 10']] 
    rets.columns = ['small_cap','large_cap'] 
    rets /= 100 
    rets.index = pd.to_datetime(rets.index, format='%Y%m')
    return rets 
