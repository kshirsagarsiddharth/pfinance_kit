import pandas as pd 
import scipy.stats as st
import numpy as np 
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

def get_hfi_returns(): 
    """
    Load and format edhec hedge fund index returns 
    """
    hifi = pd.read_csv(r"D:\python-finance\coursera_finance\data\edhec-hedgefundindices.csv", 
    header = 0,
    index_col = 0,
    parse_dates= True 
    )  
    
    hifi /= 100 
    hifi.index.freq = 'M'
    return hifi

def skewness(r):
    """
    Alternative to scipy.stats.skew() 
    Computes the skewness of the supplied series or dataframe 
    returns a float or a Series
    """
    demeaned_r = r - r.mean() 
    sigma_r = r.std(ddof = 0) 
    exp = (demeaned_r ** 3).mean() 
    return exp / sigma_r ** 3 
00
def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis() 
    Computes the kurtosis of the supplied series or dataframe 
    returns a float or a Series
    """
    demeaned_r = r - r.mean() 
    sigma_r = r.std(ddof = 0) 
    exp = (demeaned_r ** 4).mean() 
    return exp / sigma_r ** 4

def semi_deviation(r): 
    """
    Returns the semideviation i.e standard deviation of the 
    negative values of returns. where r must me a Series or a 
    data frame
    """
    is_negative = r < 0 
    return r[is_negative].std(ddof = 0) 


def is_normal(r, level = 0.01):
    """
    Applies the jarque-Bera test to determine os a series is normal or not 
    This test is applied at 1% level by default. 
    Returns True of hypothesis normality is accepted, Flase otherwise

    H0: Skewness and Kurtosis follow a normal distribution
    """
    _, p_value = st.jarque_bera(r)

    return p_value > level 

def var_historic(r, level = 5): 
    """
    Returns the historic Value at Risk at a specified level 
    i.e. returns the number such that "level" percent of the returns
    fall below that number and the (100 - level) percent are above
    """
    # so there is 5% chance that in any given month we are going to loose about <8%> or worse 
    # for short selling 
    if isinstance(r, pd.DataFrame): 
        return r.apply(lambda x : np.percentile(x,5), axis = 0) * -1
    elif isinstance(r, pd.Series): 
        return np.percentile(r,5) * -1 
    else:
        raise TypeError('Expected Pandas Series or a DataFrame') 




def var_parametric_gaussian(r, level = 5): 
    """
    Returns the Parametruc Gaussian VaR of a Series or a DataFrame column 
    wise 
    """
    # compute the z-score assuming the distribution was normal with given 
    # and standard deviation 

    # so there is 5% chance that in any given month we are going to loose about <8%> or worse 
    # for short selling 
    level /= 100 
    if isinstance(r, pd.DataFrame): 
        return r.apply(lambda x : st.norm.ppf(level, x.mean(), x.std())) * -1 
    elif isinstance(r, pd.Series):
        return st.norm.ppf(level, r.mean(), r.std()) * -1 
    else:
        raise TypeError('Expected Pandas Series or a DataFrame')

def var_parametric_cornsih_fisher(r,level = 5):
    """
    Returns the modified VaR using Cornish-Fisher modification
    """
    # so there is 5% chance that in any given month we are going to loose about <8%> or worse 
    # for short selling 
    z = st.norm.ppf(level/100) 
    s = st.skew(r) 
    k = st.kurtosis(r) 
    z = (z +
            (z**2 - 1)*s/6 +
            (z**3 -3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
        )
    return -(r.mean() + z*r.std(ddof = 0))

def cvar_historic(r, level= 5): 
    """
    Computes the Conditional VaR of a Series or a DataFrame
    """
    # if thet five percent chance happens i.e worst 5% of the possible cases 
    # the average of that is <3.6%> loss 
    is_beyond = r <= var_historic(r, level = level) * -1 
    return r[is_beyond].mean() * -1 
    


    























