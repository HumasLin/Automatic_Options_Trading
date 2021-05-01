import wrds
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import date, datetime
from dateutil.relativedelta import relativedelta, FR
from query_wrds import *
from util import sec_id
import warnings
warnings.filterwarnings("ignore")

def clean(data):
    '''
    clean up data above
    Return dataframe with date, symbol, best_bid, best_offer, impl_volatility, delta, theta
    '''
    data['price'] = (data['best_bid'] + data['best_offer'])/2
    data['days_mat'] = (data['exdate'] - data['date']) / np.timedelta64(1,'D')
    data['days_mat'] = data['days_mat'].astype(int)
    data['months_mat'] = (data['exdate'] - data['date']) / np.timedelta64(1,'M')
    data['months_mat'] = data['months_mat'].astype(int)
    return data

# Rebalance an at-the-money option every month -- but one that does NOT expire that month. 
# It should be far enough from expiration to allow options to be traded across months, but close enough to make it interesting (to capture time decay)
'''
Filter dataset for:
Only SPX options (no weekly options)
Maturity: between 21-44'ish days
Type: Calls
Money: ATM (delta is closest to 0.5)

Steps:
1. Filter for symbol contains 'SPX ', maturity = 1, and cp_flag = C
2. Groupby exdate. Keep the contract whose delta is closest to 0.5
'''
def get_atm_prices(data, sec, min_days, max_days, moneyness):
    '''
    - This function finds the ATM option at some maximum days from each month's expiration date.
    - Then it tracks the prices of that ATM option until min_days from expiration is reached
    - Then calculate the next month's ATM option (again starting at max_days to the next month's expiration)
    - The idea is track the prices of ATM options, rebalanced every month

    Input: dataframe of uncleaned data, min days to maturity, max days to maturity
    Output: prices of options that start ATM at 'max_days' to maturity
    '''

    # filter for 'SPX ' only
    data = data[data['symbol'].str.contains(sec_id[sec])] # avoid weekly options like SPXW
    # clean dataframe and add days to maturity and months to maturity
    data = clean(data)

    # filter by days to expiration (21-42) EDIT: try 46 for more contiguous data
    df_filt = data[data['days_mat'].between(min_days, max_days, inclusive=True)]

    # groupby exdate
    df_g = df_filt.groupby(['exdate'])

    # empty dataframe to populate later
    df_atm = pd.DataFrame(columns=data.columns)

    for dt, g in df_g:
        # get max value of days_mat
        max_d = g['days_mat'].max()

        # filter g by this max days_mat, then identify the ATM option. 
        # Therefore,find row where strike price is closest to forward price
        # Note: for calls, if the stock plummets, the price collapses to near zero, making future rebalances near zero. 
        # I think it's better to choose an ITM call instead, like 30% or so
        g_max = g[g['days_mat'] == max_d]
        atm_contract = g_max.iloc[(g_max['strike_price']/1000 - (1 - moneyness)*g_max['forward_price']).abs().argsort()[:1]]
        symb_needed = atm_contract['symbol'].values[0]
        
        # get prices of this option contract
        atm_contract_prices = g[g['symbol'] == symb_needed]
        # calculate daily returns
        atm_contract_prices['daily_return'] = atm_contract_prices['price'] / atm_contract_prices['price'].shift(1) - 1
        # replace nans with zeros
        atm_contract_prices['daily_return'] = atm_contract_prices['daily_return'].fillna(0)
        
        # append to df_atm
        df_atm = pd.concat([df_atm, atm_contract_prices], axis=0)
        # print(df_atm)

    return df_atm

def data_engineer(secid):

	dfs = get_data(secid)

	for i in range(10):
	    dfs[i] = get_atm_prices(dfs[i], secid, 21, 46, 0.3)

	# append dataframes
	df_atm_prices_all = pd.concat(dfs, axis=0)

	df_atm_prices_all.shape # (1820, 15)
	df_atm_prices_all.head()

	# calculate adjusted prices using daily option price returns
	start_px = df_atm_prices_all['price'].tolist()[0]
	returns = df_atm_prices_all['daily_return'].tolist()

	adj_prices = []
	m=0
	for i in returns:
	    if m==0:
	        adj_prices.append(start_px)
	        prev_px = start_px
	    else:
	        adj_px = (1 + i) * prev_px
	        adj_prices.append(adj_px)
	        prev_px = adj_px
	    m += 1

	df_atm_prices_all['adj_prices'] = adj_prices

	df_train = df_atm_prices_all[df_atm_prices_all['date']<datetime.strptime('2019-01-01','%Y-%m-%d').date()]
	df_test = df_atm_prices_all[df_atm_prices_all['date']>=datetime.strptime('2019-01-01','%Y-%m-%d').date()]

	return df_train, df_test
