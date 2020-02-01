#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 10 2019

@author: John Zhou
"""
##
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy import stats

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 
import datetime 
from collections import OrderedDict
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")


##
def cov_ewma(ret_assets, lamda = 0.94):
    ret_mat = ret_assets.values
    T = len(ret_assets)
    coeff = np.zeros((T,1))
    S = ret_assets.cov()
    for i in range(1, T):
#       S = lamda * S  + (1-lamda)*np.matmul(ret_mat[i-1,:].reshape((-1,1)),
#                          ret_mat[i-1,:].reshape((1,-1)))
        S = lamda * S  + (1-lamda)* (ret_mat[i-1,:].reshape((-1,1)) @ ret_mat[i-1,:].reshape((1,-1)) )
        
        coeff[i] = (1-lamda)*lamda**(i)
    return S/np.sum(coeff)

    
# risk budgeting approach optimisation object function
def obj_fun(W, cov_assets, risk_budget):
    var_p = np.dot(W.transpose(), np.dot(cov_assets, W))
    sigma_p = np.sqrt(var_p)
    risk_contribution = W*np.dot(cov_assets, W)/sigma_p
    risk_contribution_percent = risk_contribution/sigma_p
    return np.sum((risk_contribution_percent-risk_budget)**2)


# calculate risk budgeting portfolio weight give risk budget
def riskparity_opt(ret_assets, risk_budget, lamda, method='ewma',Wts_min=0.0, leverage=False):
    # number of assets
    num_assets = ret_assets.shape[1]
    # covariance matrix of asset returns
    if method=='ewma':
        cov_assets=cov_ewma(ret_assets, lamda)
    elif method=='ma':
        cov_assets = ret_assets.cov()
    else:
        cov_assets = cov_ewma(ret_assets, lamda)        
    
    # initial weights
    w0 = 1.0 * np.ones((num_assets, 1)) / num_assets
    # constraints
    #cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    if leverage == True:
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-2. }, # Sum of weights = 200%
              {'type':'ineq', 'fun': lambda W: W-Wts_min}) # weights greater than min wts
    else:
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. }, # Sum of weights = 100%
              {'type':'ineq', 'fun': lambda W: W-Wts_min}) # weights greater than min wts
    # portfolio optimisation
    return minimize(obj_fun, w0, args=(cov_assets, risk_budget), method='SLSQP', constraints=c_)


# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))


# function to get the return data calculated from price data 
# retrived from yahoo finance 
def getReturns(tickers, start_dt, end_dt, freq='daily'): 
    px_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    px = px_data[['Adj Close']].reset_index().pivot(index='Date', 
                           columns='Ticker', values='Adj Close')
    if (freq=='daily'):
        px = px.resample('D').last()
        
    # Calculate the daily/monthly percentage change
    ret = px.pct_change().dropna()
    
    ret.columns = tickers
    return(ret)


##
#%% get historical stock price data
if __name__ == "__main__":
    TickerNWeights = pd.read_csv('SSX_2.csv')
    Ticker_AllStock_SS = TickerNWeights['Ticker']
    wts_AllStock_SS = TickerNWeights['MC_Weight']
    Flag_downloadData = True
    # define the time period
    start_dt = datetime.datetime(2007,12,31)
    end_dt = datetime.datetime(2018,12,31)

    if Flag_downloadData:
        stock_data = getDataBatch(Ticker_AllStock_SS, start_dt, end_dt)
        # Isolate the `Adj Close` values and transform the DataFrame
        Price_AllStock_SS = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
        Price_AllStock_SS = Price_AllStock_SS[list(Ticker_AllStock_SS)]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('IndexPrice.xlsx', engine='xlsxwriter')
        Price_AllStock_SS.to_excel(writer, sheet_name='Price', startrow=0, startcol=0, header=True, index=True)
    else:
        Price_AllStock_SS = pd.read_excel('IndexPrice.xlsx', sheet_name='Price',
                                          header=0, index_col=0)
##
    # get price
    ret_AllStock = Price_AllStock_SS.pct_change().dropna()
    ret_SSX = pd.DataFrame.dot(ret_AllStock, np.array(wts_AllStock_SS).reshape(-1))
    # Scale return data by a factor. It seems that the optimizer fails when the values are too close to 0
    scale = 1
    ret_AllStock = ret_AllStock * scale
    ret_SSX = ret_SSX * scale
    
    Ticker_AllAsset = ['GLD','EURUSD=X','AGG','PCY']
    #['GLD','BND','AGG','CNY=X','TAO']
    stock_data = getDataBatch(Ticker_AllAsset, start_dt, end_dt)
            # Isolate the `Adj Close` values and transform the DataFrame
    price_AllAsset = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')       
           
##
    # 2. Calculate returns covs
    ret_assets = price_AllAsset.pct_change().dropna()
    ret_assets['SSX'] = ret_SSX
    ret_assets.dropna()
    ret_assets_demean = ret_assets - ret_assets.mean()
    
    num_assets = ret_assets.shape[1]
    lamda = 0.94
    
    SS = cov_ewma(ret_assets_demean, lamda)
    a = ret_assets.corr()
    SS1 = cov_ewma(ret_assets, lamda)
##
    # Construct risk parity portfolio
    # portfolio dates - this defines the first date of portfolio construction
    datestr_p1 = ret_assets.index[('2009-03-31'>=ret_assets.index) & (ret_assets.index>= '2008-03-31')]
    datestr_p2 = ret_assets.index[ret_assets.index >= '2009-04-01']
    # previous month
    mth_previous_p1 = datestr_p1[0]
    mth_previous_p2 = datestr_p2[0]
    # initialise portfolio weights matrix
    wts_p1 = pd.DataFrame(index=datestr_p1, columns=ret_assets.columns)
    wts_p2 = pd.DataFrame(index=datestr_p1, columns=ret_assets.columns)
    # initialise portfolio return matrix
    ret_riskParity_p1 = pd.DataFrame(index=datestr_p1, columns=['Risk Parity'])
    ret_riskParity_p2 = pd.DataFrame(index=datestr_p2, columns=['Risk Parity'])
    # how many rolling calendar days to use for covariance calculation
    window = 90
    Wts_min = 0.1
    risk_budget = 1.0/num_assets*np.ones([1,num_assets]) #risk-party
    #risk_budget = [0.7, 0.4]
    leverage = True
    varmodel = 'ma'

##

    for t in datestr_p1:
        # construct risk budgeting portfolio and re-balance on monthly basis
        if t.month == mth_previous_p2:
            # keep the same portfolio weights within the month
            wts_p1.loc[t] = wts_p2.iloc[wts_p2.index.get_loc(t) - 1]
        else:
            # update the value of the previous month
            mth_previous_p1 = t.month
            # re-balance the portfolio at the start of the month

            t_begin = t - timedelta(days=window)
            ret_used = ret_assets.loc[t_begin:t, :]
            wts_p1.loc[t] = riskparity_opt(ret_used, risk_budget, lamda, varmodel, Wts_min, leverage).x
        # calculate risk budgeting portfolio returns
        ret_riskParity_p1.loc[t] = np.sum(wts_p1.loc[t] * ret_assets.loc[t])

    # Due to precision issue, wts could be a tiny negative number instead of zero, make them zero
    wts_p1[wts_p1 < 0] = 0.0
    # Construct equal weighted portfolio
    ret_equalwted_p1 = pd.DataFrame(np.sum(1.0 * ret_assets[('2009-03-31'>=ret_assets.index) & (ret_assets.index>= '2008-03-31')] / num_assets, axis=1),
                                    columns=['Equal Weighted'])
##
    for t in datestr_p2:
        # construct risk budgeting portfolio and re-balance on monthly basis
        if t.month==mth_previous_p2:
            # keep the same portfolio weights within the month
            wts_p2.loc[t] = wts_p2.iloc[wts_p2.index.get_loc(t)-1]
        else:
            # update the value of the previous month 
            mth_previous = t.month
            # re-balance the portfolio at the start of the month
            
            t_begin = t - timedelta(days=window)
            ret_used = ret_assets.loc[t_begin:t,:]
            wts_p2.loc[t] = riskparity_opt(ret_used, risk_budget, lamda, varmodel, Wts_min, leverage).x
        # calculate risk budgeting portfolio returns
        ret_riskParity_p2.loc[t] = np.sum(wts_p2.loc[t] * ret_assets.loc[t])
        
    # Due to precision issue, wts could be a tiny negative number instead of zero, make them zero
    wts_p2[wts_p2<0]=0.0
    # Construct equal weighted portfolio
    ret_equalwted_p2 = pd.DataFrame(np.sum(1.0*ret_assets[ret_assets.index >= datestr_p2[0]]/num_assets, axis=1), columns=['Equal Weighted'])
    # Construct 60/40 weighted portfolio
    #ret_equalwted = pd.DataFrame(np.sum(1.0*ret_assets[ret_assets.index>=datestr[0]]/num_assets, axis=1), columns=['Equal Weighted'])




#%%
    # Calculate performance stats
    ret_cumu_assets_p1 = (ret_assets[('2009-03-31'>=ret_assets.index) & (ret_assets.index>= '2008-03-31')] + 1).cumprod()
    ret_cumu_riskP_p1 = (ret_riskParity_p1 + 1).cumprod()
    ret_cumu_equalwt_p1 = (ret_equalwted_p1 + 1).cumprod()
    
    ret_annual_assets_p1 = ret_cumu_assets_p1.iloc[-1]**(250/len(ret_cumu_assets_p1))-1
    std_annual_assets_p1 = ret_assets[('2009-03-31'>=ret_assets.index) & (ret_assets.index>= '2008-03-31')].std()*np.sqrt(250)
    sharpe_ratio_assets_p1 = ret_annual_assets_p1/std_annual_assets_p1
    b1=sharpe_ratio_assets_p1

    ret_annual_riskP_p1 = ret_cumu_riskP_p1.iloc[-1]**(250/len(ret_cumu_riskP_p1))-1
    std_annual_riskP_p1 = ret_riskParity_p1.std()*np.sqrt(250)
    sharpe_ratio_riskP_p1 = ret_annual_riskP_p1/std_annual_riskP_p1
    
    ret_annual_equalwt_p1 = ret_cumu_equalwt_p1.iloc[-1]**(250/len(ret_cumu_equalwt_p1))-1
    std_annual_equalwt_p1 = ret_equalwted_p1.std()*np.sqrt(250)
    sharpe_ratio_equalwt_p1 = ret_annual_equalwt_p1/std_annual_equalwt_p1
    
    #sharpe_table = [sharpe_ratio_riskP, sharpe_ratio_equalwt]
    sharpe_table_p1 = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP_p1.values),
                     ('equal_wted', sharpe_ratio_equalwt_p1.values),
                     )))
    sharpe_table1_p1 = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP_p1.values),
                                           ('SSX', sharpe_ratio_assets_p1['SSX']),
                                           ('GLD', sharpe_ratio_assets_p1['GLD']),
                                           ('AGG', sharpe_ratio_assets_p1['AGG']),
#                                           ('TAO', sharpe_ratio_assets_p1['TAO']),
                                           ('PCY', sharpe_ratio_assets_p1['PCY']),
                                           ('EURUSD=X', sharpe_ratio_assets_p1['EURUSD=X'])
                                           )))
    print('sharpe ratio of different strategies:\n',sharpe_table_p1)
    print('\nsharpe ratio of strategies vs assets:\n',sharpe_table1_p1)

##
    # Calculate performance stats
    ret_cumu_assets_p2 = (ret_assets[ret_assets.index >= '2009-04-01'] + 1).cumprod()
    ret_cumu_riskP_p2 = (ret_riskParity_p2 + 1).cumprod()
    ret_cumu_equalwt_p2 = (ret_equalwted_p2 + 1).cumprod()

    ret_annual_assets_p2 = ret_cumu_assets_p2.iloc[-1] ** (250 / len(ret_cumu_assets_p2)) - 1
    std_annual_assets_p2 = ret_assets[ret_assets.index >= '2009-04-01'].std() * np.sqrt(250)
    sharpe_ratio_assets_p2 = ret_annual_assets_p2 / std_annual_assets_p2
    b2 = sharpe_ratio_assets_p2

    ret_annual_riskP_p2 = ret_cumu_riskP_p2.iloc[-1] ** (250 / len(ret_cumu_riskP_p2)) - 1
    std_annual_riskP_p2 = ret_riskParity_p2.std() * np.sqrt(250)
    sharpe_ratio_riskP_p2 = ret_annual_riskP_p2 / std_annual_riskP_p2

    ret_annual_equalwt_p2 = ret_cumu_equalwt_p2.iloc[-1] ** (250 / len(ret_cumu_equalwt_p2)) - 1
    std_annual_equalwt_p2 = ret_equalwted_p2.std() * np.sqrt(250)
    sharpe_ratio_equalwt_p2 = ret_annual_equalwt_p2 / std_annual_equalwt_p2

    # sharpe_table = [sharpe_ratio_riskP, sharpe_ratio_equalwt]
    sharpe_table_p2 = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP_p2.values),
                                             ('equal_wted', sharpe_ratio_equalwt_p2.values),
                                             )))
    sharpe_table1_p2 = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP_p2.values),
                                              ('SSX', sharpe_ratio_assets_p2['SSX']),
                                              ('GLD', sharpe_ratio_assets_p2['GLD']),
                                              ('AGG', sharpe_ratio_assets_p2['AGG']),
    #                                          ('TAO', sharpe_ratio_assets_p2['TAO']),
                                              ('PCY', sharpe_ratio_assets_p2['PCY']),
                                              ('EURUSD=X', sharpe_ratio_assets_p2['EURUSD=X'])
                                              )))
    print('sharpe ratio of different strategies:\n', sharpe_table_p2)
    print('\nsharpe ratio of strategies vs assets:\n', sharpe_table1_p2)

##

#%%
# compare the portfolio cumulative returns
figure_count = 1
plt.figure(figure_count)
figure_count = figure_count+1
pd.concat([ret_cumu_riskP_p1, ret_cumu_equalwt_p1], axis=1).plot()
plt.ylabel('Cumulative Return during crisis')
plt.show()

# compare the portfolio cumulative returns vs. asset returns
plt.figure(figure_count)
figure_count = figure_count+1
pd.concat([ret_cumu_riskP_p1, ret_cumu_assets_p1], axis=1).plot()
plt.ylabel('Cumulative Return during crisis')
plt.show()

# plot the historical weights of the assets
# area plot showing the weights
plt.figure(figure_count)
figure_count = figure_count + 1
wts_p1.plot.area()
plt.ylabel('asset weights during crisis')

# compare the portfolio cumulative returns
plt.figure(figure_count)
figure_count = figure_count+1
pd.concat([ret_cumu_riskP_p2, ret_cumu_equalwt_p2], axis=1).plot()
plt.ylabel('Cumulative Return After Crisis')
plt.show()

# compare the portfolio cumulative returns vs. asset returns
plt.figure(figure_count)
figure_count = figure_count+1
pd.concat([ret_cumu_riskP_p2, ret_cumu_assets_p2], axis=1).plot()
plt.ylabel('Cumulative Return After Crisis')
plt.show()

# plot the historical weights of the assets
# area plot showing the weights
plt.figure(figure_count)
figure_count = figure_count + 1
wts_p2.plot.area()
plt.ylabel('asset weights After Crisis')

