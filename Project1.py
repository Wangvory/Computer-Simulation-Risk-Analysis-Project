# %%
# -*- coding: utf-8 -*-
"""
Created on Nov 18 2019

@author: John Zhou & Shiwen Li

This code gives examples of how to 
    1. forecast tracking error
    2. constructing an index replication strategy by minimizing tracking error with limited # of stocks
    3. Integer Optimizer
    4. plot adjustment
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
# pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()
import datetime
import scipy
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# load module with utility functions, including optimization
import risk_opt_2Student as riskopt


def tracking_error(wts_active, cov):
    TE = np.sqrt(np.transpose(wts_active) @ cov @ wts_active)
    return TE

# function to get the price data from yahoo finance
def getDataBatch(tickers, startdate, enddate):
    def getData(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
    datas = map(getData, tickers)
    return (pd.concat(datas, keys=tickers, names=['Ticker', 'Date'], sort=False))


# function to get the return data calculated from price data
# retrived from yahoo finance
def getReturns(tickers, start_dt, end_dt, freq='monthly'):
    px_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    px = px_data[['Adj Close']].reset_index().pivot(index='Date',
                                                    columns='Ticker', values='Adj Close')
    if (freq == 'monthly'):
        px = px.resample('M').last()

    # Calculate the daily/monthly percentage change
    ret = px.pct_change().dropna()

    ret.columns = tickers
    return (ret)
##
#Here is the Integer Optimizer
def binary_permutations(lst):
    for comb in combinations(range(len(lst)), lst.count(1)):
        result = [[0] * 2] * len(lst)
        for i in comb:
            result[i] = [0, 1]
        yield result

def constrain_creater(n):
    b_ = []
    # a=[[0]*2]*30
    a = np.zeros(15, dtype=int)
    a[:n] = 1
    for perm in binary_permutations(list(a)):
        b_.append(perm)
    return b_
##
def factorial(lst):
    factlst=np.array([])
    for n in range(len(lst)):
        fact = np.prod(lst[:n]+1)
        factlst = np.append(factlst,fact)
    return factlst
# %%

if __name__ == "__main__":

    TickerNWeights = pd.read_csv('SSX.csv')
    Ticker_AllStock_SS = TickerNWeights['Ticker']
    wts_AllStock_SS = TickerNWeights['MC_Weight']
    #if run code below,our index becomes equal weighted
    #wts_AllStock_SS = TickerNWeights['Equal_Weight']
    TickerNWeights_DJ = pd.read_excel('EquityIndexWeights.xlsx', sheet_name='DowJones', header=2, index_col=0)
    Ticker_AllStock_DJ = TickerNWeights_DJ['Symbol']
    wts_AllStock_DJ = 0.01 * TickerNWeights_DJ['Weight']

    # %% get historical stock price data

    Flag_downloadData = True
    # define the time period
    start_dt = datetime.datetime(2007,1,1)
    end_dt = datetime.datetime(2018,12,31)

    if Flag_downloadData:
        #
        stock_data = getDataBatch(Ticker_AllStock_SS, start_dt, end_dt)
        stock_data_DJ = getDataBatch(Ticker_AllStock_DJ, start_dt, end_dt)
        # Isolate the `Adj Close` values and transform the DataFrame
        Price_AllStock_SS = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
        Price_AllStock_SS = Price_AllStock_SS[list(Ticker_AllStock_SS)]
        Price_AllStock_DJ = stock_data_DJ.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
        Price_AllStock_DJ = Price_AllStock_DJ[list(Ticker_AllStock_DJ)]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('IndexPrice.xlsx', engine='xlsxwriter')
        Price_AllStock_SS.to_excel(writer, sheet_name='Price', startrow=0, startcol=0, header=True, index=True)
    else:
        Price_AllStock_SS = pd.read_excel('IndexPrice.xlsx', sheet_name='Price',
                                          header=0, index_col=0)
    # %%
    # Returns
    ret_AllStock = Price_AllStock_SS.pct_change().dropna()
    ret_AllStock_DJ = Price_AllStock_DJ.pct_change().dropna()
    ret_SSX = pd.DataFrame.dot(ret_AllStock, np.array(wts_AllStock_SS).reshape(-1))
    ret_DJ = pd.DataFrame.dot(ret_AllStock_DJ, np.array(wts_AllStock_DJ).reshape(-1))
    # Scale return data by a factor. It seems that the optimizer fails when the values are too close to 0
    scale = 1
    ret_AllStock = ret_AllStock * scale
    ret_SSX = ret_SSX * scale
    value_SSX = factorial(ret_SSX)
    value_DJ = factorial(ret_DJ)
    #
    figure_count = 1
    plt.figure(figure_count)
    figure_count = figure_count + 1
    date = ret_AllStock.index.values
    dateDJ = ret_AllStock_DJ.index.values
    num_periods, num_stock = ret_AllStock.shape
    plt.plot(date, value_SSX)
    plt.plot(dateDJ, value_DJ)
    plt.show()

    plt.figure(figure_count)
    figure_count = figure_count + 1
    plt.plot(date, ret_SSX*100)
    plt.xlabel('Time')
    plt.ylabel('Index Return(%)')
    plt.title('Historical Index Return')
    plt.show()

    # %%
    # Calulate Covariance Matrix
    #

    lamda = 0.94
    # vol of the assets
    vols = ret_AllStock.std()
    rets_mean = ret_AllStock.mean()
    # demean the returns
    ret_AllStock = ret_AllStock - rets_mean

    # var_ewma calculation of the covraiance using the function from module risk_opt.py
    var_ewma = riskopt.ewma_cov(ret_AllStock, lamda)
    # take only the covariance matrix for the last date, which is the forecast for next time period
    cov_end = var_ewma[-1, :]
    #
    cov_end_annual = cov_end * 252  # Annualize, we are trying to focus annual tracking error
    std_end_annual = np.sqrt(np.diag(cov_end)) * np.sqrt(252)
    # calculate the correlation matrix
    corr = ret_AllStock.corr()

    # %%
    # tracking error optimization
    #
    #
    # Test case - use only the top market cap stocks with highest index weights
    #
    # NOT only the top weight stocks + no shorting, but comes with trail and error

    def opt_min_te_n(n,wts,cov):
        b = constrain_creater(n)
        wtslist = []
        TElist = []
        for i in b:
            b1_ = i
            # b1_[num_topwtstock_2include:-1] = (0.0,0.0)
            c1_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # Sum of active weights = 100%
            # Calling the optimizer
            wts_min_trackingerror2 = riskopt.opt_min_te(wts, cov, b1_, c1_)
            # calc TE achieved
            wts_active2 = wts_min_trackingerror2 - wts
            TE_optimized2 = tracking_error(wts_active2, cov)
            wtslist.append(wts_min_trackingerror2)
            TElist.append(TE_optimized2)
        index_min = np.argmin(TElist)
        opt_wst=wtslist[index_min]
        opt_TE=TElist[index_min]
        return opt_TE * 10000,opt_wst

    #Example optimized TE when n=5
    opt_TE, opt_wst = opt_min_te_n(5, wts_AllStock_SS, cov_end_annual)
    print('\nfull replication TE = {0:.5f} bps'.format(opt_TE))

    # looping through number of stocks and save the history of TEs
    num_stock_b = 5
    num_stock_e = 11
    numstock_2use = range(num_stock_b, num_stock_e)
    wts_active_hist = np.zeros([len(numstock_2use), num_stock])
    TE_hist = np.zeros([len(numstock_2use), 1])
    wts_hist = np.zeros([len(numstock_2use), 15])
    count = 0

    for i in numstock_2use:
        TE_optimized_c,wts_optimized_c = opt_min_te_n(i,wts_AllStock_SS, cov_end_annual)
        TE_hist[count, :] = TE_optimized_c  # in bps
        wts_hist[count, :] = wts_optimized_c
        count = count + 1

    plt.figure(figure_count)
    figure_count = figure_count+1
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(range(num_stock_b,num_stock_e), TE_hist, 'b')
    plt.xlabel('Number of stocks in ETF', fontsize=18)
    plt.ylabel('Optimized Tracking Error (bps)', fontsize=18)
    plt.title('SSX Index ETF Tracking Error', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    #n=10 have the minimum tracking error

    #
    # %%%
    #  Plot bars of weights
    # n=5
    # ---  create plot of weights fund vs benchmark
    plt.figure(figure_count)
    figure_count = figure_count + 1
    fig, ax = plt.subplots(figsize=(18, 10))
    index = np.arange(len(wts_AllStock_SS))
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, wts_AllStock_SS, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Index Weight')

    rects2 = plt.bar(index + bar_width, wts_hist[1], bar_width,
                     alpha=opacity,
                     color='b',
                     label='ETF fund Weight when n=6')

    rects3 = plt.bar(index + 2 * bar_width, wts_hist[2], bar_width,
                     alpha=opacity,
                     color='g',
                     label='ETF fund Weight when n=7')

    rects3 = plt.bar(index + 3 * bar_width, wts_hist[3], bar_width,
                     alpha=opacity,
                     color='b',
                     label='ETF fund Weight when n=8')


    plt.xlabel('Ticker', fontsize=18)
    plt.ylabel('Weights', fontsize=18)
    plt.xticks(index + bar_width, (Ticker_AllStock_SS), fontsize=12)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=18)
    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.show()

    # ------plot TE as a function of number of stocks -------------
    plt.figure(figure_count)
    figure_count = figure_count + 1
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(range(num_stock_b, num_stock_e), TE_hist, 'b')
    plt.xlabel('Number of stocks in ETF', fontsize=18)
    plt.ylabel('Optimized Tracking Error (bps)', fontsize=18)
    plt.title('Stock Sensitivity 15 ETF', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

# %%
    # ------Train and Test the Data ---------

    # Calulate Covariance Matrix
    window = 30
    start_dt_training = datetime.datetime(2007, 1, 1)
    end_dt_training = datetime.datetime(2013, 1, 1)
    start_dt_test= datetime.datetime(2013, 1, 2)
    end_dt_test = datetime.datetime(2018, 12,31)
    ret_AllStock_total = ret_AllStock.loc[start_dt_training:end_dt_test,:]  # both training and test data
    datestr = ret_AllStock_total.index[ret_AllStock_total.index >= start_dt_test]
    mth_previous = datestr[0] - timedelta(days=window)
    wts1 = pd.DataFrame(index=datestr, columns=ret_AllStock.columns)

    TE_optimized_used = {}
    TE_realized_used = {}
    t_begin = start_dt_test

    for t in datestr:
        if t.month != mth_previous:
            # update the value of the previous month
            mth_previous = t.month
            # re-balance the portfolio at the start of the month

            ret_train = ret_AllStock.loc[start_dt_training:t_begin, :]

            t_test = t_begin + timedelta(days=window)
            ret_test = ret_AllStock.loc[t_begin:t_test, :]

            vols_train = ret_train.std()
            ret_mean_train = ret_train.mean()
            ret_train_demean = ret_train - ret_mean_train
            # ewma_train calculation of the covraiance using the function from module risk_opt.py
            ewma_train = riskopt.ewma_cov(ret_train_demean, lamda)
            #Annualize # take only the covariance matrix for the last date, which is the forecast for next time period
            cov_train = ewma_train[-1, :]
            cov_train_annual = cov_train * 252  # Annualize
            std_train_annual = np.sqrt(np.diag(cov_train)) * np.sqrt(252)
            corr_train = ret_train.corr()

            cov_realized = ret_test.cov().values
            cov_realized_annual = cov_realized * 252

            # calling the optimization function
            b1a_ = [(0.0, 1.0) for i in range(0,10)]
            b1b_ = [(0.0, 0.0) for i in range(10,15)]
            b_ = b1a_ + b1b_
            #change the b_ accpring to our n slelection
            #noticeably when n=6&8 it is not totally follow the weight.
            #Example: b_6=[(0.0, 1.0) for i in range(0,5)]+
            # [(0.0, 0.0) for i in range(5,7)]+
            # [(0.0, 1.0) for i in range(7,8)]+
            # [(0.0, 0.0) for i in range(8,15)]
            c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
            wts_min_trackingerror = riskopt.opt_min_te(wts_AllStock_SS, cov_train_annual, b_, c_)
            wts_slice = pd.DataFrame(wts_min_trackingerror, Ticker_AllStock_SS)
            wts_slice = wts_slice.T
            wts1.loc[t] = wts_slice.values

            # calc TE achieved
            wts_active1 = wts_min_trackingerror - wts_AllStock_SS
            TE_optimized = tracking_error(wts_active1, cov_train_annual)
            TE_realized = tracking_error(wts_active1, cov_realized_annual)
            TE_optimized_used[t_begin] = TE_optimized
            TE_realized_used[t_begin] = TE_realized
            t_begin = t + timedelta(days=window)
            #print(t_begin)

        else:
            # keep the same portfolio weights within the month
            wts1.loc[t] = wts1.iloc[wts1.index.get_loc(t) - 1]
            TE_optimized_used[t] = TE_optimized
            TE_realized_used[t] = TE_realized


    TE_optimized_All = pd.DataFrame(TE_optimized_used, index=[0]).T
    TE_optimized_All.columns = ['TE_optimized']

    TE_realized_All = pd.DataFrame(TE_realized_used, index=[0]).T
    TE_realized_All.colums = ['TE_realiezed']

# %%
    date = TE_optimized_All.index.values
    realizedTE = TE_realized_All.iloc[:, 0] * 10000
    forecastTE = TE_optimized_All.iloc[:, 0] * 10000

    plt.figure(figure_count)
    figure_count = figure_count + 1
    plt.plot(date, realizedTE, linewidth=2, label='Real Tracking Error', alpha=0.8)
    plt.legend()
    plt.plot(date, forecastTE, linewidth=2, label='Forecast Tracking Error', alpha=0.8)
    plt.legend()
    plt.title('Tracking Error Comparison ')
    plt.xlabel('Date')
    plt.ylabel('Tracking Error(bps)')
    plt.show()

    plt.figure(figure_count)
    figure_count = figure_count + 1
    error = realizedTE - forecastTE
    plt.plot(date, error, linewidth=2, color='r', label='$Error$', alpha=0.8)
    plt.title('Realized and Forecast Error')
    plt.xlabel('Date')
    plt.ylabel('Error (bps)')
    plt.show()

    plt.figure(figure_count)
    figure_count = figure_count + 1
    fig, ax = plt.subplots(figsize=(12, 8))
    wts_date = wts1.index.values.tolist()
    wts_date = pd.DataFrame(wts_date)
    for i in range(15):
        label = Ticker_AllStock_SS[i]
        plt.plot(wts_date, wts1.iloc[:, i] * 100, linewidth=2, label=label)
    plt.legend(loc='UpperRight', bbox_to_anchor=(1, 1.05))
    plt.title('Weights Changes')
    plt.xlabel('Date')
    plt.ylabel('Weights(%)')
    plt.show()

    plt.figure(figure_count)
    figure_count = figure_count + 1
    wts_DropZero = wts1.loc[:, (wts1 != 0).any(axis=0)]
    wts_DropZero.plot.area()
    plt.legend(loc='UpperRight', bbox_to_anchor=(1, 1))
    plt.title('Optimized Rolling Weight Given ')
    plt.show()

