"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import BagLearner as bl
import random
import numpy as np
import math


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.learner = bl.BagLearner(learner=bl.RTLearner, kwargs = {'leaf_size': 5}, bags = 30, boost=False,verbose=self.verbose);

    def compute_trainX(self,prices,syms,lookback):

        pd.options.mode.chained_assignment = None;

        sma = pd.rolling_mean(prices,window=lookback,min_periods=lookback)

        rolling_std = pd.rolling_std(prices,window=lookback,min_periods=lookback)
        top_band = sma + (2*rolling_std)
        bottom_band = sma - (2*rolling_std)

        bbp = (prices - bottom_band) / (top_band - bottom_band)
        trainX = bbp.copy();

        sma = prices / sma

        daily_rets = prices.copy();
        daily_rets.values[1:,:] = prices.values[1:,:] - prices.values[:-1,:]
        daily_rets.values[0,:] = np.NaN

        sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
        sma_cross[sma >= 1] = 1

        sma_cross[1:] = sma_cross.diff()
        sma_cross.ix[0] = 0

        trainX['sma'] = sma[syms];
        trainX['bbp'] = bbp[syms];
        trainX['momentum'] = (prices[syms] / prices[syms].shift(lookback-1)) - 1
        trainX['volatility'] = pd.rolling_std(daily_rets[syms], window=lookback, min_periods=lookback);

        trainX['bbp'] = (trainX['bbp'] - np.nanmean(trainX['bbp'])) / np.std(trainX['bbp'])
        trainX['sma'] = (trainX['sma'] - np.nanmean(trainX['sma'])) / np.std(trainX['sma'])
        trainX['momentum'] = (trainX['momentum'] - np.nanmean(trainX['momentum'])) / np.std(trainX['momentum'])
        trainX['volatility'] = (trainX['volatility'] - np.nanmean(trainX['volatility'])) / np.std(trainX['volatility'])

        trainX.drop(syms, axis=1, inplace=True)

        if 'SPY' in trainX.columns:
            trainX.drop('SPY', axis=1, inplace=True)

        return trainX

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "AAPL", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 100000):

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        period = 19

        trainX = self.compute_trainX(prices_all,syms,period);
        trainX_array = trainX.as_matrix();

        orders = prices.copy();
        orders.ix[:,:] = np.NaN

        min_momentum = trainX['momentum'].ix[np.argmin(trainX['momentum'].ix[:])]
        max_momentum = trainX['momentum'].ix[np.argmax(trainX['momentum'].ix[:])]

        orders[(trainX['sma'] < -1.39) | (trainX['bbp'] < -1.48) | (trainX['momentum'] <= 0.9945*min_momentum)] = 1
        orders[(trainX['sma'] > 1.429) | (trainX['bbp'] > 1.49) | (trainX['momentum'] >= 0.9945*max_momentum)] = -1

        orders[np.isnan(orders.ix[:,0])] = 0
        orders_array = orders.ix[:,0].as_matrix()

        self.learner.addEvidence(trainX_array,orders_array);

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "AAPL", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 100000):

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        syms = [symbol]
        trades = prices_all[[symbol, ]]
        trades.rename(columns={symbol: 'trades'}, inplace=True)

        period = 19

        dataX = self.compute_trainX(prices_all,syms,period);
        dataX_array = dataX.as_matrix();

        trades_sample = trades;

        trades_sample['trades'] = self.learner.query(dataX_array)

        trades[trades_sample['trades'] == 1] = 200
        trades[trades_sample['trades'] == -1] = -200
        trades[trades_sample['trades'] == 0] = 0

        trades_holdings = trades['trades'].cumsum();

        for i in range(0,trades['trades'].shape[0]):
            if (trades_holdings.ix[i,0] > 200):
                trades['trades'].ix[i-1] = -200
                trades_holdings = trades['trades'].cumsum()

            elif (trades_holdings.ix[i,0] < -200):
                trades['trades'].ix[i-1] = 200
                trades_holdings = trades['trades'].cumsum()

        trades_holdings = trades['trades'].cumsum()

        non_zero_holdings = trades_holdings[trades_holdings != 0]

        if non_zero_holdings.shape[0] == 0:
            for j in range(0, trades['trades'].shape[0]):
                if (trades_sample['trades'].ix[j] == 1):
                    trades['trades'].ix[j] = 200;
                    break;
                elif (trades_sample['trades'].ix[j] == -1):
                    trades['trades'].ix[j] = -200
                    break;

        trades_holdings = trades['trades'].cumsum()

        non_zero_holdings = trades_holdings[trades_holdings != 0]

        if non_zero_holdings.shape[0] == 0:
            trades['trades'].ix[0] = -200;
            trades['trades'].ix[trades['trades'].shape[0]-1] = 200;

        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all

        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
