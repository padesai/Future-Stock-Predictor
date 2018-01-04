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
import matplotlib.pyplot as plt


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, leaf_size=6,bags=20):
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.bags = bags
        self.learner = bl.BagLearner(learner=bl.RTLearner, kwargs = {'leaf_size': self.leaf_size}, bags = self.bags, boost=False,verbose=self.verbose);

    def compute_trainX(self,prices,syms,lookback):

        pd.options.mode.chained_assignment = None;

        sma = pd.rolling_mean(prices,window=lookback,min_periods=lookback)
        bbp = prices.copy()

        rolling_std = pd.rolling_std(prices,window=lookback,min_periods=lookback)
        top_band = sma + (2*rolling_std)
        bottom_band = sma - (2*rolling_std)

        bbp = (prices - bottom_band) / (top_band - bottom_band)
        trainX = bbp.copy();

        sma = prices / sma

        rsi = prices.copy();

        daily_rets = prices.copy();
        daily_rets.values[1:,:] = prices.values[1:,:] - prices.values[:-1,:]
        daily_rets.values[0,:] = np.NaN

        for day in range(prices.shape[0]):
            up_gain = daily_rets.ix[day-lookback+1:day+1,:].where(daily_rets >= 0).sum()
            down_loss = -1*daily_rets.ix[day-lookback+1:day+1,:].where(daily_rets < 0).sum()
            rs = (up_gain/lookback) / (down_loss/lookback)
            rsi.ix[day,:] = 100 - (100 / (1 + rs))
            rsi[rsi == np.inf] = 100

        up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
        down_rets = -1*daily_rets[daily_rets < 0].fillna(0).cumsum()

        up_gain = prices.copy()
        up_gain.ix[:,:] = 0
        up_gain.values[lookback:,:] = up_rets.values[lookback:,:] - up_rets.values[:-lookback,:]

        down_loss = prices.copy()
        down_loss.ix[:,:] = 0
        down_loss.values[lookback:,:] =  down_rets.values[lookback:,:] - down_rets.values[:-lookback,:]

        rs = (up_gain / lookback) / (down_loss / lookback)
        rsi = 100 - (100 / (1 + rs))
        rsi.ix[:lookback,:] = np.nan

        rsi[rsi == np.inf] = 100;

        spy_rsi = rsi.copy()
        spy_rsi.values[:,:] = spy_rsi.ix[:,['SPY']]

        sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
        sma_cross[sma >= 1] = 1

        sma_cross[1:] = sma_cross.diff()
        sma_cross.ix[0] = 0
        #trainX['sma_cross'] = sma_cross[syms];
        #trainX['rsi'] = rsi[syms];
        trainX['sma'] = sma[syms];

        trainX['bbp'] = bbp[syms];

        #trainX['spy_rsi'] = spy_rsi['SPY'];
        trainX['momentum'] = (prices[syms] / prices[syms].shift(lookback-1)) - 1
        trainX['volatility'] = pd.rolling_std(daily_rets[syms], window=lookback, min_periods=lookback);
        trainX.drop(syms, axis=1, inplace=True)
        trainX.drop('SPY', axis=1, inplace=True)

        # f, axarr = plt.subplots(2, sharex=True)
        # axarr[0].plot(trainX.index.values, trainX['momentum'])
        # axarr[0].set_title('Sharing X axis')
        # axarr[1].scatter(trainX.index.values, prices[syms])

        trainX['bbp'] = (trainX['bbp'] - np.nanmean(trainX['bbp'])) / np.std(trainX['bbp'])
        trainX['sma'] = (trainX['sma'] - np.nanmean(trainX['sma'])) / np.std(trainX['sma'])
        trainX['momentum'] = (trainX['momentum'] - np.nanmean(trainX['momentum'])) / np.std(trainX['momentum'])
        trainX['volatility'] = (trainX['volatility'] - np.nanmean(trainX['volatility'])) / np.std(trainX['volatility'])

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

        st_dt_for_Y = sd
        ed_dt_for_Y = ed + dt.timedelta(days=40)

        dates_Y = pd.date_range(st_dt_for_Y, ed_dt_for_Y);
        prices_all_Y = ut.get_data(syms, dates_Y)
        prices_Y = prices_all_Y[syms];

        normalized_prices_period = prices_Y.div(prices_Y.ix[0], axis='columns');
        normalized_allocations_period = normalized_prices_period.mul(1.0, axis='columns');
        position_values_period = normalized_allocations_period.mul(sv, axis='columns');

        cum_ret = position_values_period.shift(13) / position_values_period

        cr_period = prices_all[[symbol, ]]
        cr_period.rename(columns={symbol: 'Return'}, inplace=True)

        cr_period.ix[:,:] = np.NaN
        cr_period['Return'] = cum_ret;

        cr_period.ix[trainX.shape[0]-period:trainX.shape[0]] = np.NaN;

        orders = prices.copy();
        orders.ix[:,:] = np.NaN

        # shares = 0
        #
        max_return = cr_period['Return'].ix[np.argmax(cr_period['Return'].ix[:])]
        min_return = cr_period['Return'].ix[np.argmin(cr_period['Return'].ix[:])]

        min_momentum = trainX['momentum'].ix[np.argmin(trainX['momentum'].ix[:])]
        max_momentum = trainX['momentum'].ix[np.argmax(trainX['momentum'].ix[:])]

        min_sma = trainX['sma'].ix[np.argmin(trainX['sma'].ix[:])]
        max_sma = trainX['sma'].ix[np.argmax(trainX['sma'].ix[:])]

        #& (cr_period['Return'] >= 0.3 * max_return)
        #& (cr_period['Return'] <= 0.3 * min_return)

        #  | ((orders.cumsum() < 200) & (cr_period['Return'] >= 0.625*max_return))

        orders[(trainX['sma'] < -1.39) | (trainX['bbp'] < -1.48) | (trainX['momentum'] <= 0.9945*min_momentum) ] = 1
        orders[(trainX['sma'] > 1.429) | (trainX['bbp'] > 1.49) | (trainX['momentum'] >= 0.9945*max_momentum)] = -1


        # f, axarr = plt.subplots(2, sharex=True)
        # axarr[0].plot(trainX.index.values, trainX['sma'])
        # axarr[0].set_title('Sharing X axis')
        # axarr[1].scatter(trainX.index.values, prices)

        #orders[(cr_period['Return'] >= 0.95* max_return)] = 1
        #orders[(cr_period['Return'] <= 0.95*min_return)] = -1
        # orders[(cr_period['Return'] >= 0.85*max_return)] = 30
        # orders[(cr_period['Return'] <= 0.8*min_return)] = -30
        # orders[(cr_period['Return'] > 0.6*max_return) & (cr_period['Return'] < 0.85*max_return)] = 5;
        # orders[(cr_period['Return'] < 0.6*min_return) & (cr_period['Return'] > 0.8*min_return)] = -5
        #
        # orders[(trainX['sma'] < 0.95) & (trainX['bbp'] < 0) & (trainX['rsi'] < 30) & (trainX['spy_rsi'] > 30)] = 200
        # orders[(trainX['sma'] > 1.05) & (trainX['bbp'] > 1) & (trainX['rsi'] > 70) & (trainX['spy_rsi'] < 70)] = -200
        #
        # orders[(trainX['sma_cross'] != 0)] = 0

        # orders.ffill(inplace=True)
        # orders.fillna(0,inplace=True)
        #
        # orders[1:] = orders.diff()
        # orders.ix[0] = 0
        #
        # del orders['SPY']


        # for i in range(0,orders.shape[0]):
        #     if ((trainX['sma'].ix[i] < 0.95) & (trainX['bbp'].ix[i] < 0) & (trainX['rsi'].ix[i] < 30) & (trainX['spy_rsi'].ix[i] > 30)) | ((max_return > 0) & (cr_period['Return'].ix[i] >= 0.625 * max_return)):
        #         if (shares < 200):
        #             orders.ix[i,0] = 200;
        #             shares += 200
        #         else:
        #             orders.ix[i,0] = 0;
        #
        #     if ((trainX['sma'].ix[i] > 1.05) & (trainX['bbp'].ix[i] > 1) & (trainX['rsi'].ix[i] > 70) & (trainX['spy_rsi'].ix[i] < 70)) |((min_return < 0) & (cr_period['Return'].ix[i] <= 0.6 * min_return)):
        #         if (shares > -200):
        #             orders.ix[i,0] = -200
        #             shares -= 200
        #         else:
        #             orders.ix[i,0] = 0;

        orders[np.isnan(orders.ix[:,0])] = 0
        orders_array = orders.ix[:,0].as_matrix()

        self.learner.addEvidence(trainX_array,orders_array);

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "AAPL", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 100000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
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

        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
