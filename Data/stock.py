import math

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time


class Graph:

    def __init__(self, ticker, period=None):
        self.ticker = ticker
        self.period = period
        self.df = pd.read_csv(f'{self.ticker}.csv')
        if self.period is not None:
            self.df = self.df.tail(self.period)
        self.df['Avg Price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['Index'] = np.arange(len(self.df))
        self.fig = go.Figure(
            data=go.Candlestick(open=self.df['Open'], close=self.df['Close'], high=self.df['High'], low=self.df['Low'],
                                name=''))
        self.fig.update_layout(xaxis_rangeslider_visible=False)

    def download(self):
        if self.period is None:
            tag = 'MAX'
        else:
            tag = self.period
        self.fig.write_image(f'{self.ticker}{tag}.png', width=3840, height=2160, scale=2)

    def display(self):
        self.fig.update_layout(xaxis_rangeslider_visible=True)
        self.fig.show()
        self.fig.update_layout(xaxis_rangeslider_visible=False)

    # ========== #
    # INDICATORS #
    # ========== #

    def add_sma(self, period=20):
        sma = self.df['Avg Price'].rolling(window=period).mean()
        self.fig.add_trace(go.Scatter(y=sma, name=f'SMA{period}', line_color='#000000'))

    def add_ema(self, period=20):
        df = self.df.sort_index()
        ema = df['Avg Price'].ewm(span=period, min_periods=0, adjust=False, ignore_na=False).mean()
        self.fig.add_trace(go.Scatter(y=ema, name=f'EMA{period}', line_color='#6a0dad'))

    def add_bb(self, period=20, deviations=2):
        ma = self.df['Avg Price'].rolling(window=period).mean()
        std_dev = self.df['Avg Price'].rolling(window=period).std()
        self.fig.add_trace(go.Scatter(y=ma + (deviations * std_dev), name=f'BB Upper{period}', line_color='#7fe5f0'))
        self.fig.add_trace(go.Scatter(y=ma - (deviations * std_dev), name=f'BB Lower{period}', line_color='#7fe5f0'))

    def add_rollingvwap(self, period=10):
        rpv = (self.df['Avg Price'] * self.df['Volume']).rolling(window=period).sum()
        rv = self.df['Volume'].rolling(window=period).sum()
        vwap = rpv / rv
        self.fig.add_trace(go.Scatter(y=vwap, name=f'Rolling VWAP{period}', line_color='#ef4135'))

    def add_linregress(self):
        cov = self.df.cov()['Index']['Avg Price']
        var = self.df.var()['Index']
        slope = cov / var
        intercept = self.df['Avg Price'].mean() - (self.df['Index'].mean() * slope)
        regression = self.df['Index'] * slope + intercept
        self.fig.add_trace(
            go.Scatter(y=regression, name=f'Linear Regression (y={round(slope, 4)}x + {round(intercept, 2)})',
                       line_color='#ffa500'))

    def spedscat(self):
        self.df['PC'] = self.df['Avg Price'].pct_change()
        adjuster = self.df['Volume'].max() / self.df['PC'].max()
        self.df['volp'] = self.df['Volume']/adjuster

        print(adjuster)

        fig = go.Figure(data=go.Scatter(x=self.df['PC'], y=self.df['volp'], mode='markers'))
        fig.update_layout(
            xaxis_title=self.ticker,
            yaxis_title=f'volume / adj. factor ({round(adjuster)})',
        )
        cov = self.df.cov()['volp']['PC']
        var = self.df.var()['volp']
        slope = cov / var
        intercept = self.df['volp'].mean() - (self.df['PC'].mean() * slope)
        regression = self.df['PC'] * slope + intercept
        print(f'{self.ticker} y={round(slope,6)}x + {round(intercept,4)}')
        print(f'{self.ticker} correlation {self.df.corr()["PC"]["volp"]}')
        self.fig.add_trace(
            go.Scatter(y=regression, name=f'Linear Regression (y={round(slope, 6)}x + {round(intercept, 2)})',
                       line_color='#ffa500'))
        fig.write_image(f'{self.ticker}adjVolcorr.png')

########################################################################################################################


class Correlate:

    def __init__(self, ticker1, ticker2, period=None):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.period = period

        self.df1 = pd.read_csv(f'{self.ticker1}.csv')
        if self.period is not None:
            self.df1 = self.df1.tail(self.period)
        self.df1['Avg Price'] = (self.df1['High'] + self.df1['Low'] + self.df1['Close']) / 3
        self.df2 = pd.read_csv(f'{self.ticker2}.csv')
        if self.period is not None:
            self.df2 = self.df2.tail(self.period)
        self.df2['Avg Price'] = (self.df2['High'] + self.df2['Low'] + self.df2['Close']) / 3
        column = ['Date']
        self.pchange = pd.DataFrame(columns=column)
        temp1 = self.pchange
        temp2 = temp1
        temp1['Date'] = self.df1['Date']
        temp2['Date'] = self.df2['Date']
        self.pchange[self.ticker1] = self.df1['Avg Price'].pct_change()
        temp2[self.ticker2] = self.df2['Avg Price'].pct_change()
        self.pchange.merge(temp2, on='Date')

    def get_scatplot(self, costand=0):
        fig = go.Figure(data=go.Scatter(x=self.pchange[self.ticker1], y=self.pchange[self.ticker2], mode='markers'))
        cov = self.pchange.cov()[self.ticker1][self.ticker2]
        var = self.pchange.var()[self.ticker1]
        slope = cov / var
        intercept = self.pchange[self.ticker2].mean() - (self.pchange[self.ticker1].mean() * slope)
        regression = self.pchange[self.ticker1] * slope + intercept
        fig.add_trace(
            go.Scatter(y=regression, x=self.pchange[self.ticker1],
                       name=f'Linear Regression (y={round(slope, 4)}x + {round(intercept, 4)})',
                       line_color='#ffa500'))
        fig.update_layout(
            xaxis_title=self.ticker1,
            yaxis_title=self.ticker2,
        )

        if costand > 0:
            costd = math.sqrt(cov)
            fig.add_trace(go.Scatter(y=regression+costd, x=self.pchange[self.ticker1]))
            fig.add_trace(go.Scatter(y=regression-costd, x=self.pchange[self.ticker1]))
        fig.write_image(f'{self.ticker1}_{self.ticker2}_corr.png')

    def get_corr(self):
        return self.pchange.corr()[self.ticker1][self.ticker2]

stonks = ['^DJI', '^GSPC', '^IXIC', '^NYA']
#for s in stonks:
#    for x in stonks:
#        a = Correlate(s, x)
#        a.get_scatplot()
for tick in stonks:
    a = Graph(ticker=tick)
    a.spedscat()